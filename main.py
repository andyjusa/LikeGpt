from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader,UnstructuredWordDocumentLoader,UnstructuredPowerPointLoader,RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import json
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
import os
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_community.chat_models import ChatOllama,ChatHuggingFace
from langchain_community.llms import HuggingFacePipeline
import ollama

os.environ['HF_HOME'] = "/Users/jeongjaemin/Desktop/LikeGpt/models"

DATASET_PATH = "db/"
CHAT_PATH = "chats/"
TEMP = 0.1

embed_model_en = OllamaEmbeddings(model="nomic-embed-text",show_progress=True) #For English
embed_model_ko = HuggingFaceBgeEmbeddings( #For Korean
    model_name="/Users/jeongjaemin/Desktop/ko-sroberta-nli",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
) #For korean

embedModel = lambda x: embed_model_ko if x else embed_model_en

ollamaModelLoad = lambda : tuple([i['name'] for i in ollama.list()['models']])

def mkdir(s):
    if f"{s}" not in os.listdir("./"):
        os.mkdir(f"./{s}")

if "db" not in os.listdir("./"):
    os.mkdir("./db")

def initChatData(s):
    with open(f"{CHAT_PATH}{s}.json") as f:
        data = json.load(f)
    return chatData(data["name"], data["langModelName"],data["isOllama"],data["dataSet"],data["messages"],data
                    ["chatHistory"])

class chatData:
    def __init__(self,name,langModel,isOllama,dataSet,messages,history):
        self.name:str = name
        self.langModelName:str = langModel
        self.isOllama:bool = isOllama
        self.dataSet:str = dataSet
        self.messages:list = messages
        self.chatHistory:list = history
    def exportToDict(self):
        return {
            "name":self.name,
            "langModelName":self.langModelName,
            "isOllama":self.isOllama,
            "dataSet":self.dataSet,
            "messages":self.messages,
            "chatHistory":self.chatHistory
            }
    def messageAppend(self,message):
        self.messages.append(message)
        return self
    
    def historyAppend(self,history):
        for i in history:
            if type(i) is AIMessage:
                self.chatHistory.append(["ai",i.content])
            if type(i) is HumanMessage:
                self.chatHistory.append(["human",i.content])
        return self
    def exportHistory(self):
        return [(AIMessage(content=i[1]) if i[0]== "ai" else HumanMessage(content=i[1]))for i in self.chatHistory]


def main():
    st.session_state.chatList = os.listdir(CHAT_PATH)
    resetValue("title","Null")
    resetValue("chatList",list())
    resetValue("messages",[{"role": "assistant", 
                                        "content": "안녕하세요! 채팅방을 설정하여 채팅을 시작하세요."}])
    resetValue("conversation",None)
    resetValue("prevTitle","Null")
    resetValue("chat_history",None)

    st.set_page_config(
        page_title="Chat",
        page_icon=":books:")
    st.title(st.session_state.title)

    with st.sidebar:
        tab1,tab2 = st.tabs(["dataSet","chatList"])
        with tab1:
            uploadedFiles = st.file_uploader("Upload your files",type=['pdf,docx,pptx'],accept_multiple_files=True)
            num = st.number_input("Number of URLs", 0)
            left_column, right_column = st.columns(2)
            scan = left_column.toggle("Scan")
            double_scan = right_column.toggle("Double Scan") if scan else None
            condition = left_column.text_input("Condition", "") if scan else None
            form = right_column.text_input("Form (add \{input\})", "") if scan else None
            urls = {f"URL {i}": st.text_input(f"URL {i}") for i in range(num)}
            col1,col2 = st.columns([17,3])
            modelName = col1.text_input("dataName")
            col2.text("korean")
            korean = col2.toggle(" ")
            if st.button(" export "):
                files_text = getText(uploadedFiles,scan,double_scan,condition,form,urls)
                text_chunks = getTextChunk(files_text)
                lang = "kr" if korean else "en"
                get_vectorstore(text_chunks,embedModel(korean),f"{DATASET_PATH}{modelName}_{lang}")

        with tab2:
            for i in st.session_state.chatList:
                if st.button(i.split('.')[0],use_container_width=True):
                    st.session_state.title = i.split('.')[0]
                    st.session_state.messages = initChatData(st.session_state.title).messages
                    st.rerun()
            with st.popover("\+",use_container_width=True):
                models = [i+";Ollama" for i in ollamaModelLoad()]
                dbs = [i for i in os.listdir(DATASET_PATH)]
                if ".DS_Store" in dbs:
                    dbs.remove(".DS_Store")
                chatName = st.text_input("chatName")
                dataSet = st.selectbox("Dataset",dbs)
                langModelName = st.selectbox("langModel",models)
                if st.button("add",use_container_width=True):
                    if f"{chatName}.json" not in os.listdir(CHAT_PATH):
                        with open(CHAT_PATH+f'{chatName}.json', 'w') as json_file:
                            json.dump(chatData(
                                chatName,
                                langModelName.split(';')[0],
                                (langModelName.split(';')[1] == "Ollama"),
                                dataSet,
                                [{"role": "assistant", 
                                    "content": "안녕하세요! 선택한 자료와 관련된 질문을 하세요."}]
                                ,[]).exportToDict(), json_file)
                        st.session_state.chatList = os.listdir(CHAT_PATH)
                    st.rerun()
    if st.session_state.title != "Null":
        d = initChatData(st.session_state.title).messages
    else:
        d = st.session_state.messages
    for message in d:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    query = st.chat_input("질문을 입력해주세요.")
    if query:
        if st.session_state.title == "Null":
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                st.markdown("You don't select model")
            st.stop()
        data = initChatData(st.session_state.title).messageAppend({"role": "user", "content": query})
        if st.session_state.prevTitle != st.session_state.title:
            st.session_state.conversation = get_conversation_chain(
                Chroma(
                    persist_directory=f"db/{data.dataSet}",
                    embedding_function=embedModel(str(data.dataSet).split('_')[1] == "kr")
                    ),
                    llm=ChatOllama(model=data.langModelName,temperature=TEMP) if data.isOllama else ""
                    )#not coded huggingface
            st.session_state.chat_history = data.exportHistory()
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                
                data = data.historyAppend(result['chat_history'])
                response = result['answer']
                source_documents = result['source_documents']
                st.markdown(response)
                try:
                    with st.expander("참고 문서 확인"):
                        st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                        st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                        st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                except IndexError:
                    pass
                data = data.messageAppend({"role": "assistant", "content": response})
            with open(CHAT_PATH+f'{st.session_state.title}.json', 'w') as json_file:
                json.dump(data.exportToDict(), json_file)
            st.session_state.chat_history = data.exportHistory()


def getText(datas,scan,double_scan,condition,form,url):
    doc_list = []
    for doc in datas:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
        if '.pdf' in doc.name:
            loader = UnstructuredPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = UnstructuredWordDocumentLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        doc_list.extend(documents)

        
    if scan and double_scan:
        a = list()
        for url in url.values():
            for i in urlScanning(url=url,condition=condition,form=form):
                for ii in urlScanning(url=i,condition=condition,form=form):
                    a.append(ii)  
        for i in a:
            loader = RecursiveUrlLoader(url=i,max_depth=2,extractor=lambda x: Soup(x,"html.parser").text)
            doc_list.extend(loader.load())
    elif scan:
        for url in url.values():
            for i in urlScanning(url=url,condition=condition,form=form):
                loader = RecursiveUrlLoader(url=i,max_depth=2,extractor=lambda x: Soup(x,"html.parser").text)
                doc_list.extend(loader.load())
    else:
        for url in url.values():
            loader = RecursiveUrlLoader(url=url,max_depth=2,extractor=lambda x: Soup(x,"html.parser").text)
            doc_list.extend(loader.load())
    return doc_list

def urlScanning(url,condition,form):
    a = list()
    a.append(url)
    for i in requests.get(url).text.split('\"'):
        if i[:len(condition)] == condition:
            a.append(form.format(input=i))
    return a

def get_vectorstore(text_chunks, embedding, path=None):
    try:
        if path is None:
            vectordb = Chroma.from_documents(text_chunks, embedding)
        else:
            vectordb = Chroma.from_documents(text_chunks, embedding, persist_directory=path)
        return vectordb
    except ValueError as e:
        error_msg = str(e)
        if "NoneType" in error_msg:
            filtered_text_chunks = filter_complex_metadata(text_chunks)
            return get_vectorstore(filtered_text_chunks, embedding, path)
        else:
            raise e
        
def getTextChunk(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def resetValue(key:str,value = None):
    if key not in st.session_state:
        st.session_state[key] = value

def get_conversation_chain(vetorestore,llm):
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain

if __name__ == '__main__':
    main()