import os

import streamlit as st

#os.environ["GOOGLE_API_KEY"] = "AIzaSyD-sWQnBESucuj6WBnuZdTsXGpTmfM3-Hc"
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")
#result = llm.invoke("네이버에 대해 보고서를 작성해줘")
#print(result.content)

#from langchain.chains import RetrievalQA
#from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

st.write("---")
st.title("ChatPDF기반 Q&A시스템입니다")
# 파일 업로드
#uploaded_files = st.file_uploader("PDF 파일을 올려주세요", type = ['pdf'])
loader = PyPDFLoader("C:/Users/beomg/PycharmProjects/pythonProject5/unsu.pdf")
st.write("---")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(pages)

from langchain_community.embeddings import HuggingFaceEmbeddings
model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':True}
hf = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

docsearch = Chroma.from_documents(texts, hf)

retriever = docsearch.as_retriever(
    search_type = "mmr",
    search_kwargs = {'k':3, 'fetch_k':10}
)
#result_01 = retriever.get_relevant_documents('question_01')
#print(result)

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

template = """Answer the question as based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

gemini = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0)

chain = RunnableMap({
    "context": lambda x : retriever.get_relevant_documents(x['question']),
    "question":lambda x : x['question']
}) | prompt | gemini

st.header("PDF에게 질문해 보세요")
query_01 = st.text_input("질문을 입력하세요")
if st.button('질문하기'):
    with st.spinner('wait for it'):
        result_01 = retriever.get_relevant_documents(query_01)
        result_02 = chain.invoke({'question': query_01}).content

        st.write(result_01)
        st.write(result_02)