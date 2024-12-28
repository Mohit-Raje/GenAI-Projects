import streamlit as st 
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings , ChatNVIDIA 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

os.environ['NVIDIA_API_KEY']="Enter Your API Key"

llm=ChatNVIDIA(model_name = "mistralai/mistral-7b-instruct-v0.3")  ##NVIDIA NIM Inferencing


def vector_embedding():
    
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("/us_census")
        st.seesion_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700 , chunk_overlap=50)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents , st.session_state.embeddings)
        
st.title("Nvidia NIM RAG Application")

prompt=ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

prompt1=st.text_input("Enter Your Question From Doduments")

if st.button("Documnet Embedding"):
    vector_embedding()
    st.write("Documents are embedded and stored into FAISS database")

if prompt1:
    document_chain=create_stuff_documents_chain(llm , prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever , document_chain)
    response=retrieval_chain.invoke({'input':prompt1})
    st.write(response['answer'])
    
    