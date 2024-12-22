from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chains import RetrievalQA
import cassio

# Initialize environment variables for API keys
ASTRA_DB_APPLICATION_TOKEN = ""
ASTRA_DB_ID = ""
HF_API_KEY = ""
os.environ['HF_API_KEY'] = HF_API_KEY
api_key = ""
os.environ['GROQ_API_KEY'] = api_key

# Initialize the LLM
llm = ChatGroq(groq_api_key=api_key, model="gemma2-9b-it")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
        Answer the question based on the provided context and add a few more things related to it.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        </context>
        Question:{input}
    """
)

# Initialize Astra DB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name='qa_mini_demo',
    session=None,
    keyspace=None
)

# Main function for the application
def execute():
    # Data Ingestion from PDF
    pdfreader = PdfReader('budget_speech.pdf')
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # Text Splitting
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    text = text_splitter.split_text(raw_text)

    # Dump the text chunks into Astra DB
    astra_vector_store.add_texts(text)
    st.write("All text inserted")

    # Wrap it up
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Streamlit app
st.title("Budget 2024-25 RAG Document Q&A With Gemma2 And AstraDB")
st.write("Please click the button below to do the document embedding and storing it in Astra DB")

if st.button("Document Embeddings"):
    execute()
    st.write("Vector Database is ready")

user_prompt = st.text_input("Enter your query from the Budget 2024-25")
if user_prompt:
    try:
        retriever = astra_vector_store.as_retriever(k=4)
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = chain.run(user_prompt).strip()
        st.write(answer)

    except Exception as e:
        st.write(f"Error occurred: {e}")

