import os
import json
import boto3
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Initialize Bedrock Runtime client
bedrock = boto3.client('bedrock-runtime')

# Define the model ARN (replace with your actual ARN)
model_arn = "Enter Your ARN Here"

# Initialize HuggingFace embeddings
huggingface_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Prompt template for question answering
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
100 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.Ensure u keep 
100-200 words explanations not more than that . 
Give examples if needed about cricket
Please provide dont know if u dont have the right context.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Data ingestion function
def data_ingestion():
    loader = PyPDFDirectoryLoader('data')
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Embedding and vector store creation
def embedding_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs, 
        huggingface_embeddings
    )
    vectorstore_faiss.save_local('faiss_index')

# Generate response from Bedrock model
def query_bedrock(prompt_data):
    payload = {
        "prompt": prompt_data,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    body = json.dumps(payload)

    response = bedrock.invoke_model(
        modelId=model_arn,
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response.get("body").read())
    return response_body['generation']

# Main Streamlit app function
def main():
    st.set_page_config("Chat PDF")
    st.header("Cricket Wisdom with Bedrock - A RAG Application 🏏🏟️")

    user_question = st.text_input('Ask your question about PDF Files')

    with st.sidebar:
        st.title("Update or create vector store")

        if st.button("Vector update"):
            with st.spinner("Processing"):
                docs = data_ingestion()
                embedding_vector_store(docs)
                st.success("Vector store updated successfully!")

    if st.button("Ask PDF"):
        with st.spinner("Processing..."):
            if user_question.strip():
                # Load the vector store
                faiss_index = FAISS.load_local(
                    "faiss_index", huggingface_embeddings, allow_dangerous_deserialization=True
                )

                # Retrieve relevant context
                context = faiss_index.similarity_search(user_question, k=3)
                context_text = "\n".join([doc.page_content for doc in context])

                # Format the prompt
                formatted_prompt = prompt.format(context=context_text, question=user_question)

                # Query Bedrock model
                answer = query_bedrock(formatted_prompt)

                st.text_area("Answer", answer, height=200)

if __name__ == "__main__":
    main()
