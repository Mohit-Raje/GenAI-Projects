import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain_groq import ChatGroq


# Setup Streamlit app:
st.set_page_config(page_title="Code Assistant 🤖", page_icon="🤖💜")
st.title("Code Assistant using Llama3 🤖📘")

with st.sidebar:
    groq_api_key = st.text_input(label="Enter the Groq API key", type="password")
    os.environ['GROQ_API_KEY'] = groq_api_key

    if groq_api_key:
        llm=ChatGroq(groq_api_key=groq_api_key , model='llama3-8b-8192')
        

# Define the prompt template using PromptTemplate
prompt_template = PromptTemplate(
    template="""
    You are an expert coder and act as a coding teacher/expert to answer the question given by the user.
    Write the code in detail along with an explanation of the code in the form of comments only 
    so that it is easy to copy paste the code, and write the code in the language. 
    Try to seperate the code from its explanation so that it becomes easy for the end use to copy and paste.
    Your name is CodeSage and you are developed by Mohit a AI Engineer , if asked then say this in a creative way
    Question: {question}
    """,
    input_variables=["question"]
)

# Input for coding question
coding_question = st.text_area("Enter your coding question:", placeholder="Write your coding question here...")

if st.button("Generate Code"):
    if not groq_api_key.strip() or not coding_question.strip():
        st.error("Please provide the API key and enter a question to get started.")
    else:
        try:
            with st.spinner("Generating code, please wait..."):
                # Initialize LLMChain with the correct prompt and LLM
                chain = LLMChain(llm=llm, prompt=prompt_template)
                output = chain.run({"question": coding_question})
                st.code(output, language='python')
        except Exception as e:
            st.error(f"An error occurred: {e}")




## write a program using python to display the fibonacci series