import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_csv_agent
import streamlit as st

# Set up local storage directory
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Streamlit setup
st.set_page_config(page_title="Chat with CSV", page_icon="🦜")
st.title("Effortless Data Analysis 📊: Ask Questions and Get Insights from Your CSV 📂")

# Sidebar for API key
with st.sidebar:    
    GROQ_API_KEY = st.text_input(label="Enter your GROQ API KEY", type="password") 
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY
    # Initialize the model 
    if GROQ_API_KEY:
        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="gemma2-9b-it")

# Prompt template
prompt_template = """
Act as a Professional data scientist/analyst and answer the question asked by human.
If the question is about prediction of some event, answer that wisely. if the output is categorical then ensure u give some statistical info assicaited with , u r a professional data scientist
ensure one thing your give the output in proper formatted way , it shoud be visually appealing , do not attempt to draws any plots but give the outputs in form of table when needed .
Question: {question}
"""

# File upload feature
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
csv_path = None

if uploaded_file:
    # Save the file locally
    csv_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded and saved as {uploaded_file.name}")

# CSV Agent function
def CSV_Agent(question):
    if not csv_path:
        st.error("No CSV file provided. Please upload a file to proceed.")
        return None

    # Add the question to the custom prompt
    custom_prompt = prompt_template.format(question=question)

    # Create the CSV agent
    agent_executer = create_csv_agent(
        llm,
        csv_path, 
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True
    )
    
    # Invoke the agent with the custom prompt
    return agent_executer.invoke(custom_prompt)    

# Question input and response display
question = st.text_input("Enter your question:")

if st.button("Ask Question"):
    if csv_path:
        response = CSV_Agent(question)
        if response:
            # Extract the "output" part of the response
            final_output = response.get("output", "No output received.")
            
            # Clean the output (example: remove unnecessary lists or dicts)
            if isinstance(final_output, str) and "[" in final_output:
                # Format list-like response
                formatted_output = final_output.replace("[", "").replace("]", "").replace("nan", "null").strip()
            else:
                formatted_output = final_output
            
            st.markdown(f"### Answer:\n{formatted_output}")
        else:
            st.warning("No response received from the agent.")
    else:
        st.error("Please upload a CSV file to ask questions.")
