import streamlit as st
import boto3
import json
from langchain.prompts import PromptTemplate

# Define the PromptTemplate
prompt_data = """
    Act as an expert who can summarize the text and give important points out of it.
    Text : {text}
"""
prompt = PromptTemplate(template=prompt_data, input_variables=["text"])

# Initialize the Bedrock Runtime client
bedrock = boto3.client('bedrock-runtime')

# Define the inference profile ARN
model_arn = "Enter your ARN here"  

# Streamlit app
st.title("Text Summarization App")

# Text input from user
user_input = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if user_input.strip():
        # Define the payload
        payload = {
            "prompt": prompt.format(text=user_input),
            "max_gen_len": 512,
            "temperature": 0.5,
            "top_p": 0.9
        }

        # Convert the payload to JSON
        body = json.dumps(payload)

        try:
            # Invoke the model
            response = bedrock.invoke_model(
                modelId=model_arn,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            # Extract and display the response
            response_body = json.loads(response.get("body").read())
            response_text = response_body['generation']
            st.subheader("Summary:")
            st.write(response_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to summarize.")
