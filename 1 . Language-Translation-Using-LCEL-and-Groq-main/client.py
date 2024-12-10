import requests
import streamlit as st

def get_response(input_text):
    # Corrected JSON structure with "input" field
    json_body = {
        "input": {
            "language": "French",
            "text": input_text  # Dynamically use the input text
        },
        "config": {},
        "kwargs": {}
    }

    # Use the `json` keyword to pass JSON data
    response = requests.post("http://127.0.0.1:8001/chain/invoke", json=json_body)

    # Check for HTTP errors
    if response.status_code != 200:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None

    # Return the parsed JSON response
    return response.json()

# Streamlit app
st.title("LLM Application Using LCEL")
input_text = st.text_input("Enter the text you want to convert to French")

if input_text:
    response_data = get_response(input_text)
    if response_data:
        st.write("Response:")
        st.write(response_data)
