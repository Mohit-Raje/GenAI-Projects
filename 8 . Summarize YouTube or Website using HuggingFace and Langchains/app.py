import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document  # Import Document for proper formatting
import yt_dlp
from langchain_huggingface import HuggingFaceEndpoint
import os

## Streamlit - App
st.set_page_config(page_title="Summarize YouTube or Website using HuggingFace and Langchain", page_icon="🦜")
st.title("Summarize YouTube or Website using HuggingFace and Langchain")
st.subheader("Summarize URL")

## Get the groq_api_key
with st.sidebar:
    hf_api_key = st.text_input(label="Enter HuggingFace API Key", type="password")
    os.environ['HF_TOKEN'] = hf_api_key

## Mistral Model using HF API
    if hf_api_key:
        repo_id="mistralai/Mistral-7B-Instruct-v0.3"

        llm= HuggingFaceEndpoint(
        repo_id=repo_id , 
        max_length=200 , 
        temperature=0.7 , 
        token=hf_api_key
        )

prompt_template = """
    Act as an expert in providing the summary of the content provided.
    Provide a summary of the following content in 400 words
    Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])



generic_url = st.text_input("URL", label_visibility="collapsed")

def get_youtube_transcript(url):
    """Extract transcript using yt_dlp"""
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'format': 'bestaudio/best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subtitles = info.get("subtitles", {})
        if "en" in subtitles:
            return "Transcript fetched successfully. However, subtitle downloading requires manual handling."
        else:
            return info.get("description", "No subtitles or transcript available.")

if st.button("Summarize the content from YouTube or Website"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please enter valid information :(")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Waiting......"):
                
                if "youtube.com" in generic_url:
                    
                    content = get_youtube_transcript(generic_url)
                    docs = [Document(page_content=content)] 
                    
                else:
                    from langchain_community.document_loaders import UnstructuredURLLoader
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False)
                    docs = loader.load()
                
                # Chain for Summarization
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)  # Pass documents to chain
                st.success(output_summary)
        
        except Exception as e:
            st.exception(f"Exception: {e}")
