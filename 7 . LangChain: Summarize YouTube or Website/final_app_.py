import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document  # Import Document for proper formatting
import yt_dlp
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Streamlit - App
st.set_page_config(page_title="LangChain: Summarize YouTube or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize YouTube or Website")
st.subheader("Summarize URL")

## Get the groq_api_key
with st.sidebar:
    groq_api_key = st.text_input(label="Enter Groq API Key", type="password")
    os.environ['GROQ_API_KEY'] = groq_api_key

## Gemma Model using Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

prompt_template = """
    Act as an expert in providing the summary of the content provided.
    Provide a summary of the following content in 400 words
    Content: {text}
"""
map_reduce_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


final_prompt="""
    Based on the received input generate brief summary of the provided text,
    from the below text
    Text : {text}
"""
final_prompt_template=PromptTemplate(template = final_prompt  , input_variables=['text'])

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
    if not groq_api_key.strip() or not generic_url.strip():
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
                text_splitter=RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap=200)
                final_splits = text_splitter.split_documents(docs)
                chain = load_summarize_chain(llm=llm, 
                                             chain_type="map_reduce", 
                                             map_prompt=map_reduce_prompt , 
                                             combine_prompt=final_prompt_template)
                output_summary = chain.run(docs)  # Pass documents to chain
                st.success(output_summary)
        
        except Exception as e:
            st.exception(f"Exception: {e}")
