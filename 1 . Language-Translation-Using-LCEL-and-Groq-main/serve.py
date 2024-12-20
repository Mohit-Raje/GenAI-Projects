from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()

## Loading the model
os.environ["GROQ_API_KEY"] = "Enter Your key"
model=ChatGroq(model='gemma2-9b-it')

# Prompt template
system_template = "Translate the following to {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ('system' , system_template),
    ('user' , '{text}')
]
)

## Parser Initialization
parser=StrOutputParser()

#create chain

chain=prompt_template|model|parser

## App definition

app=FastAPI(title="Langchain Serve",
            version="1.0" , 
            description = "A simple API serve using Langchain runnable interfaces")

## Adding chain routes
add_routes(
    app , 
    chain,
    path='/chain'
)


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app , host="127.0.0.1" , port=8001)
