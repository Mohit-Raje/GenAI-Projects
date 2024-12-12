import streamlit as st 
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import os


st.set_page_config(page_title="Langchain powered solution for chatting with SQL Database - Online Retail Store")
st.title("🦜 Langchain powered solution for chatting with SQL Database - Online Retail Store")

LOCALDB="USE_LOCALDB"

radio_opt=["SQLite3 Database : OnlineRetailStore.db"]

select_opt=st.sidebar.radio(label="Database for Chatting" , options=radio_opt)


if radio_opt.index(select_opt)==0:
    db_uri=LOCALDB
    
api_key=st.sidebar.text_input(label="Enter the Groq API key" , type="password")
os.environ['GROQ_API_KEY']=api_key


llm=ChatGroq(groq_api_key=api_key , model="Llama3-8b-8192")


@st.cache_resource(ttl="2h")
def config_db(db_uri):
    if db_uri==LOCALDB:
        dbfilepath=(Path(__file__).parent/"OnlineRetailStore.db").absolute()
        print(dbfilepath)
        creator= lambda : sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))

db=config_db(db_uri)

toolkit=SQLDatabaseToolkit(llm=llm , db=db)


agent=create_sql_agent(
    llm=llm ,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)


if "messages" not in st.session_state or st.sidebar.button("Clear Message history"):
    st.session_state["messages"] = [{'role':'assistant' , 'content':'How can I help you'}]
    
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg["content"])


user_query=st.chat_input(placeholder="Ask anything to the database")

if user_query:
    st.session_state.messages.append({'role':'user' , 'content':user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message("assistant"):
        streamlit_callbacks=StreamlitCallbackHandler(st.container())
        response=agent.run(user_query , callbacks=[streamlit_callbacks])
        st.session_state.messages.append({'role':'assistant' , 'content':response})
        st.write(response)
        
