import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool , initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import os



# Steup streamlit app
st.set_page_config(page_title="GenieMind" , page_icon="🧠")
st.title("GenieMind 🧠 - Using Google Gemma2")

api_key=st.sidebar.text_input(label="Groq API key" , type="password")

if not api_key:
    st.info("Please add your Groq API key")
    st.stop()

os.environ['GROQ_API_KEY']=api_key

llm=ChatGroq(groq_api_key=api_key , model="gemma2-9b-it")


## Initialize the tool:

wiki_wrapper=WikipediaAPIWrapper()
wiki_tool=Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="A tool for searching the internet to find info about the topic"
)

##Initialize the math tool:

math_chain=LLMMathChain.from_llm(llm=llm)
cal_tool=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for solving the math problem"
)

##prompt:

prompt="""
   Your a agent tasked for solving user mathematical question , logically arrive at the solution
   provide a detail explanation and display it point wise , also answer the questions that user ask about 
   other topics and answers should be correct for the question below
   Question:{question}
   Answer:
"""

prompt_template=PromptTemplate(
    input_variables=['question'],
    template=prompt
)


## cobine all the tools into chain

chain=LLMChain(llm=llm , prompt=prompt_template)

reasoning_tool=Tool(
    name='reasoning tool' ,
    func=chain.run,
    description="Tool for answering logic based and reasoning question"
)

##initialize the agent

assistant_agent=initialize_agent(
    tools=[wiki_tool , cal_tool , reasoning_tool]  , ##LLM chain to tool 
    llm=llm , 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION , 
    verbose=True,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role" : "assistant"  , "content" : "Hi I am GenieMind who can answer your question. I am trained to solve Logical Math problems as well as the general questions asked by users"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


##function to create response: - depricated 

def gen_response(question):
    response=assistant_agent.invoke({'input' : question})
    return response


## INteraction:

question=st.text_area("Enter the question : ")
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating the response........."):
            st.session_state.messages.append({'role' : 'user' , 'content':question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container() , expand_new_thoughts=True)
            response=assistant_agent.run(st.session_state.messages , callbacks=[st_cb])
            
            st.session_state.messages.append({'role':'assistant' , 'content' : response})
            st.write('Response : ')
            st.success(response)
            


    
    else:
        st.warning("Enter the question")
        
    

