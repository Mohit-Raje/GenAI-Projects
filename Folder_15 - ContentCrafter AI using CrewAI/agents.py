from crewai import Agent, LLM
from tools import yt_tool
import os
# from langchain_groq import Chatgroq
from crewai_tools import YoutubeChannelSearchTool
from embedchain import App
from dotenv import load_dotenv
load_dotenv()
import litellm


# Set Groq API Key

OPENAI_API_KEY = "Enter OpenAI API Key"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['OPENAI_MODEL_NAME'] = "gpt-4-0125-preview"

# Initialize Chatgroq LLM
llm = LLM(
    model="groq/llama-3.1-70b-versatile",
    temperature=0.7
)



# Blog Researcher Agent
blog_researcher = Agent(
    role='Blog Researcher from Youtube Videos',
    goal='Get the relevant video transcription for the topic {topic} from the provided YT channel',
    verbose=True,
    memory=True,
    backstory="Expert in understanding videos in AI, Data Science, Machine Learning, and Gen AI.",
    tools=[yt_tool],
    llm=llm,
    allow_delegation=True
)

# Blog Writer Agent
blog_writer = Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the video {topic} from YT video',
    verbose=True,
    memory=True,
    backstory="Crafting engaging narratives to simplify complex topics in tech and AI.",
    tools=[yt_tool],
    llm=llm,
    allow_delegation=False
)

print("Agents initialized successfully.")