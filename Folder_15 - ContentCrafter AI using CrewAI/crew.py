from crewai import Crew , Process
from agents import blog_researcher , blog_writer
from tasks import research_task , write_task
# from langchain_groq import Chatgroq
from dotenv import load_dotenv
load_dotenv()




crew=Crew(
    agents=[blog_researcher , blog_writer],
    tasks=[research_task , write_task],
    process=Process.sequential,
    memory=True , 
    cache=True ,
    max_rpm=100,
    share_crew=True
)

result=crew.kickoff(inputs={'topic': 'AI VS ML VS DL VS Data Science'})
print(result)