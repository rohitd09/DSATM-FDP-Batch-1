from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv

load_dotenv()

class TeachingAssistantAgent:
    def __init__(self):
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def run_llm(query: str) -> str:
        pass

if __name__ ==  "__main__":
    query = input("Enter your query: ")
    print(run_llm(query))