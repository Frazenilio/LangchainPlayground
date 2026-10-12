import array
import enum
from json import tool
from warnings import deprecated

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

class Models(enum.Enum):
    LLAMA = "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:latest"
    QWEN = "qwen3.5:0.8b"
    GEMMA = "gemma3:1b"

SYS_PROMPT: str = """
    You are an helpful weather assitant who is specialized in returning the weather of a city
"""

class OllamaAgent():
    """
        Class for handling an agent running in Ollama local client
    """

    def build_agent(self, model: Models, custom_sys: str = "", tools = []):# 
        """
            Build and return an agent with 0 temperature
        """
        llm = ChatOllama(model = model.value, temperature=0, )

        agent = create_agent(llm, tools=tools if tools else [], \
            system_prompt= custom_sys if custom_sys else SYS_PROMPT)
        
        return agent

    def __init__(self, model: Models, custom_sys: str = "", tools = []) -> None:
        self.agent = self.build_agent(model, custom_sys, tools)

    def askLLM(self, prompt: str) -> str:
        """
            Given a prompt, query the agent created in init with that prompt and returns the answer
        """
        message = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        result = self.agent.invoke(
            {
                "messages": message
            }
        )

        return result["messages"][-1].content


@deprecated("Use the OllamaAgent class instead. Create an agent with the method and query it")
def ollama_agent(model: Models, tools: array, prompt: str) -> str:
    """
        Old method for querying a model agent
    """
    llm = ChatOllama(model = model.value, temperature=0, 
                    )

    agent = create_agent(llm, tools=tools, system_prompt=SYS_PROMPT)

    message = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = agent.invoke(
        {
            "messages": message
        }
    )

    return response["messages"][-1].content