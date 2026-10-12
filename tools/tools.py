import array

from langchain.tools import tool

import requests

@tool("get_weather", description="return weather information for a given city", return_direct=False)
def get_weather(city: str) -> str:
    respone = requests.get(f"https://wttr.in/{city}?format=j1")
    return respone.json()





def get_tools() -> array:
    return [get_weather]