from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from tools.weather import get_weather
from tools.calculator import calculate
from tools.search_web import search_web

agent = create_agent(
    model=init_chat_model("anthropic:claude-haiku-4-5"),
    tools=[get_weather, calculate, search_web],
)
