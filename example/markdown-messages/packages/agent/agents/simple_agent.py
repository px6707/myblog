from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

agent = create_agent(
    model=init_chat_model("anthropic:claude-haiku-4-5"),
)
