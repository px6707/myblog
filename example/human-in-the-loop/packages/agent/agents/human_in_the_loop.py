from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from tools.email import send_email

agent = create_agent(
    model=init_chat_model("anthropic:claude-haiku-4-5"),
    tools=[send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Review email before sending",
                },
            }
        ),
    ],
)
