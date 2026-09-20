"""
Human-in-the-loop with per-tool interrupt forms.

Unlike `human_in_the_loop` (which uses HumanInTheLoopMiddleware to wrap a
single tool with a generic approve/edit/reject card), this agent raises
`interrupt()` from within each tool. Every interrupt carries a "card" that
describes the exact form the frontend should render — so each of the three
tools surfaces a different UI.

The flow mirrors the customer pattern:
 1. The tool builds a "card" (form spec + context) and `interrupt()`s with it.
 2. The frontend renders the matching form, then resolves the interrupt AND
    pushes the card into state in the *same* superstep via
    `respond(decision, { update: { messages: [card] } })`. The backend never
    writes the card, so it never flickers or disappears while the (slow) tool
    business logic runs.
 3. On reject the tool short-circuits with a tool result; on approve it runs
    its (simulated slow) business logic and returns the result — never the
    card, which the frontend already committed.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from tools.hitl_interrupt_forms import book_flight, issue_refund, publish_post

SYSTEM_PROMPT = """You help users perform sensitive actions that each require human review.

You have three tools:
- "book_flight": reserve a flight. Extract origin, destination, date and passenger count from the request.
- "issue_refund": refund a customer order. Extract the order id, amount, currency and reason.
- "publish_post": publish a social post. Extract the platform and draft the post content.

Pick exactly the tool that matches the user's request and call it with your best-guess arguments — a human will confirm or adjust the details before anything happens. After the tool returns, briefly summarize the outcome for the user."""

agent = create_agent(
    model=init_chat_model("anthropic:claude-haiku-4-5"),
    tools=[book_flight, issue_refund, publish_post],
    system_prompt=SYSTEM_PROMPT,
)
