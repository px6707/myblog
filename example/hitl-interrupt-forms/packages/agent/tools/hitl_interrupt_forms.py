"""Per-tool interrupt forms for the hitl-interrupt-forms pattern."""

import random
import string
import time
from typing import Any

from langchain.tools import tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

BUSINESS_LOGIC_DELAY_S = 1.5


def _random_ref(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"{prefix}-{suffix}"


class BookFlightInput(BaseModel):
    origin: str = Field(description="Origin city or airport code")
    destination: str = Field(description="Destination city or airport code")
    date: str = Field(description="Travel date, e.g. 2026-07-14")
    passengers: int = Field(ge=1, description="Number of passengers")


class IssueRefundInput(BaseModel):
    orderId: str = Field(description="The order identifier, e.g. ORD-4821")
    amount: float = Field(description="Requested refund amount")
    currency: str = Field(default="USD", description="ISO currency code")
    reason: str = Field(description="Why the customer is requesting a refund")


class PublishPostInput(BaseModel):
    platform: str = Field(description="Target platform, e.g. X, LinkedIn")
    content: str = Field(description="The draft post content")


@tool(
    "book_flight",
    description="Book a flight for the user. Requires human confirmation of trip details.",
    args_schema=BookFlightInput,
)
def book_flight(origin: str, destination: str, date: str, passengers: int) -> str:
    decision: dict[str, Any] = interrupt({
        "formType": "flight-booking",
        "tool": "book_flight",
        "title": "Confirm flight booking",
        "context": {
            "origin": origin,
            "destination": destination,
            "date": date,
            "passengers": passengers,
        },
        "fields": [
            {
                "name": "seatClass",
                "label": "Seat class",
                "type": "select",
                "options": ["Economy", "Premium Economy", "Business"],
                "default": "Economy",
            },
            {
                "name": "meal",
                "label": "Meal preference",
                "type": "select",
                "options": ["Standard", "Vegetarian", "Vegan", "Kosher", "None"],
                "default": "Standard",
            },
            {
                "name": "insurance",
                "label": "Add trip insurance",
                "type": "checkbox",
                "default": False,
            },
        ],
    })

    if not decision.get("approved"):
        return (
            f"Booking cancelled by the user. No flight from {origin} to {destination} was reserved."
        )

    time.sleep(BUSINESS_LOGIC_DELAY_S)

    values = decision.get("values") or {}
    seat_class = str(values.get("seatClass", "Economy"))
    meal = str(values.get("meal", "Standard"))
    insurance = values.get("insurance") is True
    ref = _random_ref("BK")

    return " ".join([
        f"Flight booked. Confirmation {ref}.",
        f"Route: {origin} → {destination} on {date} for {passengers} passenger(s).",
        f"Seat class: {seat_class}. Meal: {meal}. Trip insurance: {'yes' if insurance else 'no'}.",
    ])


@tool(
    "issue_refund",
    description=(
        "Issue a refund for a customer order. Requires human approval, "
        "optionally adjusting the amount."
    ),
    args_schema=IssueRefundInput,
)
def issue_refund(orderId: str, amount: float, reason: str, currency: str = "USD") -> str:
    decision: dict[str, Any] = interrupt({
        "formType": "refund-approval",
        "tool": "issue_refund",
        "title": "Approve refund",
        "context": {
            "orderId": orderId,
            "amount": amount,
            "currency": currency,
            "reason": reason,
        },
        "fields": [
            {
                "name": "approvedAmount",
                "label": "Approved amount",
                "type": "currency",
                "currency": currency,
                "default": amount,
            },
            {
                "name": "note",
                "label": "Note to customer",
                "type": "textarea",
                "default": "",
            },
        ],
    })

    if not decision.get("approved"):
        return (
            f"Refund for order {orderId} was rejected by the reviewer. "
            "No money was returned to the customer."
        )

    time.sleep(BUSINESS_LOGIC_DELAY_S)

    values = decision.get("values") or {}
    approved_amount = float(values.get("approvedAmount", amount))
    note = values.get("note")
    note_suffix = f' Note: "{note}".' if isinstance(note, str) and note else ""

    return f"Refund of {currency} {approved_amount} issued for order {orderId}.{note_suffix}"


@tool(
    "publish_post",
    description=(
        "Publish a social media post on the user's behalf. "
        "Requires human review of the content."
    ),
    args_schema=PublishPostInput,
)
def publish_post(platform: str, content: str) -> str:
    decision: dict[str, Any] = interrupt({
        "formType": "content-review",
        "tool": "publish_post",
        "title": "Review post before publishing",
        "context": {"platform": platform},
        "fields": [
            {
                "name": "platform",
                "label": "Platform",
                "type": "select",
                "options": ["X", "LinkedIn", "Bluesky", "Mastodon"],
                "default": platform,
            },
            {
                "name": "content",
                "label": "Post content",
                "type": "textarea",
                "default": content,
            },
            {
                "name": "schedule",
                "label": "Schedule",
                "type": "select",
                "options": ["Publish now", "In 1 hour", "Tomorrow 9am"],
                "default": "Publish now",
            },
        ],
    })

    if not decision.get("approved"):
        return f"Post discarded by the user. Nothing was published to {platform}."

    time.sleep(BUSINESS_LOGIC_DELAY_S)

    values = decision.get("values") or {}
    final_platform = str(values.get("platform", platform))
    final_content = str(values.get("content", content))
    schedule = str(values.get("schedule", "Publish now"))
    slug = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    url = f"https://example.com/{final_platform.lower()}/posts/{slug}"

    action = "published" if schedule == "Publish now" else f"scheduled ({schedule})"
    return (
        f'Post {action} to {final_platform}: "{final_content}". Link: {url}'
    )
