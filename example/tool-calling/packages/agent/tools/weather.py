import json
import random

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
    temp = random.randint(50, 80)
    return json.dumps({
        "city": city,
        "temperature": temp,
        "condition": random.choice(conditions),
        "unit": "fahrenheit",
    })
