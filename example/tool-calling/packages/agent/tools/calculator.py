import json
import re

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 64
MAX_LITERAL_DIGITS = 15


def _validation_error(sanitized: str) -> str | None:
    if len(sanitized) > MAX_EXPRESSION_LENGTH:
        return f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)."
    if "**" in sanitized:
        return "Exponentiation (**) is not allowed."
    if not re.match(r"^[\d+\-*/().]+$", sanitized):
        return "Only numbers and +, -, *, /, (, ) are allowed."
    for match in re.finditer(r"\d+", sanitized):
        if len(match.group()) > MAX_LITERAL_DIGITS:
            return f"Number literals may contain at most {MAX_LITERAL_DIGITS} digits."
    return None


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, and parentheses."""
    sanitized = re.sub(r"\s", "", expression)
    error = _validation_error(sanitized)
    if error is not None:
        return json.dumps({
            "expression": expression,
            "error": f'Invalid expression: "{expression}". {error}',
        })
    try:
        result = eval(sanitized, {"__builtins__": {}}, {})  # noqa: S307
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})
