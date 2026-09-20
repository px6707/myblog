from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field


def _client_tool_args(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def _interrupt_for_client(
    tool_name: str,
    args: dict[str, Any],
    runtime: ToolRuntime,
) -> Any:
    return interrupt({
        "type": "tool",
        "tool_call": {
            "id": runtime.tool_call_id,
            "name": tool_name,
            "args": args,
        },
    })


class MemoryPutInput(BaseModel):
    key: str = Field(
        description="Unique identifier for this memory (e.g. 'user_name', 'preferred_language')"
    )
    value: Any = Field(
        description="The value to store — can be a string, object, or any JSON-serializable data"
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tags to categorize this memory (e.g. ['preference', 'work'])",
    )
    ttlDays: float | None = Field(
        default=None,
        description="Optional: days until this memory expires (omit for permanent)",
    )


class MemoryGetInput(BaseModel):
    key: str = Field(description="The key of the memory to retrieve")


class MemoryListInput(BaseModel):
    tags: list[str] | None = Field(default=None, description="Filter memories by these tags")
    limit: float | None = Field(
        default=None,
        description="Maximum number of memories to return (default 20)",
    )


class MemorySearchInput(BaseModel):
    query: str = Field(description="Search query to find matching memories")
    tags: list[str] | None = Field(
        default=None,
        description="Optionally filter to memories with these tags",
    )
    limit: float | None = Field(
        default=None,
        description="Maximum results to return (default 10)",
    )


class MemoryForgetInput(BaseModel):
    key: str | None = Field(default=None, description="The key of the memory to delete")
    tag: str | None = Field(default=None, description="Delete all memories with this tag")
    confirmForgetAll: bool | None = Field(
        default=None,
        description="Set to true to delete ALL memories (use with caution)",
    )


class GeolocationGetInput(BaseModel):
    save: bool | None = Field(
        default=None,
        description="Save the location to memory for future reference (default true)",
    )


@tool(
    "memory_put",
    description=(
        "Store a memory in the user's browser for long-term recall. "
        "Use this to save user preferences, important facts, or context that should persist "
        "across sessions. Memories are stored locally and never leave the user's device."
    ),
    args_schema=MemoryPutInput,
)
def memory_put(
    key: str,
    value: Any,
    runtime: ToolRuntime,
    tags: list[str] | None = None,
    ttlDays: float | None = None,
) -> Any:
    return _interrupt_for_client(
        "memory_put",
        _client_tool_args(key=key, value=value, tags=tags, ttlDays=ttlDays),
        runtime,
    )


@tool(
    "memory_get",
    description=(
        "Retrieve a specific memory by its key. "
        "Use this to recall previously stored information like user preferences or saved context."
    ),
    args_schema=MemoryGetInput,
)
def memory_get(key: str, runtime: ToolRuntime) -> Any:
    return _interrupt_for_client(
        "memory_get",
        _client_tool_args(key=key),
        runtime,
    )


@tool(
    "memory_list",
    description=(
        "List all stored memories, optionally filtered by tags. "
        "Use this to see what the user has asked you to remember or to find relevant context."
    ),
    args_schema=MemoryListInput,
)
def memory_list(
    runtime: ToolRuntime,
    tags: list[str] | None = None,
    limit: float | None = None,
) -> Any:
    return _interrupt_for_client(
        "memory_list",
        _client_tool_args(tags=tags, limit=limit),
        runtime,
    )


@tool(
    "memory_search",
    description=(
        "Search through stored memories by content. "
        "Use this to find relevant memories when you're not sure of the exact key."
    ),
    args_schema=MemorySearchInput,
)
def memory_search(
    query: str,
    runtime: ToolRuntime,
    tags: list[str] | None = None,
    limit: float | None = None,
) -> Any:
    return _interrupt_for_client(
        "memory_search",
        _client_tool_args(query=query, tags=tags, limit=limit),
        runtime,
    )


@tool(
    "memory_forget",
    description=(
        "Delete a memory by key, all memories with a tag, or clear all memories. "
        "Use this when the user asks you to forget something."
    ),
    args_schema=MemoryForgetInput,
)
def memory_forget(
    runtime: ToolRuntime,
    key: str | None = None,
    tag: str | None = None,
    confirmForgetAll: bool | None = None,
) -> Any:
    return _interrupt_for_client(
        "memory_forget",
        _client_tool_args(key=key, tag=tag, confirmForgetAll=confirmForgetAll),
        runtime,
    )


@tool(
    "geolocation_get",
    description=(
        "Get the user's current GPS coordinates using the browser's Geolocation API. "
        "Saves latitude, longitude, accuracy, and timestamp to local memory so they can be "
        "referenced in future conversations. The browser will prompt for permission the first time."
    ),
    args_schema=GeolocationGetInput,
)
def geolocation_get(runtime: ToolRuntime, save: bool | None = None) -> Any:
    return _interrupt_for_client(
        "geolocation_get",
        _client_tool_args(save=save),
        runtime,
    )
