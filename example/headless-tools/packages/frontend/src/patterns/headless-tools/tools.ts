/**
 * Browser Tools: Long-Term Memory
 *
 * Browser tools that provide durable, user-controlled memory using IndexedDB.
 * The agent can remember user preferences, facts, and context across sessions —
 * all stored locally in the browser, never leaving the device.
 *
 * These tools run client-side via the LangGraph interrupt mechanism:
 * the server registers the schema; execution happens in the browser.
 */

import { tool } from "langchain";
import { z } from "zod";

export const memoryPut = tool({
  name: "memory_put",
  description:
    "Store a memory in the user's browser for long-term recall. " +
    "Use this to save user preferences, important facts, or context that should persist across sessions. " +
    "Memories are stored locally and never leave the user's device.",
  schema: z.object({
    key: z
      .string()
      .describe("Unique identifier for this memory (e.g. 'user_name', 'preferred_language')"),
    value: z
      .unknown()
      .describe("The value to store — can be a string, object, or any JSON-serializable data"),
    tags: z
      .array(z.string())
      .optional()
      .describe("Tags to categorize this memory (e.g. ['preference', 'work'])"),
    ttlDays: z
      .number()
      .optional()
      .describe("Optional: days until this memory expires (omit for permanent)"),
  }),
});

export const memoryGet = tool({
  name: "memory_get",
  description:
    "Retrieve a specific memory by its key. " +
    "Use this to recall previously stored information like user preferences or saved context.",
  schema: z.object({
    key: z.string().describe("The key of the memory to retrieve"),
  }),
});

export const memoryList = tool({
  name: "memory_list",
  description:
    "List all stored memories, optionally filtered by tags. " +
    "Use this to see what the user has asked you to remember or to find relevant context.",
  schema: z.object({
    tags: z.array(z.string()).optional().describe("Filter memories by these tags"),
    limit: z.number().optional().describe("Maximum number of memories to return (default 20)"),
  }),
});

export const memorySearch = tool({
  name: "memory_search",
  description:
    "Search through stored memories by content. " +
    "Use this to find relevant memories when you're not sure of the exact key.",
  schema: z.object({
    query: z.string().describe("Search query to find matching memories"),
    tags: z.array(z.string()).optional().describe("Optionally filter to memories with these tags"),
    limit: z.number().optional().describe("Maximum results to return (default 10)"),
  }),
});

export const memoryForget = tool({
  name: "memory_forget",
  description:
    "Delete a memory by key, all memories with a tag, or clear all memories. " +
    "Use this when the user asks you to forget something.",
  schema: z.object({
    key: z.string().optional().describe("The key of the memory to delete"),
    tag: z.string().optional().describe("Delete all memories with this tag"),
    confirmForgetAll: z
      .boolean()
      .optional()
      .describe("Set to true to delete ALL memories (use with caution)"),
  }),
});

export const geolocationGet = tool({
  name: "geolocation_get",
  description:
    "Get the user's current GPS coordinates using the browser's Geolocation API. " +
    "Saves latitude, longitude, accuracy, and timestamp to local memory so they can be " +
    "referenced in future conversations. The browser will prompt for permission the first time.",
  schema: z.object({
    save: z
      .boolean()
      .optional()
      .describe("Save the location to memory for future reference (default true)"),
  }),
});
