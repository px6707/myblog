<script setup lang="ts">
import { ref } from "vue";
import { useStream } from "@langchain/vue";
import { HumanMessage, AIMessage } from "langchain";
import type { headlessToolsAgent } from "@langchain/playground-agents";

import { AGENT_SERVER_URL, SLUG_TO_ASSISTANT } from "@/constants";
import {
  PresetPrompts,
  AIBubble,
  HumanBubble,
  ChatContainer,
  ChatInput,
  Markdown,
  TypingIndicator,
} from "@/components/playground";

import MemoryToolCallCard from "./cards/MemoryToolCallCard.vue";
import MemoryStats from "./cards/MemoryStats.vue";
import {
  memoryPut,
  memoryGet,
  memoryList,
  memorySearch,
  memoryForget,
  geolocationGet,
} from "./impl";

const PRESETS = [
  "What do you remember about me?",
  "Remember that my name is Alex and I'm a developer",
  "I prefer concise, technical answers",
  "Where am I right now?",
];

const threadId = ref<string | null>(null);
const stream = useStream<typeof headlessToolsAgent>({
  apiUrl: AGENT_SERVER_URL,
  assistantId: SLUG_TO_ASSISTANT["headless-tools"],
  tools: [memoryPut, memoryGet, memoryList, memorySearch, memoryForget, geolocationGet],
  threadId,
  onThreadId: (id: string) => {
    threadId.value = id;
  },
});

function handleSubmit(text: string) {
  stream.submit({ messages: [{ type: "human" as const, content: text }] });
}

function getToolCallsForMessage(msg: AIMessage) {
  return stream.toolCalls.value.filter((tc) => msg.tool_calls?.find((t) => t.id === tc.callId));
}

function handleNewThread() {
  threadId.value = null;
}
</script>

<template>
  <ChatContainer>
    <PresetPrompts
      v-if="stream.messages.value.length === 0"
      :prompts="PRESETS"
      @select="handleSubmit"
    />

    <template v-for="msg in stream.messages.value" :key="msg.id">
      <HumanBubble v-if="HumanMessage.isInstance(msg)">
        <Markdown :content="msg.text" />
      </HumanBubble>

      <template v-else-if="AIMessage.isInstance(msg)">
        <div v-if="getToolCallsForMessage(msg).length > 0" class="flex flex-col gap-2">
          <MemoryToolCallCard
            v-for="tc in getToolCallsForMessage(msg)"
            :key="tc.callId"
            :tool-call="tc"
          />
        </div>
        <AIBubble v-else-if="msg.text">
          <Markdown :content="msg.text" />
        </AIBubble>
      </template>
    </template>

    <TypingIndicator
      v-if="
        stream.isLoading.value &&
        !stream.messages.value.some((m: any) => AIMessage.isInstance(m) && m.text) &&
        stream.toolCalls.value.length === 0
      "
    />

    <div
      v-if="stream.error.value"
      class="rounded-lg border border-red-200 bg-red-50 dark:border-red-900/50 dark:bg-red-950/20 px-3 py-2 text-sm text-red-700 dark:text-red-400"
    >
      <div class="flex items-center gap-2">
        <svg
          class="w-4 h-4 shrink-0"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <span>{{
          stream.error.value instanceof Error ? stream.error.value.message : "An error occurred"
        }}</span>
      </div>
    </div>

    <template #input>
      <MemoryStats />
      <ChatInput
        @submit="handleSubmit"
        :disabled="stream.isLoading.value"
        placeholder="Tell me something to remember, or ask what I know about you..."
        :showNewThread="stream.messages.value.length > 0"
        @newThread="handleNewThread"
      />
    </template>
  </ChatContainer>
</template>
