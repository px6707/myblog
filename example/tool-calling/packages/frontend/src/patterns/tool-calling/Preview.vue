<script setup lang="ts">
import { ref } from "vue";
import { useStream } from "@langchain/vue";
import { HumanMessage, AIMessage } from "langchain";
import type { toolCallingAgent } from "@langchain/playground-agents";

import { SLUG_TO_ASSISTANT } from "@/constants";
import {
  PresetPrompts,
  AIBubble,
  HumanBubble,
  ChatContainer,
  ChatInput,
  Markdown,
  TypingIndicator,
} from "@/components/playground";
import ToolCallCard from "./cards/ToolCallCard.vue";

const PRESETS = [
  "What's the weather like in San Francisco?",
  "What is (25 * 4) + 130 / 2?",
  "Search for LangChain documentation",
];

const threadId = ref<string | null>(null);
const stream = useStream<typeof toolCallingAgent>({
  assistantId: SLUG_TO_ASSISTANT["tool-calling"],
  threadId,
  onThreadId: (id: string) => {
    threadId.value = id;
  },
});

function handleSubmit(text: string) {
  stream.submit({ messages: [{ type: "human" as const, content: text }] });
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

      <div v-else-if="AIMessage.isInstance(msg)" class="space-y-2">
        <AIBubble v-if="msg.text">
          <Markdown :content="msg.text" />
        </AIBubble>
        <div
          v-for="tc in stream.toolCalls.value.filter((tc) =>
            msg.tool_calls?.find((t: { id?: string }) => t.id === tc.callId),
          )"
          :key="tc.callId"
          class="pl-9 max-w-[80%]"
        >
          <ToolCallCard :tool-call="tc" />
        </div>
      </div>
    </template>

    <TypingIndicator v-if="stream.isLoading.value" />

    <template #input>
      <ChatInput
        @submit="handleSubmit"
        :disabled="stream.isLoading.value"
        :showNewThread="stream.messages.value.length > 0"
        @newThread="handleNewThread"
      />
    </template>
  </ChatContainer>
</template>
