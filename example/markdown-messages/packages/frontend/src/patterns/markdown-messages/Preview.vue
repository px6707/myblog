<script setup lang="ts">
import { computed, ref } from "vue";
import { useStream } from "@langchain/vue";
import { AIMessage, HumanMessage } from "langchain";

import { SLUG_TO_ASSISTANT } from "@/constants";
import {
  ChatContainer,
  AIBubble,
  HumanBubble,
  ChatInput,
  TypingIndicator,
  PresetPrompts,
  Markdown,
} from "@/components/playground";

const PRESETS = [
  "Write a quick-start guide for building a REST API with Express.js",
  "Compare Python and Rust in a table with pros and cons",
  "Explain the merge sort algorithm with code examples",
];

const threadId = ref<string | null>(null);
const stream = useStream({
  assistantId: SLUG_TO_ASSISTANT["markdown-messages"],
  threadId,
  onThreadId: (id: string) => {
    threadId.value = id;
  },
});

const messages = computed(() => stream.messages.value);
const hasMessages = computed(() => messages.value.length > 0);

function handleSubmit(text: string) {
  stream.submit({ messages: [{ type: "human" as const, content: text }] });
}

function handleNewThread() {
  threadId.value = null;
}
</script>

<template>
  <ChatContainer>
    <PresetPrompts v-if="!hasMessages" :prompts="PRESETS" @select="handleSubmit" />

    <template v-for="(msg, i) in messages" :key="msg.id ?? i">
      <HumanBubble v-if="HumanMessage.isInstance(msg)">
        <Markdown :content="msg.text" />
      </HumanBubble>
      <AIBubble v-else-if="AIMessage.isInstance(msg)">
        <Markdown :content="msg.text" />
      </AIBubble>
    </template>

    <TypingIndicator v-if="stream.isLoading.value" />

    <template #input>
      <ChatInput @submit="handleSubmit" :showNewThread="hasMessages" @newThread="handleNewThread" />
    </template>
  </ChatContainer>
</template>
