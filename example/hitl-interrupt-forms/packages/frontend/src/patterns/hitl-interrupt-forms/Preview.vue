<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useStream } from "@langchain/vue";
import { AIMessage, HumanMessage, ToolMessage } from "langchain";
import type {
  hitlInterruptFormsAgent,
  InterruptCard,
  ReviewDecision,
} from "@langchain/playground-agents";

import { AGENT_SERVER_URL, SLUG_TO_ASSISTANT } from "@/constants";
import {
  ChatContainer,
  AIBubble,
  HumanBubble,
  ChatInput,
  TypingIndicator,
  PresetPrompts,
  Markdown,
} from "@/components/playground";
import InterruptFormCard from "./InterruptFormCard.vue";

const PRESETS = [
  "Book a flight from SFO to JFK on 2026-07-14 for 2 passengers",
  "Refund order ORD-4821 for $129.99 — the package arrived damaged",
  "Publish a post on LinkedIn announcing our new HITL playground",
];

/** Pull a persisted card off an AIMessage that the frontend pushed into state. */
function readMessageCard(message: AIMessage): InterruptCard | null {
  const metadata = message.response_metadata as { cards?: InterruptCard } | undefined;
  return metadata?.cards ?? null;
}

const threadId = ref<string | null>(null);
const stream = useStream<typeof hitlInterruptFormsAgent>({
  apiUrl: AGENT_SERVER_URL,
  assistantId: SLUG_TO_ASSISTANT["hitl-interrupt-forms"],
  threadId,
  onThreadId: (id: string) => {
    threadId.value = id;
  },
});

const messages = computed(() => stream.messages.value);
const interrupt = computed(() => stream.interrupt.value);
const pendingCard = computed(() => interrupt.value?.value as InterruptCard | undefined);

// The user has acted on the current interrupt and we're awaiting the resumed
// run. Used only to swap the interactive form for the resolved card; the card
// itself is kept in state by the SDK (see `handleResolve`). Reset when the
// interrupt clears so a later turn's interrupt shows its form again.
const isResolving = ref(false);
watch(interrupt, (value) => {
  if (!value) isResolving.value = false;
});

function handleSubmit(text: string) {
  stream.submit({ messages: [{ type: "human" as const, content: text }] });
}

function handleNewThread() {
  threadId.value = null;
}

// Resolve the interrupt AND push the card into state in a single atomic
// `respond(decision, { update })`, mapped to `Command(resume, update)`. The
// SDK applies the update optimistically — the card paints immediately and
// reconciles by id when the resumed run echoes it back — so it stays rendered
// with no flicker while the (slow) tool runs, and the backend never re-emits it.
function handleResolve(decision: ReviewDecision) {
  const card = pendingCard.value;
  if (!card) return;
  isResolving.value = true;
  const resolvedCard: InterruptCard = { ...card, resolved: true, decision };
  const cardMessage = new AIMessage({
    content: `Review ${decision.approved ? "approved" : "declined"} for ${card.tool}.`,
    response_metadata: { cards: resolvedCard },
  });
  void stream.respond(decision, { update: { messages: [cardMessage] } });
}
</script>

<template>
  <ChatContainer>
    <PresetPrompts
      v-if="messages.length === 0 && !stream.isLoading.value"
      :prompts="PRESETS"
      @select="handleSubmit"
    />

    <template v-for="msg in messages" :key="msg.id">
      <HumanBubble v-if="HumanMessage.isInstance(msg)">
        <Markdown :content="msg.text" />
      </HumanBubble>

      <template v-else-if="AIMessage.isInstance(msg)">
        <div v-if="readMessageCard(msg)" class="pl-9 max-w-[80%]">
          <InterruptFormCard :card="readMessageCard(msg)!" read-only />
        </div>
        <AIBubble v-else-if="msg.text">
          <Markdown :content="msg.text" />
        </AIBubble>
      </template>

      <div
        v-else-if="ToolMessage.isInstance(msg) && typeof msg.content === 'string' && msg.content"
        class="ml-9 max-w-[80%] border-l-2 border-border pl-3 text-xs text-text-tertiary"
      >
        {{ msg.content }}
      </div>
    </template>

    <TypingIndicator v-if="stream.isLoading.value && !pendingCard" />

    <div v-if="pendingCard && !isResolving" class="pl-9 max-w-[80%]">
      <InterruptFormCard :card="pendingCard" @resolve="handleResolve" />
    </div>

    <template #input>
      <ChatInput
        @submit="handleSubmit"
        :disabled="stream.isLoading.value || !!interrupt"
        :placeholder="
          interrupt
            ? 'Review the pending action above...'
            : 'Ask the agent to book a flight, issue a refund, or publish a post...'
        "
        :showNewThread="messages.length > 0"
        @newThread="handleNewThread"
      />
    </template>
  </ChatContainer>
</template>
