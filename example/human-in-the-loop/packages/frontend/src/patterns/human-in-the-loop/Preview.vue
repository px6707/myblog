<script setup lang="ts">
import { computed, ref } from "vue";
import { useStream } from "@langchain/vue";
import { AIMessage, HumanMessage, type HITLRequest, type HITLResponse } from "langchain";
import type { humanInTheLoopAgent } from "@langchain/playground-agents";

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
import ApprovalCard from "./ApprovalCard.vue";

const PRESETS = [
  "Send an email to team@acme.com with subject 'Q4 Results' saying revenue grew 15% this quarter",
];

const threadId = ref<string | null>(null);
const stream = useStream<typeof humanInTheLoopAgent>({
  assistantId: SLUG_TO_ASSISTANT["human-in-the-loop"],
  threadId,
  onThreadId: (id: string) => {
    threadId.value = id;
  },
});

const isProcessing = ref(false);
const messages = computed(() => stream.messages.value);
const hitlRequest = computed(() => stream.interrupt.value?.value as HITLRequest | undefined);
const actionRequests = computed(() => hitlRequest.value?.actionRequests ?? []);
const reviewConfigs = computed(() => hitlRequest.value?.reviewConfigs ?? []);

function handleSubmit(text: string) {
  stream.submit({ messages: [{ type: "human" as const, content: text }] });
}

function handleNewThread() {
  threadId.value = null;
}

async function handleApprove() {
  if (!hitlRequest.value) return;
  isProcessing.value = true;
  try {
    const resume: HITLResponse = {
      decisions: actionRequests.value.map(() => ({ type: "approve" })),
    };
    await stream.respond(resume);
  } finally {
    isProcessing.value = false;
  }
}

async function handleReject(index: number, reason: string) {
  if (!hitlRequest.value) return;
  isProcessing.value = true;
  try {
    const resume: HITLResponse = {
      decisions: actionRequests.value.map((_, i: number) =>
        i === index
          ? { type: "reject" as const, message: reason || "User rejected" }
          : { type: "reject" as const, message: "Rejected along with other actions" },
      ),
    };
    await stream.respond(resume);
  } finally {
    isProcessing.value = false;
  }
}

async function handleEdit(index: number, editedArgs: Record<string, unknown>) {
  if (!hitlRequest.value) return;
  isProcessing.value = true;
  try {
    const originalAction = actionRequests.value[index];
    const resume: HITLResponse = {
      decisions: actionRequests.value.map((_, i: number) =>
        i === index
          ? { type: "edit" as const, editedAction: { name: originalAction.name, args: editedArgs } }
          : { type: "approve" as const },
      ),
    };
    await stream.respond(resume);
  } finally {
    isProcessing.value = false;
  }
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

      <AIBubble v-else-if="AIMessage.isInstance(msg) && msg.text">
        <Markdown :content="msg.text" />
      </AIBubble>
    </template>

    <TypingIndicator v-if="stream.isLoading.value && !hitlRequest" />

    <div v-if="hitlRequest && actionRequests.length > 0 && !isProcessing" class="pl-9 max-w-[80%]">
      <ApprovalCard
        v-for="(actionRequest, idx) in actionRequests"
        :key="idx"
        :action-request="actionRequest"
        :review-config="reviewConfigs[idx]"
        :is-processing="isProcessing"
        @approve="handleApprove()"
        @reject="(reason) => handleReject(idx, reason)"
        @edit="(editedArgs) => handleEdit(idx, editedArgs)"
      />
    </div>

    <template #input>
      <ChatInput
        @submit="handleSubmit"
        :disabled="stream.isLoading.value || isProcessing || !!hitlRequest"
        :placeholder="
          hitlRequest
            ? 'Please approve or reject the pending action...'
            : 'Ask the agent to do something sensitive...'
        "
        :showNewThread="messages.length > 0"
        @newThread="handleNewThread"
      />
    </template>
  </ChatContainer>
</template>
