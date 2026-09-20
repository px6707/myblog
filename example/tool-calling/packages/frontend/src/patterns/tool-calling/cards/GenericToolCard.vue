<script setup lang="ts">
import { ref } from "vue";
import type { AssembledToolCall } from "@langchain/vue";
import GenericResultValue from "./GenericResultValue.vue";
import GenericResultObject from "./GenericResultObject.vue";

defineProps<{
  toolCall: AssembledToolCall;
}>();

const resultOpen = ref(false);
</script>

<template>
  <div
    class="rounded-xl border border-border bg-surface-secondary p-4"
    data-testid="sdk-preview-chat-turn"
  >
    <div class="flex items-center gap-2.5">
      <svg
        class="w-4 h-4 text-primary"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <path
          d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"
        />
      </svg>
      <span class="font-semibold text-sm text-text font-mono flex-1">{{ toolCall.name }}</span>
      <svg
        v-if="toolCall.status === 'running'"
        class="w-4 h-4 text-primary animate-spin"
        viewBox="0 0 24 24"
        fill="none"
      >
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" />
        <path
          class="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
    </div>

    <div
      class="mt-2 rounded-md bg-surface-tertiary px-3 py-2 font-mono text-xs text-text-secondary overflow-x-auto"
    >
      <pre class="whitespace-pre-wrap">{{ JSON.stringify(toolCall.input, null, 2) }}</pre>
    </div>

    <div v-if="toolCall.status === 'finished' && toolCall.output !== null" class="mt-2">
      <button
        type="button"
        @click="resultOpen = !resultOpen"
        class="flex items-center gap-1.5 text-xs font-medium text-text-tertiary hover:text-text-secondary transition-colors"
      >
        <svg
          :class="['w-4 h-4 transition-transform duration-200', resultOpen ? 'rotate-90' : '']"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        Result
      </button>
      <div
        :class="[
          'overflow-hidden transition-all duration-200',
          resultOpen ? 'max-h-60 opacity-100 mt-2' : 'max-h-0 opacity-0',
        ]"
      >
        <div
          v-if="
            toolCall.output &&
            typeof toolCall.output === 'object' &&
            !Array.isArray(toolCall.output)
          "
          class="rounded-md border-l-2 border-emerald-500 dark:border-emerald-400 bg-surface-tertiary px-3 py-2.5 text-xs overflow-x-auto"
        >
          <GenericResultObject :data="toolCall.output as Record<string, unknown>" />
        </div>
        <div
          v-else-if="Array.isArray(toolCall.output)"
          class="rounded-md border-l-2 border-emerald-500 dark:border-emerald-400 bg-surface-tertiary px-3 py-2.5 text-xs overflow-x-auto"
        >
          <GenericResultValue :value="toolCall.output" />
        </div>
        <div
          v-else
          class="rounded-md bg-emerald-50 dark:bg-emerald-900/20 px-3 py-2 font-mono text-xs text-emerald-800 dark:text-emerald-300 overflow-x-auto"
        >
          <pre class="whitespace-pre-wrap">{{ String(toolCall.output) }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>
