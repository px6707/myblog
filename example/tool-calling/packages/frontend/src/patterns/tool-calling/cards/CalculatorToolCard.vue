<script setup lang="ts">
import type { ToolCallFromTool } from "@langchain/vue";
import type { calculatorTool } from "@langchain/playground-agents";

defineProps<{
  toolCall: ToolCallFromTool<typeof calculatorTool>;
}>();
</script>

<template>
  <div
    v-if="toolCall.status === 'running'"
    class="rounded-xl border border-accent-gold bg-surface p-4"
    data-testid="sdk-preview-chat-turn"
  >
    <div class="flex items-center gap-3">
      <svg class="w-5 h-5 text-accent-gold animate-spin" viewBox="0 0 24 24" fill="none">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" />
        <path
          class="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      <div class="text-sm font-medium text-text-secondary font-mono">
        Computing {{ toolCall.input.expression }}…
      </div>
    </div>
  </div>
  <div
    v-else
    class="rounded-xl border border-accent-gold bg-surface p-4"
    data-testid="sdk-preview-chat-turn"
  >
    <div class="flex items-center gap-2 mb-3">
      <div class="w-6 h-6 rounded-md bg-accent-gold/20 flex items-center justify-center">
        <span class="text-text-secondary text-xs font-bold">fx</span>
      </div>
      <span class="text-[11px] font-semibold text-text-tertiary uppercase tracking-wider"
        >Calculator</span
      >
    </div>
    <div v-if="toolCall.error" class="text-sm text-error font-mono">{{ toolCall.error }}</div>
    <div v-else class="font-mono">
      <div class="text-sm text-text-tertiary">{{ toolCall.input.expression }}</div>
      <div class="text-2xl font-bold text-text tabular-nums mt-1">
        = {{ toolCall.output?.result }}
      </div>
    </div>
  </div>
</template>
