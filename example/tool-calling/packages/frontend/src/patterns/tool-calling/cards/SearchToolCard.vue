<script setup lang="ts">
import type { ToolCallFromTool } from "@langchain/vue";
import type { searchWebTool } from "@langchain/playground-agents";

defineProps<{
  toolCall: ToolCallFromTool<typeof searchWebTool>;
}>();
</script>

<template>
  <div
    v-if="!toolCall.output"
    class="rounded-xl border border-border bg-surface p-4"
    data-testid="sdk-preview-chat-turn"
  >
    <div class="flex items-center gap-3">
      <svg class="w-5 h-5 text-primary animate-spin" viewBox="0 0 24 24" fill="none">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" />
        <path
          class="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      <div class="text-sm font-medium text-text-secondary">
        Searching for &ldquo;{{ toolCall.args.query }}&rdquo;…
      </div>
    </div>
  </div>
  <div
    v-else
    class="rounded-xl border border-border bg-surface p-4"
    data-testid="sdk-preview-chat-turn"
  >
    <div class="flex items-center gap-2 mb-3">
      <svg
        class="w-4 h-4 text-primary"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
      <span class="text-[11px] font-semibold text-text-tertiary uppercase tracking-wider"
        >Search Results</span
      >
    </div>
    <div class="space-y-3">
      <div
        v-for="(r, i) in toolCall.output.results"
        :key="i"
        :class="i > 0 ? 'pt-3 border-t border-border' : ''"
      >
        <div class="text-sm font-medium text-text">{{ r.title }}</div>
        <a
          :href="r.url"
          target="_blank"
          rel="noopener noreferrer"
          class="text-[11px] text-primary truncate mt-0.5"
          >{{ r.url }}</a
        >
        <div class="text-xs text-text-tertiary mt-0.5 leading-relaxed">{{ r.snippet }}</div>
      </div>
    </div>
  </div>
</template>
