<script setup lang="ts">
import { computed } from "vue";
import type { AssembledToolCall } from "@langchain/vue";

import LocationMap from "./LocationMap.vue";
import MemoryValue from "./MemoryValue.vue";

const props = defineProps<{
  toolCall: AssembledToolCall;
}>();

const TOOL_NAMES: Record<string, string> = {
  memory_put: "Saving to memory",
  memory_get: "Recalling memory",
  memory_list: "Listing memories",
  memory_search: "Searching memories",
  memory_forget: "Forgetting memory",
  geolocation_get: "Getting your location",
};

const call = computed(() => props.toolCall);
const isLoading = computed(() => call.value.status === "running");

const toolLabel = computed(() => {
  const name = call.value.name;
  return TOOL_NAMES[name] ?? name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
});

const subtitle = computed(() => {
  const name = call.value.name;
  const args = (call.value.input ?? {}) as Record<string, unknown>;
  if (!args || Object.keys(args).length === 0) return null;
  if (name === "memory_put" && args.key) return `Key: ${args.key}`;
  if (name === "memory_search" && args.query) return `Query: "${args.query}"`;
  return JSON.stringify(args);
});

const parsedData = computed(() => {
  if (call.value.status !== "finished" || call.value.output === undefined) return null;
  const output = call.value.output;
  const content = typeof output === "string" ? output : JSON.stringify(output);
  try {
    const parsed = JSON.parse(content);
    const id = call.value.callId;
    const data = id && parsed[id] !== undefined ? parsed[id] : parsed;
    return { data: data as Record<string, unknown>, raw: null };
  } catch {
    return { data: null, raw: content };
  }
});

type MemoryItem = { key: string; value: unknown; tags?: string[] };

const locationData = computed(() => {
  const d = parsedData.value?.data;
  if (!d || d.latitude === undefined || d.longitude === undefined) return null;
  return {
    lat: d.latitude as number,
    lng: d.longitude as number,
    accuracy: d.accuracy as number | undefined,
    saved: d.saved as boolean | undefined,
  };
});

const memoriesData = computed(() => {
  const d = parsedData.value?.data;
  if (!d || d.count === undefined || !d.memories) return null;
  return { count: d.count as number, memories: (d.memories as MemoryItem[]).slice(0, 5) };
});

const singleMemory = computed(() => {
  const d = parsedData.value?.data;
  if (!d || d.found === undefined) return null;
  return {
    found: d.found as boolean,
    key: d.key as string,
    value: d.value,
    message: d.message as string,
  };
});

const messageOnly = computed(() => {
  const d = parsedData.value?.data;
  if (!d || d.latitude !== undefined || d.count !== undefined || d.found !== undefined) return null;
  return d.message as string | undefined;
});

const fallbackJson = computed(() => {
  const d = parsedData.value?.data;
  if (!d) return null;
  if (d.latitude !== undefined || d.count !== undefined || d.found !== undefined || d.message)
    return null;
  return JSON.stringify(d, null, 2);
});
</script>

<template>
  <div class="rounded-lg border border-border bg-surface p-3">
    <div class="flex items-center gap-2 mb-2">
      <div
        class="w-7 h-7 rounded-md bg-primary/10 border border-primary/20 flex items-center justify-center text-primary"
      >
        <svg
          v-if="call.name === 'memory_put'"
          class="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
          <polyline points="17 21 17 13 7 13 7 21" />
          <polyline points="7 3 7 8 15 8" />
        </svg>
        <svg
          v-else-if="call.name === 'memory_get'"
          class="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
        <svg
          v-else-if="call.name === 'memory_list'"
          class="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <ellipse cx="12" cy="5" rx="9" ry="3" />
          <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
        </svg>
        <svg
          v-else-if="call.name === 'memory_search'"
          class="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.35-4.35" />
        </svg>
        <svg
          v-else-if="call.name === 'memory_forget'"
          class="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <polyline points="3 6 5 6 21 6" />
          <path
            d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"
          />
          <line x1="10" y1="11" x2="10" y2="17" />
          <line x1="14" y1="11" x2="14" y2="17" />
        </svg>
        <svg
          v-else
          class="w-4 h-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path
            d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.96-3 2.5 2.5 0 0 1-1.32-4.24 3 3 0 0 1 .34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.18-1.5A2.5 2.5 0 0 1 9.5 2z"
          />
          <path
            d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 1.96-3 2.5 2.5 0 0 0 1.32-4.24 3 3 0 0 0-.34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.18-1.5A2.5 2.5 0 0 0 14.5 2z"
          />
        </svg>
      </div>

      <div class="flex-1 min-w-0">
        <div class="text-sm font-medium">{{ toolLabel }}</div>
        <div v-if="subtitle" class="text-xs text-text-secondary truncate">{{ subtitle }}</div>
      </div>

      <svg
        v-if="isLoading"
        class="w-4 h-4 animate-spin text-primary shrink-0"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </svg>
    </div>

    <div
      v-if="parsedData?.data || parsedData?.raw"
      class="text-sm rounded-md p-2 bg-surface-secondary border border-border text-text-primary"
    >
      <LocationMap
        v-if="locationData"
        :latitude="locationData.lat"
        :longitude="locationData.lng"
        :accuracy="locationData.accuracy"
        :saved="locationData.saved"
      />

      <template v-else-if="memoriesData">
        <div class="space-y-2">
          <div class="text-text-secondary text-xs">
            Found {{ memoriesData.count }}
            {{ memoriesData.count === 1 ? "memory" : "memories" }}
          </div>
          <div
            v-for="(m, i) in memoriesData.memories"
            :key="i"
            class="bg-surface-secondary rounded p-2 text-xs"
          >
            <div class="font-medium">{{ m.key }}</div>
            <div class="text-text-secondary">
              <MemoryValue :value="m.value" />
            </div>
            <div v-if="m.tags && m.tags.length > 0" class="flex gap-1 mt-1 flex-wrap">
              <span
                v-for="tag in m.tags"
                :key="tag"
                class="px-1.5 py-0.5 bg-primary/10 text-primary rounded text-xs"
                >{{ tag }}</span
              >
            </div>
          </div>
        </div>
      </template>

      <template v-else-if="singleMemory">
        <div v-if="singleMemory.found" class="space-y-1">
          <div class="font-medium text-xs">{{ singleMemory.key }}</div>
          <div class="text-text-secondary text-xs">
            <MemoryValue :value="singleMemory.value" />
          </div>
        </div>
        <span v-else class="text-xs">{{ singleMemory.message }}</span>
      </template>

      <span v-else-if="messageOnly" class="text-xs">{{ messageOnly }}</span>

      <pre v-else-if="fallbackJson" class="text-xs overflow-auto whitespace-pre-wrap">{{
        fallbackJson
      }}</pre>

      <span v-else-if="parsedData?.raw" class="text-xs">{{ parsedData.raw }}</span>
    </div>
  </div>
</template>
