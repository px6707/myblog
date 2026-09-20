<script setup lang="ts">
const props = defineProps<{
  latitude: number;
  longitude: number;
  accuracy?: number;
  saved?: boolean;
}>();

const delta = 0.005;
const bbox = `${props.longitude - delta},${props.latitude - delta},${props.longitude + delta},${props.latitude + delta}`;
const iframeSrc = `https://www.openstreetmap.org/export/embed.html?bbox=${bbox}&layer=mapnik&marker=${props.latitude},${props.longitude}`;
const osmHref = `https://www.openstreetmap.org/?mlat=${props.latitude}&mlon=${props.longitude}#map=16/${props.latitude}/${props.longitude}`;
</script>

<template>
  <div class="space-y-2">
    <div class="overflow-hidden rounded-lg border border-border" style="height: 200px">
      <iframe
        :src="iframeSrc"
        title="Your location on OpenStreetMap"
        class="w-full h-full"
        style="border: 0"
        loading="lazy"
        referrerpolicy="no-referrer"
      />
    </div>
    <div class="flex items-center justify-between text-xs">
      <span class="font-mono text-text-secondary">
        {{ latitude.toFixed(5) }}, {{ longitude.toFixed(5) }}
        <span v-if="accuracy" class="ml-2 opacity-70">±{{ Math.round(accuracy) }} m</span>
      </span>
      <div class="flex items-center gap-3">
        <span v-if="saved" class="text-green-600 dark:text-green-400 flex items-center gap-1">
          <svg
            class="w-3 h-3"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <circle cx="12" cy="12" r="10" />
            <path d="m9 12 2 2 4-4" />
          </svg>
          Saved to memory
        </span>
        <a
          :href="osmHref"
          target="_blank"
          rel="noopener noreferrer"
          class="text-primary hover:underline"
          >Open in OSM ↗</a
        >
      </div>
    </div>
  </div>
</template>
