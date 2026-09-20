<script setup lang="ts">
import LocationMap from "./LocationMap.vue";

defineProps<{
  value: unknown;
}>();

function isLocation(
  value: unknown,
): value is { latitude: number; longitude: number; accuracy?: number } {
  return value !== null && typeof value === "object" && "latitude" in value && "longitude" in value;
}
</script>

<template>
  <LocationMap
    v-if="isLocation(value)"
    :latitude="value.latitude"
    :longitude="value.longitude"
    :accuracy="value.accuracy"
  />
  <span v-else-if="typeof value === 'string'" class="truncate">{{ value }}</span>
  <pre v-else class="text-xs overflow-auto whitespace-pre-wrap">{{
    JSON.stringify(value, null, 2)
  }}</pre>
</template>
