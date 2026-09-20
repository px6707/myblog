<script setup lang="ts">
import GenericResultObject from "./GenericResultObject.vue";

defineProps<{
  value: unknown;
}>();
</script>

<template>
  <span v-if="value === null || value === undefined" class="text-text-tertiary italic">null</span>
  <span v-else-if="typeof value === 'boolean'" class="text-text-secondary font-medium">{{
    value ? "true" : "false"
  }}</span>
  <span v-else-if="typeof value === 'number'" class="text-text-secondary tabular-nums">{{
    value
  }}</span>
  <span v-else-if="typeof value === 'string'" class="text-text-secondary">{{ value }}</span>
  <ul
    v-else-if="Array.isArray(value)"
    class="list-disc list-outside pl-4 space-y-1 marker:text-emerald-500 dark:marker:text-emerald-400"
  >
    <li v-for="(item, i) in value" :key="i" class="text-text-secondary">
      <GenericResultObject
        v-if="typeof item === 'object' && item !== null"
        :data="item as Record<string, unknown>"
        nested
      />
      <GenericResultValue v-else :value="item" />
    </li>
  </ul>
  <GenericResultObject
    v-else-if="typeof value === 'object'"
    :data="value as Record<string, unknown>"
    nested
  />
  <span v-else class="text-text-secondary">{{ String(value) }}</span>
</template>
