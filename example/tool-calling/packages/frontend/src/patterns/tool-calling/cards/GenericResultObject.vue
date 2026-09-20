<script setup lang="ts">
import GenericResultValue from "./GenericResultValue.vue";

defineProps<{
  data: Record<string, unknown>;
  nested?: boolean;
}>();
</script>

<template>
  <div :class="nested ? 'space-y-1.5' : 'space-y-2'">
    <div v-for="[key, value] in Object.entries(data)" :key="key">
      <div
        class="text-[11px] font-semibold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider mb-0.5"
      >
        {{ key.replace(/[_-]/g, " ") }}
      </div>
      <GenericResultValue :value="value" />
    </div>
  </div>
</template>
