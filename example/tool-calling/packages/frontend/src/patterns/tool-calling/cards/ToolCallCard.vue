<script setup lang="ts">
import type { AgentToolCalls } from "../types";
import WeatherToolCard from "./WeatherToolCard.vue";
import CalculatorToolCard from "./CalculatorToolCard.vue";
import SearchToolCard from "./SearchToolCard.vue";
import GenericToolCard from "./GenericToolCard.vue";

defineProps<{
  toolCall: AgentToolCalls;
}>();
</script>

<template>
  <WeatherToolCard v-if="toolCall.name === 'get_weather'" :tool-call="toolCall" />
  <CalculatorToolCard v-else-if="toolCall.name === 'calculate'" :tool-call="toolCall" />
  <SearchToolCard v-else-if="toolCall.name === 'search_web'" :tool-call="toolCall" />
  <GenericToolCard v-else :tool-call="toolCall" />
</template>
