<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";

const count = ref(0);
const tags = ref<string[]>([]);
const loading = ref(true);

function checkMemories() {
  try {
    const request = indexedDB.open("agent-memory", 2);
    request.onsuccess = () => {
      const db = request.result;
      if (db.objectStoreNames.contains("memories")) {
        const transaction = db.transaction("memories", "readonly");
        const store = transaction.objectStore("memories");
        const countRequest = store.count();
        const allRequest = store.getAll();

        countRequest.onsuccess = () => {
          allRequest.onsuccess = () => {
            const memories = allRequest.result as Array<{ tags: string[] }>;
            const allTags = new Set<string>();
            memories.forEach((m) => m.tags?.forEach((t) => allTags.add(t)));
            count.value = countRequest.result;
            tags.value = Array.from(allTags).slice(0, 5);
            loading.value = false;
          };
        };
      } else {
        count.value = 0;
        tags.value = [];
        loading.value = false;
      }
      db.close();
    };
    request.onerror = () => {
      count.value = 0;
      tags.value = [];
      loading.value = false;
    };
  } catch {
    count.value = 0;
    tags.value = [];
    loading.value = false;
  }
}

let interval: ReturnType<typeof setInterval>;

onMounted(() => {
  checkMemories();
  interval = setInterval(checkMemories, 5000);
});

onUnmounted(() => {
  clearInterval(interval);
});
</script>

<template>
  <div v-if="!loading" class="flex items-center gap-3 text-xs text-text-secondary px-4 py-1.5">
    <div class="flex items-center gap-1">
      <svg
        class="w-3 h-3"
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
      <span>{{ count }} {{ count === 1 ? "memory" : "memories" }} stored</span>
    </div>
    <div v-if="tags.length > 0" class="flex items-center gap-1">
      <span v-for="tag in tags" :key="tag" class="px-1.5 py-0.5 bg-surface-secondary rounded">{{
        tag
      }}</span>
    </div>
  </div>
</template>
