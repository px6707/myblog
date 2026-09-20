<script setup lang="ts">
import { ref, computed } from "vue";
import type { ActionRequest, ReviewConfig } from "langchain";

const props = defineProps<{
  actionRequest: ActionRequest;
  reviewConfig?: ReviewConfig;
  isProcessing: boolean;
}>();

const emit = defineEmits<{
  approve: [];
  reject: [reason: string];
  edit: [editedArgs: Record<string, unknown>];
}>();

const isEditing = ref(false);
const editedArgs = ref<Record<string, unknown>>({ ...props.actionRequest.args });
const rejectReason = ref("");
const showRejectInput = ref(false);

const allowed = computed(() => props.reviewConfig?.allowedDecisions ?? ["approve", "reject"]);
const canEdit = computed(() => allowed.value.includes("edit"));
const canReject = computed(() => allowed.value.includes("reject"));

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}
</script>

<template>
  <!-- Editing mode -->
  <div
    v-if="isEditing"
    class="rounded-xl border border-border bg-surface overflow-hidden"
    data-testid="sdk-preview-chat-turn"
  >
    <div class="flex items-center gap-2 px-4 py-2.5 border-b border-border bg-surface-secondary">
      <svg
        class="h-4 w-4 text-primary"
        fill="none"
        viewBox="0 0 24 24"
        stroke-width="1.5"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.832 19.82a4.5 4.5 0 0 1-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 0 1 1.13-1.897L16.863 4.487Zm0 0L19.5 7.125"
        />
      </svg>
      <span class="text-sm font-medium text-text">
        Edit —
        <code class="font-mono text-xs bg-surface-tertiary px-1.5 py-0.5 rounded">{{
          actionRequest.name
        }}</code>
      </span>
    </div>
    <div class="p-4 space-y-3">
      <div v-for="[key, value] in Object.entries(editedArgs)" :key="key">
        <label :for="`edit-${key}`" class="block text-xs font-medium text-text-secondary mb-1.5">{{
          key
        }}</label>
        <textarea
          v-if="formatValue(value).length > 80 || formatValue(value).includes('\n')"
          :id="`edit-${key}`"
          :value="formatValue(value)"
          @input="editedArgs[key] = ($event.target as HTMLTextAreaElement).value"
          rows="4"
          class="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text font-mono focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary resize-y"
        />
        <input
          v-else
          :id="`edit-${key}`"
          type="text"
          :value="formatValue(value)"
          @input="editedArgs[key] = ($event.target as HTMLInputElement).value"
          class="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary"
        />
      </div>
      <div class="flex items-center gap-2 pt-1">
        <button
          type="button"
          @click="
            emit('edit', editedArgs);
            isEditing = false;
          "
          :disabled="isProcessing"
          class="inline-flex items-center gap-1.5 rounded-lg bg-primary-dark px-3.5 py-2 text-sm font-medium text-white hover:opacity-90 transition-opacity disabled:opacity-40"
        >
          <svg
            class="h-3.5 w-3.5"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="2"
            stroke="currentColor"
          >
            <path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" />
          </svg>
          Save &amp; Approve
        </button>
        <button
          type="button"
          @click="
            editedArgs = { ...actionRequest.args };
            isEditing = false;
          "
          class="inline-flex items-center rounded-lg border border-border bg-surface px-3.5 py-2 text-sm font-medium text-text-secondary hover:bg-surface-secondary transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  </div>

  <!-- Review mode -->
  <div
    v-else
    class="rounded-xl border border-border bg-surface overflow-hidden"
    data-testid="sdk-preview-chat-turn"
  >
    <div class="flex items-center gap-2 px-4 py-2.5 border-b border-border bg-surface-secondary">
      <svg
        class="h-4 w-4 text-warning"
        fill="none"
        viewBox="0 0 24 24"
        stroke-width="1.5"
        stroke="currentColor"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          d="M9 12.75 11.25 15 15 9.75m-3-7.036A11.959 11.959 0 0 1 3.598 6 11.99 11.99 0 0 0 3 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.249-8.25-3.285Z"
        />
      </svg>
      <span class="text-sm font-medium text-text">Review Required</span>
      <span class="ml-auto text-[11px] font-medium px-2 py-0.5 rounded-full bg-warning text-white"
        >Awaiting Approval</span
      >
    </div>

    <div class="p-4 space-y-3">
      <div class="rounded-lg bg-surface-tertiary border border-border p-3">
        <code class="text-sm font-mono font-semibold text-text">{{ actionRequest.name }}</code>
        <p v-if="actionRequest.description" class="text-xs text-text-tertiary mt-1">
          {{ actionRequest.description }}
        </p>

        <div class="mt-2.5 space-y-2">
          <div v-for="[key, value] in Object.entries(actionRequest.args)" :key="key">
            <div class="text-[11px] font-medium text-text-tertiary uppercase tracking-wider mb-0.5">
              {{ key }}
            </div>
            <pre
              v-if="formatValue(value).includes('\n')"
              class="text-sm text-text whitespace-pre-wrap wrap-break-word font-mono bg-surface rounded-md px-2.5 py-1.5 border border-border"
              >{{ formatValue(value) }}</pre
            >
            <div v-else class="text-sm text-text wrap-break-word">{{ formatValue(value) }}</div>
          </div>
        </div>
      </div>

      <div v-if="showRejectInput" class="space-y-2">
        <label for="reject-reason" class="block text-xs font-medium text-text-secondary">
          Reason for rejection <span class="text-text-tertiary font-normal">(optional)</span>
        </label>
        <input
          id="reject-reason"
          v-model="rejectReason"
          type="text"
          placeholder="Enter reason..."
          class="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-red-400/30 focus:border-red-400"
        />
      </div>

      <div class="flex flex-wrap items-center gap-2">
        <template v-if="showRejectInput">
          <button
            type="button"
            @click="
              emit('reject', rejectReason || 'User rejected');
              showRejectInput = false;
            "
            :disabled="isProcessing"
            class="inline-flex items-center gap-1.5 rounded-lg bg-red-600 hover:bg-red-700 px-3.5 py-2 text-sm font-medium text-white transition-colors disabled:opacity-40"
          >
            <svg
              class="h-3.5 w-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke-width="2"
              stroke="currentColor"
            >
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
            Confirm Rejection
          </button>
          <button
            type="button"
            @click="showRejectInput = false"
            class="inline-flex items-center rounded-lg border border-border bg-surface px-3.5 py-2 text-sm font-medium text-text-secondary hover:bg-surface-secondary transition-colors"
          >
            Cancel
          </button>
        </template>
        <template v-else>
          <button
            type="button"
            @click="emit('approve')"
            :disabled="isProcessing"
            class="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 dark:bg-emerald-600 dark:hover:bg-emerald-500 px-3.5 py-2 text-sm font-medium text-white transition-colors disabled:opacity-40"
          >
            <svg
              class="h-3.5 w-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke-width="2"
              stroke="currentColor"
            >
              <path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" />
            </svg>
            Approve
          </button>
          <button
            v-if="canEdit"
            type="button"
            @click="isEditing = true"
            :disabled="isProcessing"
            class="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3.5 py-2 text-sm font-medium text-text-secondary hover:bg-surface-secondary transition-colors disabled:opacity-40"
          >
            <svg
              class="h-3.5 w-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke-width="1.5"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.832 19.82a4.5 4.5 0 0 1-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 0 1 1.13-1.897L16.863 4.487Zm0 0L19.5 7.125"
              />
            </svg>
            Edit
          </button>
          <button
            v-if="canReject"
            type="button"
            @click="showRejectInput = true"
            :disabled="isProcessing"
            class="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3.5 py-2 text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/20 transition-colors disabled:opacity-40"
          >
            <svg
              class="h-3.5 w-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke-width="2"
              stroke="currentColor"
            >
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
            </svg>
            Reject
          </button>
        </template>
      </div>
    </div>
  </div>
</template>
