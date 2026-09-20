<script setup lang="ts">
import { computed, ref } from "vue";
import type {
  FormField,
  FormType,
  InterruptCard,
  ReviewDecision,
} from "@langchain/playground-agents";

const props = defineProps<{
  card: InterruptCard;
  readOnly?: boolean;
  isProcessing?: boolean;
}>();

const emit = defineEmits<{
  resolve: [decision: ReviewDecision];
}>();

type HeaderMeta = {
  iconPath: string;
  accent: string;
  approveLabel: string;
  rejectLabel: string;
};

const HEADER: Record<FormType, HeaderMeta> = {
  "flight-booking": {
    iconPath:
      "M6 12 3.27 4.36A.6.6 0 0 1 4.1 3.6l15.5 7.86a.6.6 0 0 1 0 1.08L4.1 20.4a.6.6 0 0 1-.83-.76L6 12Zm0 0h6",
    accent: "text-sky-600 dark:text-sky-400",
    approveLabel: "Confirm booking",
    rejectLabel: "Cancel",
  },
  "refund-approval": {
    iconPath:
      "M9 8.25h6M9 12h6m-6 3.75h3M5.25 3.75v16.5l2.25-1.5 2.25 1.5 2.25-1.5 2.25 1.5 2.25-1.5 2.25 1.5V3.75l-2.25 1.5-2.25-1.5-2.25 1.5-2.25-1.5-2.25 1.5-2.25-1.5Z",
    accent: "text-amber-600 dark:text-amber-400",
    approveLabel: "Approve refund",
    rejectLabel: "Reject",
  },
  "content-review": {
    iconPath:
      "M10.34 15.84c-.688.06-1.386.09-2.09.09H6.75A2.25 2.25 0 0 1 4.5 13.68V10.32A2.25 2.25 0 0 1 6.75 8.07h1.5c.704 0 1.402.03 2.09.09m0 7.68a24.3 24.3 0 0 1 5.16 2.025c.243.143.546-.028.546-.31V6.06c0-.282-.303-.453-.546-.31a24.3 24.3 0 0 1-5.16 2.025m0 7.68V8.16",
    accent: "text-violet-600 dark:text-violet-400",
    approveLabel: "Publish",
    rejectLabel: "Discard",
  },
};

function initialValues(fields: FormField[]): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const field of fields) {
    values[field.name] = field.default ?? (field.type === "checkbox" ? false : "");
  }
  return values;
}

const values = ref<Record<string, unknown>>(
  props.card.resolved && props.card.decision?.values
    ? { ...initialValues(props.card.fields), ...props.card.decision.values }
    : initialValues(props.card.fields),
);

const meta = computed(() => HEADER[props.card.formType]);
const resolvedDecision = computed(() => (props.card.resolved ? props.card.decision : undefined));
const interactive = computed(() => !props.readOnly && !props.card.resolved);

const inputClass =
  "w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-text focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary disabled:opacity-60";

function onCurrencyInput(name: string, raw: string) {
  values.value[name] = raw === "" ? "" : Number(raw);
}
</script>

<template>
  <div
    class="rounded-xl border border-border bg-surface overflow-hidden"
    data-testid="sdk-preview-chat-turn"
  >
    <div class="flex items-center gap-2 px-4 py-2.5 border-b border-border bg-surface-secondary">
      <svg
        :class="['h-4 w-4', meta.accent]"
        fill="none"
        viewBox="0 0 24 24"
        stroke-width="1.6"
        stroke="currentColor"
      >
        <path stroke-linecap="round" stroke-linejoin="round" :d="meta.iconPath" />
      </svg>
      <span class="text-sm font-medium text-text">{{ card.title }}</span>
      <span
        v-if="resolvedDecision"
        :class="[
          'ml-auto text-[11px] font-medium px-2 py-0.5 rounded-full text-white',
          resolvedDecision.approved ? 'bg-emerald-600' : 'bg-red-600',
        ]"
      >
        {{ resolvedDecision.approved ? "Confirmed" : "Declined" }}
      </span>
      <span
        v-else
        class="ml-auto text-[11px] font-medium px-2 py-0.5 rounded-full bg-warning text-white"
      >
        Awaiting review
      </span>
    </div>

    <div class="p-4 space-y-3">
      <div
        v-if="Object.keys(card.context).length > 0"
        class="rounded-lg bg-surface-tertiary border border-border p-3 space-y-1.5"
      >
        <div
          v-for="[key, value] in Object.entries(card.context)"
          :key="key"
          class="flex items-baseline justify-between gap-3 text-sm"
        >
          <span class="text-text-tertiary">{{ key }}</span>
          <span class="font-medium text-text text-right wrap-break-word">{{ String(value) }}</span>
        </div>
      </div>

      <div class="space-y-3">
        <template v-for="field in card.fields" :key="field.name">
          <label
            v-if="field.type === 'checkbox'"
            :for="`field-${field.name}`"
            class="flex items-center gap-2 text-sm text-text cursor-pointer"
          >
            <input
              :id="`field-${field.name}`"
              type="checkbox"
              :checked="values[field.name] === true"
              :disabled="!interactive || isProcessing"
              @change="values[field.name] = ($event.target as HTMLInputElement).checked"
              class="h-4 w-4 rounded border-border text-primary focus:ring-primary/30"
            />
            {{ field.label }}
          </label>

          <div v-else>
            <label
              :for="`field-${field.name}`"
              class="block text-xs font-medium text-text-secondary mb-1.5"
            >
              {{ field.label }}
            </label>

            <select
              v-if="field.type === 'select'"
              :id="`field-${field.name}`"
              :value="String(values[field.name] ?? '')"
              :disabled="!interactive || isProcessing"
              @change="values[field.name] = ($event.target as HTMLSelectElement).value"
              :class="inputClass"
            >
              <option v-for="opt in field.options ?? []" :key="opt" :value="opt">{{ opt }}</option>
            </select>

            <textarea
              v-else-if="field.type === 'textarea'"
              :id="`field-${field.name}`"
              :rows="4"
              :value="String(values[field.name] ?? '')"
              :disabled="!interactive || isProcessing"
              @input="values[field.name] = ($event.target as HTMLTextAreaElement).value"
              :class="[inputClass, 'resize-y']"
            />

            <div v-else-if="field.type === 'currency'" class="relative">
              <span
                class="pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm text-text-tertiary"
              >
                {{ field.currency ?? "USD" }}
              </span>
              <input
                :id="`field-${field.name}`"
                type="number"
                :value="String(values[field.name] ?? '')"
                :disabled="!interactive || isProcessing"
                @input="onCurrencyInput(field.name, ($event.target as HTMLInputElement).value)"
                :class="[inputClass, 'pl-14']"
              />
            </div>
          </div>
        </template>
      </div>

      <div v-if="interactive" class="flex flex-wrap items-center gap-2 pt-1">
        <button
          type="button"
          :disabled="isProcessing"
          @click="emit('resolve', { approved: true, values })"
          class="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 px-3.5 py-2 text-sm font-medium text-white transition-colors disabled:opacity-40"
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
          {{ meta.approveLabel }}
        </button>
        <button
          type="button"
          :disabled="isProcessing"
          @click="emit('resolve', { approved: false, values })"
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
          {{ meta.rejectLabel }}
        </button>
      </div>
    </div>
  </div>
</template>
