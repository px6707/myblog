<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from "vue";
import type { ToolCallFromTool } from "@langchain/vue";
import type { weatherTool } from "@langchain/playground-agents";

const props = defineProps<{
  toolCall: ToolCallFromTool<typeof weatherTool>;
}>();

type WeatherTheme = "sunny" | "cloudy" | "rainy";

function getWeatherTheme(condition: string): WeatherTheme {
  const c = condition.toLowerCase();
  if (c.includes("sunny") || c.includes("clear")) return "sunny";
  if (c.includes("rain") || c.includes("storm") || c.includes("drizzle")) return "rainy";
  return "cloudy";
}

function getConditionIcon(condition: string): "sunny" | "rain" | "partly" | "cloud" {
  const c = condition.toLowerCase();
  if (c.includes("sunny") || c.includes("clear")) return "sunny";
  if (c.includes("rain")) return "rain";
  if (c.includes("partly")) return "partly";
  return "cloud";
}

const weatherFont = '"SF Pro Display", "SF Pro Text", Inter, "Noto Sans", system-ui, sans-serif';

const weatherGradients: Record<WeatherTheme, { bg: string; overlay: string }> = {
  sunny: {
    bg: "linear-gradient(135deg, #38bdf8 0%, #0ea5e9 40%, #2563eb 100%)",
    overlay: "radial-gradient(ellipse at 80% 20%, rgba(250,204,21,0.25) 0%, transparent 60%)",
  },
  cloudy: {
    bg: "linear-gradient(135deg, #94a3b8 0%, #64748b 50%, #475569 100%)",
    overlay: "radial-gradient(ellipse at 30% 30%, rgba(203,213,225,0.2) 0%, transparent 60%)",
  },
  rainy: {
    bg: "linear-gradient(135deg, #64748b 0%, #475569 40%, #334155 100%)",
    overlay: "radial-gradient(ellipse at 50% 0%, rgba(148,163,184,0.15) 0%, transparent 50%)",
  },
};

const weatherGradientsDark: Record<WeatherTheme, { bg: string; overlay: string }> = {
  sunny: {
    bg: "linear-gradient(135deg, #1e3a5f 0%, #0c4a6e 40%, #1e40af 100%)",
    overlay: "radial-gradient(ellipse at 80% 20%, rgba(250,204,21,0.12) 0%, transparent 60%)",
  },
  cloudy: {
    bg: "linear-gradient(135deg, #334155 0%, #1e293b 50%, #0f172a 100%)",
    overlay: "radial-gradient(ellipse at 30% 30%, rgba(148,163,184,0.08) 0%, transparent 60%)",
  },
  rainy: {
    bg: "linear-gradient(135deg, #1e293b 0%, #0f172a 40%, #020617 100%)",
    overlay: "radial-gradient(ellipse at 50% 0%, rgba(100,116,139,0.1) 0%, transparent 50%)",
  },
};

function sineEasedGradient(
  x: number,
  y: number,
  radius: number,
  peakOpacity: number,
  steps = 8,
): string {
  const stops: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const eased = Math.sin((t * Math.PI) / 2);
    const opacity = peakOpacity * (1 - eased);
    const position = t * 100;
    stops.push(`rgba(255,255,255,${opacity.toFixed(4)}) ${position.toFixed(1)}%`);
  }
  return `radial-gradient(circle ${radius}px at ${x}px ${y}px, ${stops.join(", ")})`;
}

const cardRef = ref<HTMLDivElement | null>(null);
const glowX = ref(0);
const glowY = ref(0);
const glowIntensity = ref(0);
const isDark = ref(false);

let observer: MutationObserver | null = null;

onMounted(() => {
  isDark.value = document.documentElement.classList.contains("dark");
  observer = new MutationObserver(() => {
    isDark.value = document.documentElement.classList.contains("dark");
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
});

onUnmounted(() => {
  observer?.disconnect();
});

const theme = computed(() => {
  if (!props.toolCall.output) return "cloudy" as WeatherTheme;
  return getWeatherTheme(props.toolCall.output.condition);
});

const gradients = computed(() =>
  isDark.value ? weatherGradientsDark[theme.value] : weatherGradients[theme.value],
);

const pendingGradient = computed(() =>
  isDark.value
    ? "linear-gradient(135deg, #334155 0%, #1e293b 100%)"
    : "linear-gradient(135deg, #94a3b8 0%, #64748b 100%)",
);

const conditionIcon = computed(() => {
  if (!props.toolCall.output) return "cloud";
  return getConditionIcon(props.toolCall.output.condition);
});

const unitSymbol = computed(() => (props.toolCall.output?.unit === "fahrenheit" ? "F" : "C"));

const glowGradient = computed(() =>
  sineEasedGradient(glowX.value, glowY.value, 120, isDark.value ? 0.12 : 0.2),
);
const edgeShineGradient = computed(() =>
  sineEasedGradient(glowX.value, glowY.value, 100, isDark.value ? 0.5 : 0.8),
);

const rainDrops = Array.from({ length: 20 }, (_, i) => ({
  left: `${5 + i * 4.5}%`,
  height: `${6 + (i % 4) * 3}px`,
  duration: `${0.7 + (i % 3) * 0.3}s`,
  delay: `${(i * 0.08) % 1}s`,
}));

function onMouseMove(e: MouseEvent) {
  const card = cardRef.value;
  if (!card) return;
  const rect = card.getBoundingClientRect();
  glowX.value = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
  glowY.value = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
  const dx =
    e.clientX < rect.left
      ? rect.left - e.clientX
      : e.clientX > rect.right
        ? e.clientX - rect.right
        : 0;
  const dy =
    e.clientY < rect.top
      ? rect.top - e.clientY
      : e.clientY > rect.bottom
        ? e.clientY - rect.bottom
        : 0;
  glowIntensity.value = Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy) / 120);
}

function onMouseLeave() {
  glowIntensity.value = 0;
}
</script>

<template>
  <div
    v-if="!toolCall.output"
    class="relative overflow-hidden rounded-2xl p-5"
    data-testid="sdk-preview-chat-turn"
    :style="{ background: pendingGradient, fontFamily: weatherFont }"
  >
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-full bg-white/10 backdrop-blur-sm animate-pulse" />
      <div class="flex-1 space-y-2">
        <div
          class="text-sm font-medium text-white/90"
          style="text-shadow: 0 1px 4px rgba(0, 0, 0, 0.2)"
        >
          Checking weather in {{ toolCall.input.city }}…
        </div>
        <div class="flex gap-2">
          <div class="h-2 w-16 rounded-full bg-white/15 animate-pulse" />
          <div
            class="h-2 w-10 rounded-full bg-white/10 animate-pulse"
            style="animation-delay: 0.15s"
          />
        </div>
      </div>
    </div>
  </div>

  <div
    v-else
    ref="cardRef"
    class="relative overflow-hidden rounded-2xl select-none"
    data-testid="sdk-preview-chat-turn"
    :style="{ fontFamily: weatherFont }"
    @mousemove="onMouseMove"
    @mouseleave="onMouseLeave"
  >
    <div class="absolute inset-0" :style="{ background: gradients.bg }" />
    <div class="absolute inset-0" :style="{ background: gradients.overlay }" />
    <div
      class="absolute inset-0 pointer-events-none transition-opacity duration-300 ease-out"
      :style="{ opacity: glowIntensity, background: glowGradient }"
    />
    <div
      v-if="theme === 'rainy'"
      class="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none"
    >
      <div
        v-for="(drop, i) in rainDrops"
        :key="i"
        class="absolute rounded-full"
        :style="{
          left: drop.left,
          width: '1px',
          height: drop.height,
          background: 'linear-gradient(to bottom, transparent, rgba(147,197,253,0.4))',
          animationName: 'rain',
          animationDuration: drop.duration,
          animationTimingFunction: 'linear',
          animationIterationCount: 'infinite',
          animationDelay: drop.delay,
        }"
      />
    </div>

    <div class="relative p-5">
      <div class="flex items-start justify-between">
        <div>
          <h3
            class="text-[11px] font-semibold uppercase tracking-widest text-white/70 mb-1"
            style="text-shadow: 0 1px 3px rgba(0, 0, 0, 0.15)"
          >
            {{ toolCall.output.city }}
          </h3>
          <div class="flex items-start">
            <span
              class="font-[250] tabular-nums leading-none tracking-tight text-white/95"
              :style="{
                fontSize: '48px',
                fontFeatureSettings: '\'tnum\' 1, \'case\' 1',
                textShadow: isDark
                  ? '0 2px 16px rgba(0,0,0,0.3)'
                  : '0 2px 16px rgba(255,255,255,0.2)',
              }"
            >
              {{ toolCall.output.temperature }}
            </span>
            <span
              class="mt-1 ml-0.5 font-light tabular-nums text-white/60"
              :style="{ fontSize: '18px', fontFeatureSettings: '\'tnum\' 1, \'case\' 1' }"
            >
              °{{ unitSymbol }}
            </span>
          </div>
          <div
            class="text-[13px] mt-1 capitalize text-white/75 font-medium tracking-wide"
            style="text-shadow: 0 1px 3px rgba(0, 0, 0, 0.1)"
          >
            {{ toolCall.output.condition }}
          </div>
        </div>
        <div class="mt-1">
          <svg
            v-if="conditionIcon === 'sunny'"
            class="w-12 h-12 drop-shadow-lg"
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle cx="12" cy="12" r="4" fill="#fbbf24" opacity="0.9" />
            <circle cx="12" cy="12" r="5.5" stroke="#fbbf24" stroke-width="0.6" opacity="0.3" />
            <path
              d="M12 3v2.5M12 18.5V21M5.64 5.64l1.77 1.77M16.59 16.59l1.77 1.77M3 12h2.5M18.5 12H21M5.64 18.36l1.77-1.77M16.59 7.41l1.77-1.77"
              stroke="#fcd34d"
              stroke-width="1.5"
              stroke-linecap="round"
              opacity="0.8"
            />
          </svg>
          <svg
            v-else-if="conditionIcon === 'rain'"
            class="w-12 h-12 drop-shadow-lg"
            viewBox="0 0 24 24"
            fill="none"
          >
            <path
              d="M4 14.2A4 4 0 0 1 6.2 7a5.5 5.5 0 0 1 10.6-.5A4.5 4.5 0 0 1 18 15H6a4 4 0 0 1-2-.8z"
              fill="rgba(255,255,255,0.25)"
              stroke="rgba(255,255,255,0.4)"
              stroke-width="0.8"
              stroke-linejoin="round"
            />
            <path
              d="M8.5 17.5v2M12 17.5v2M15.5 17.5v2"
              stroke="rgba(147,197,253,0.8)"
              stroke-width="1.5"
              stroke-linecap="round"
            />
          </svg>
          <svg
            v-else-if="conditionIcon === 'partly'"
            class="w-12 h-12 drop-shadow-lg"
            viewBox="0 0 24 24"
            fill="none"
          >
            <circle cx="9" cy="7" r="3" fill="#fbbf24" opacity="0.7" />
            <path
              d="M9 2.5v1.5M9 10.5v1.5M5.17 3.67l1.06 1.06M11.77 10.27l1.06 1.06M3 7h1.5M13 7h1.5M5.17 10.33l1.06-1.06M11.77 3.73l1.06-1.06"
              stroke="#fcd34d"
              stroke-width="1"
              stroke-linecap="round"
              opacity="0.5"
            />
            <path
              d="M7 17.2A3.5 3.5 0 0 1 8.8 11a5 5 0 0 1 9.2-.5A4 4 0 0 1 19 18H9a3.5 3.5 0 0 1-2-.8z"
              fill="rgba(255,255,255,0.3)"
              stroke="rgba(255,255,255,0.45)"
              stroke-width="0.8"
              stroke-linejoin="round"
            />
          </svg>
          <svg v-else class="w-12 h-12 drop-shadow-lg" viewBox="0 0 24 24" fill="none">
            <path
              d="M4 17.2A4 4 0 0 1 6.2 10a5.5 5.5 0 0 1 10.6-.5A4.5 4.5 0 0 1 18 18H6a4 4 0 0 1-2-.8z"
              :fill="theme === 'cloudy' ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.2)'"
              stroke="rgba(255,255,255,0.4)"
              stroke-width="0.8"
              stroke-linejoin="round"
            />
          </svg>
        </div>
      </div>
    </div>

    <div
      class="absolute inset-0 rounded-2xl pointer-events-none transition-opacity duration-300 ease-out"
      :style="{
        opacity: glowIntensity * 0.6,
        background: edgeShineGradient,
        mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
        maskComposite: 'exclude',
        WebkitMaskComposite: 'xor',
        padding: '0.5px',
      }"
    />
  </div>
</template>
