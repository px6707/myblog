<script setup lang="ts">
import { nextTick, onActivated, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

const props = defineProps<{ compact: boolean }>()
const el = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null
let observer: ResizeObserver | null = null
let frame = 0
let previousWidth = 0
let previousHeight = 0

function option(width: number): echarts.EChartsOption {
  const narrow = props.compact || width < 420
  return {
    animationDuration: 350,
    color: ['#35d8ff'],
    grid: { left: narrow ? 12 : 20, right: 20, top: 20, bottom: 14, containLabel: true },
    tooltip: { trigger: 'axis', valueFormatter: (value) => `${value} 万` },
    xAxis: {
      type: 'category',
      data: ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月'],
      axisLabel: { color: '#a8bfd8', fontSize: narrow ? 14 : 16, interval: narrow ? 1 : 0 },
      axisLine: { lineStyle: { color: '#31516d' } },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: '#21394e' } },
      axisLabel: { color: '#a8bfd8', fontSize: narrow ? 14 : 16 },
    },
    series: [{
      type: 'line',
      smooth: true,
      symbolSize: narrow ? 5 : 8,
      areaStyle: { color: 'rgba(53,216,255,.18)' },
      lineStyle: { width: 3 },
      data: [32, 46, 42, 58, 55, 72, 69, 86],
    }],
  }
}

function update(width: number, height: number) {
  if (!el.value || width < 1 || height < 1) return
  if (!chart) chart = echarts.init(el.value, undefined, { renderer: 'canvas' })
  // resize 更新绘图尺寸；setOption 更新标签密度等内容策略。
  if (width !== previousWidth || height !== previousHeight) chart.resize()
  if (width !== previousWidth || !previousWidth) chart.setOption(option(width), true)
  previousWidth = width
  previousHeight = height
}

function schedule(width: number, height: number) {
  cancelAnimationFrame(frame)
  frame = requestAnimationFrame(() => update(width, height))
}

onMounted(async () => {
  await nextTick()
  if (!el.value) return
  observer = new ResizeObserver(([entry]) => {
    if (!entry) return
    schedule(entry.contentRect.width, entry.contentRect.height)
  })
  observer.observe(el.value)
  schedule(el.value.clientWidth, el.value.clientHeight)
})

watch(() => props.compact, () => {
  if (chart && el.value) chart.setOption(option(el.value.clientWidth), true)
})

// 如果组件被 KeepAlive 缓存，重新显示时再同步尺寸。
onActivated(() => {
  if (el.value) schedule(el.value.clientWidth, el.value.clientHeight)
})

onBeforeUnmount(() => {
  observer?.disconnect()
  cancelAnimationFrame(frame)
  chart?.dispose()
  chart = null
})
</script>

<template>
  <div ref="el" class="chart" role="img" aria-label="最近八个月业务量趋势图" />
</template>
