<script setup lang="ts">
import { computed, ref } from 'vue'
import TrendChart from './components/TrendChart.vue'
import { useScreenStage } from './useScreenStage'

const { host, width, height, scale, mode, stageStyle } = useScreenStage()
type Strategy = 'fixed' | 'hybrid' | 'fluid'
const strategy = ref<Strategy>('hybrid')
const dense = ref(false)
const compact = computed(() => dense.value || (strategy.value === 'hybrid' && mode.value === 'compact'))
const displaySize = computed(() => `${Math.round(width.value)} × ${Math.round(height.value)}`)
const strategyName = computed(() => ({ fixed: '固定画布', hybrid: '混合布局', fluid: '完全响应式' })[strategy.value])
</script>

<template>
  <main ref="host" class="viewport" :class="{ 'viewport--fluid': strategy === 'fluid' }">
    <div class="background" aria-hidden="true" />
    <div
      class="stage"
      :class="[`stage--${strategy}`, { 'stage--compact': strategy === 'hybrid' && mode === 'compact', 'stage--wide': strategy === 'hybrid' && mode === 'wide' }]"
      :style="strategy === 'fluid' ? undefined : stageStyle"
    >
      <header class="header">
        <span class="eyebrow">SCREEN ADAPTATION LAB</span>
        <h1>城市运行数字大屏</h1>
        <span class="clock">示例数据 · 2026</span>
      </header>

      <div class="content" :class="{ 'content--dense': dense }">
        <aside class="panel side">
          <h2>关键指标</h2>
          <div class="metric"><span>服务企业</span><strong>12,860</strong><small>家</small></div>
          <div class="metric"><span>今日访问</span><strong>8,432</strong><small>次</small></div>
          <div class="metric"><span>在线设备</span><strong>1,024</strong><small>台</small></div>
          <p class="note">重要数字保持可读。窗口过窄时，优先隐藏次要说明，而不是无限缩小。</p>
        </aside>

        <section class="center">
          <div class="hero panel">
            <span class="eyebrow">区域态势</span>
            <strong>运行平稳</strong>
            <p>{{ strategy === 'fluid' ? 'Grid 重排 · Flex 填充 · 容器图表' : '基准画布 1920 × 1080 · 等比缩放 · 居中展示' }}</p>
          </div>
          <div class="panel trend-panel">
            <div class="panel-title"><h2>业务趋势</h2><span>单位：万</span></div>
            <TrendChart :compact="compact" />
          </div>
        </section>

        <aside class="panel side right">
          <h2>运营概览</h2>
          <div class="progress"><span>任务完成率</span><strong>86%</strong><i style="--amount: 86%" /></div>
          <div class="progress"><span>设备在线率</span><strong>94%</strong><i style="--amount: 94%" /></div>
          <div class="progress"><span>告警处理率</span><strong>79%</strong><i style="--amount: 79%" /></div>
          <p class="note">切换信息密度会改变图表容器尺寸。图表通过 ResizeObserver 重新布局。</p>
        </aside>
      </div>

      <footer class="footer">
        <label class="strategy-picker">方案：
          <select v-model="strategy" aria-label="选择大屏适配方案">
            <option value="fixed">固定画布</option>
            <option value="hybrid">混合布局</option>
            <option value="fluid">完全响应式</option>
          </select>
        </label>
        <span>{{ strategyName }} · {{ mode }}</span>
        <span>视口：{{ displaySize }}</span>
        <span>画布缩放：{{ strategy === 'fluid' ? '无' : scale.toFixed(3) }}</span>
        <button type="button" @click="dense = !dense">{{ dense ? '普通密度' : '紧凑密度' }}</button>
      </footer>
    </div>
  </main>
</template>
