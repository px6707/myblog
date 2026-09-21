import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

const DESIGN_WIDTH = 1920
const DESIGN_HEIGHT = 1080

export function useScreenStage() {
  const host = ref<HTMLElement | null>(null)
  const width = ref(DESIGN_WIDTH)
  const height = ref(DESIGN_HEIGHT)
  let observer: ResizeObserver | null = null

  const scale = computed(() => Math.min(width.value / DESIGN_WIDTH, height.value / DESIGN_HEIGHT))
  const aspect = computed(() => width.value / Math.max(height.value, 1))
  const mode = computed(() => {
    if (aspect.value < 1.45 || scale.value < 0.68) return 'compact'
    if (aspect.value > 2.15) return 'wide'
    return 'standard'
  })
  const stageStyle = computed(() => ({
    width: `${DESIGN_WIDTH}px`,
    height: `${DESIGN_HEIGHT}px`,
    transform: `scale(${scale.value})`,
    transformOrigin: 'center center',
  }))

  onMounted(() => {
    if (!host.value) return
    observer = new ResizeObserver(([entry]) => {
      if (!entry) return
      const { width: nextWidth, height: nextHeight } = entry.contentRect
      // 只有尺寸真的变化才写入响应式状态；不伪造 window.resize。
      if (nextWidth !== width.value) width.value = nextWidth
      if (nextHeight !== height.value) height.value = nextHeight
    })
    observer.observe(host.value)
  })

  onBeforeUnmount(() => observer?.disconnect())

  return { host, width, height, scale, mode, stageStyle }
}
