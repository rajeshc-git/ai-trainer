import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '@/lib/api'

/**
 * Tracks backend health + GPU availability. Polls periodically so the
 * navbar badge stays accurate (e.g. GPU appears after the container warms up).
 */
export const useGpuStore = defineStore('gpu', () => {
  const online = ref(false)
  const hasGpu = ref(false)
  const gpuName = ref<string | null>(null)
  const totalMb = ref<number | null>(null)
  const checking = ref(true)
  let timer: number | undefined

  async function refresh(): Promise<void> {
    try {
      const h = await api.health()
      online.value = true
      hasGpu.value = h.gpu
      gpuName.value = h.gpu_name
      totalMb.value = h.gpu_total_mb
    } catch {
      online.value = false
      hasGpu.value = false
      gpuName.value = null
      totalMb.value = null
    } finally {
      checking.value = false
    }
  }

  function startPolling(intervalMs = 15000): void {
    void refresh()
    timer = window.setInterval(refresh, intervalMs)
  }

  function stopPolling(): void {
    if (timer) window.clearInterval(timer)
  }

  return { online, hasGpu, gpuName, totalMb, checking, refresh, startPolling, stopPolling }
})
