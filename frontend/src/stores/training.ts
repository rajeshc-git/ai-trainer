import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api } from '@/lib/api'
import type { JobStatus } from '@/lib/api'

export interface LogLine {
  ts: number
  level: 'INFO' | 'WARNING' | 'ERROR'
  message: string
}

export interface LossPoint {
  step: number
  loss: number
}

/**
 * Global training state store.
 *
 * Owns the WebSocket connection to the backend training stream so that live
 * status, logs, and the loss curve stay up-to-date even when the user
 * navigates away from the Train page. The WebSocket is opened once per
 * training run and closed only when the job finishes (or is cancelled/failed).
 *
 * Every reactive consumer (GPU-fan animation, dashboard job table, metric
 * cards) reads from this store — no polling required.
 */
export const useTrainingStore = defineStore('training', () => {
  const jobId = ref<string | null>(null)
  const status = ref<JobStatus | null>(null)
  const logs = ref<LogLine[]>([])
  const lossCurve = ref<LossPoint[]>([])
  const jobs = ref<JobStatus[]>([])

  const isRunning = computed(
    () => status.value?.status === 'running' || status.value?.status === 'queued',
  )

  // ── WebSocket state (persistent across navigations) ──────────
  const wsConnected = ref(false)
  const wsReconnecting = ref(false)
  let ws: WebSocket | null = null
  let wsUrl = ''
  let retries = 0
  let stopped = true
  let retryTimer: number | undefined
  const MAX_RETRIES = 10

  /** Called by consumers to receive terminal-status toasts. */
  let _onDone: ((terminal: string) => void) | null = null

  function onDone(cb: (terminal: string) => void): void {
    _onDone = cb
  }

  function clearOnDone(): void {
    _onDone = null
  }

  // ── WebSocket internals ──────────────────────────────────────
  function _openWs(): void {
    if (stopped) return
    try {
      ws = new WebSocket(wsUrl)
    } catch {
      _scheduleReconnect()
      return
    }

    ws.onopen = () => {
      wsConnected.value = true
      wsReconnecting.value = false
      retries = 0
    }

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data as string)
        if (data.type === 'status' && data.status) {
          applyStatus(_normalizeStatus(data.status))
        } else if (data.type === 'done') {
          // Pin the terminal status so the UI updates even if the final
          // 'status' frame was missed.
          if (status.value) {
            status.value = {
              ...status.value,
              status: data.status,
              finished_at: Math.floor(Date.now() / 1000),
            }
            applyStatus(status.value)
          }
          _onDone?.(data.status)
          disconnectWs()
          refreshJobs()
        } else if (data.level && data.message) {
          addLog(data)
        }
      } catch {
        /* ignore malformed frames */
      }
    }

    ws.onclose = () => {
      wsConnected.value = false
      if (!stopped) _scheduleReconnect()
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  function _scheduleReconnect(): void {
    if (stopped || retries >= MAX_RETRIES) return
    wsReconnecting.value = true
    const delay = Math.min(1000 * 2 ** retries, 10000)
    retries += 1
    retryTimer = window.setTimeout(_openWs, delay)
  }

  function _normalizeStatus(s: Record<string, any>): any {
    return {
      job_id: s.job_id ?? jobId.value,
      status: s.status ?? 'running',
      percent: s.percent ?? 0,
      current_epoch: s.current_epoch ?? 0,
      total_epochs: s.total_epochs ?? 0,
      current_step: s.current_step ?? 0,
      total_steps: s.total_steps ?? 0,
      loss: s.loss ?? null,
      learning_rate: s.learning_rate ?? null,
      eta_seconds: s.eta_seconds ?? null,
      gpu_memory_mb: s.gpu_memory_mb ?? null,
      gpu_total_mb: s.gpu_total_mb ?? null,
      gpu_memory_percent: s.gpu_memory_percent ?? null,
      model_name: s.model_name ?? null,
      started_at: s.started_at ?? null,
      finished_at: s.finished_at ?? null,
      error: s.error ?? null,
      suggestion: s.suggestion ?? null,
    }
  }

  // ── Public API ───────────────────────────────────────────────

  /** Open (or reopen) the WebSocket for a given training job. */
  function connectWs(url: string): void {
    // Close any lingering connection first.
    disconnectWs()
    wsUrl = url
    stopped = false
    retries = 0
    _openWs()
  }

  /** Gracefully close the WebSocket (called on job completion). */
  function disconnectWs(): void {
    stopped = true
    if (retryTimer) window.clearTimeout(retryTimer)
    ws?.close()
    ws = null
    wsConnected.value = false
    wsReconnecting.value = false
  }

  function reset(id: string | null): void {
    jobId.value = id
    status.value = null
    logs.value = []
    lossCurve.value = []
  }

  function applyStatus(s: JobStatus): void {
    status.value = s
    if (s.loss != null && s.current_step != null) {
      const last = lossCurve.value[lossCurve.value.length - 1]
      if (!last || last.step !== s.current_step) {
        lossCurve.value.push({ step: s.current_step, loss: s.loss })
        if (lossCurve.value.length > 1000) lossCurve.value.shift()
      }
    }
    const idx = jobs.value.findIndex((j) => j.job_id === s.job_id)
    if (idx !== -1) {
      jobs.value[idx] = { ...jobs.value[idx], ...s }
    } else {
      refreshJobs()
    }
  }

  function addLog(line: LogLine): void {
    logs.value.push(line)
    if (logs.value.length > 1000) logs.value.shift()
  }

  async function refreshJobs(): Promise<void> {
    const { jobs: list } = await api.listJobs()
    jobs.value = list
  }

  /**
   * Reconcile the store's running state with the real API job list.
   * Safety net for edge cases (e.g. WebSocket never connected, browser
   * was suspended, etc.).
   */
  function syncFromJobs(apiJobs: JobStatus[]): void {
    if (!isRunning.value) return
    const anyActive = apiJobs.some(
      (j) => j.status === 'running' || j.status === 'queued',
    )
    if (!anyActive && status.value) {
      status.value = { ...status.value, status: 'completed' }
      disconnectWs()
    }
  }

  return {
    // State
    jobId,
    status,
    logs,
    lossCurve,
    jobs,
    isRunning,
    // WebSocket state (read-only for templates)
    wsConnected,
    wsReconnecting,
    // Actions
    reset,
    applyStatus,
    addLog,
    refreshJobs,
    syncFromJobs,
    connectWs,
    disconnectWs,
    onDone,
    clearOnDone,
  }
})
