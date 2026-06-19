import { onUnmounted, ref } from 'vue'

/**
 * Auto-reconnecting WebSocket helper.
 *
 * Handles disconnects during training by transparently reconnecting with a
 * capped backoff. Exposes `connected` state and lifecycle callbacks. The
 * caller owns message parsing.
 */
export interface UseWebSocketOptions {
  onMessage: (raw: string) => void
  onOpen?: () => void
  onClose?: () => void
  maxRetries?: number
}

export function useWebSocket(options: UseWebSocketOptions) {
  const connected = ref(false)
  const reconnecting = ref(false)
  let ws: WebSocket | null = null
  let url = ''
  let retries = 0
  let stopped = false
  let retryTimer: number | undefined
  const maxRetries = options.maxRetries ?? 10

  function connect(targetUrl: string): void {
    url = targetUrl
    stopped = false
    open()
  }

  function open(): void {
    if (stopped) return
    try {
      ws = new WebSocket(url)
    } catch {
      scheduleReconnect()
      return
    }

    ws.onopen = () => {
      connected.value = true
      reconnecting.value = false
      retries = 0
      options.onOpen?.()
    }
    ws.onmessage = (ev) => options.onMessage(ev.data as string)
    ws.onclose = () => {
      connected.value = false
      options.onClose?.()
      if (!stopped) scheduleReconnect()
    }
    ws.onerror = () => {
      // onclose will follow and handle the reconnect.
      ws?.close()
    }
  }

  function scheduleReconnect(): void {
    if (stopped || retries >= maxRetries) return
    reconnecting.value = true
    // Exponential backoff starting at 1s: 1, 2, 4, … capped at 10s.
    const delay = Math.min(1000 * 2 ** retries, 10000)
    retries += 1
    retryTimer = window.setTimeout(open, delay)
  }

  function close(): void {
    stopped = true
    if (retryTimer) window.clearTimeout(retryTimer)
    ws?.close()
    ws = null
    connected.value = false
  }

  onUnmounted(close)

  return { connected, reconnecting, connect, close }
}
