import { defineStore } from 'pinia'
import { ref } from 'vue'

export type ToastKind = 'success' | 'error' | 'info' | 'warning'

export interface Toast {
  id: number
  kind: ToastKind
  title: string
  message?: string
}

/**
 * Global toast notifications shown top-right. Auto-dismiss after a timeout.
 */
export const useToastStore = defineStore('toast', () => {
  const toasts = ref<Toast[]>([])
  let seq = 0

  function push(kind: ToastKind, title: string, message?: string, ttl = 4500): void {
    const id = ++seq
    toasts.value.push({ id, kind, title, message })
    window.setTimeout(() => dismiss(id), ttl)
  }

  function dismiss(id: number): void {
    toasts.value = toasts.value.filter((t) => t.id !== id)
  }

  const success = (t: string, m?: string) => push('success', t, m)
  const error = (t: string, m?: string) => push('error', t, m, 7000)
  const info = (t: string, m?: string) => push('info', t, m)
  const warning = (t: string, m?: string) => push('warning', t, m)

  return { toasts, push, dismiss, success, error, info, warning }
})
