import { defineStore } from 'pinia'
import { computed, watch } from 'vue'
import { useStorage } from '@vueuse/core'

export type Theme = 'light' | 'dark'

/**
 * App theme (light default, dark optional). Persisted to localStorage under
 * `ft-theme` and applied by toggling the `dark` class on <html> — the same key
 * the inline boot script in index.html reads to avoid a flash on first paint.
 */
export const useThemeStore = defineStore('theme', () => {
  const theme = useStorage<Theme>('ft-theme', 'light')
  const isDark = computed(() => theme.value === 'dark')

  function apply(t: Theme): void {
    document.documentElement.classList.toggle('dark', t === 'dark')
  }

  function set(t: Theme): void {
    theme.value = t
  }

  function toggle(): void {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  // Keep the DOM class in sync with the stored value (covers cross-tab changes).
  watch(theme, apply, { immediate: true })

  return { theme, isDark, set, toggle }
})
