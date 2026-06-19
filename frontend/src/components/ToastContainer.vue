<script setup lang="ts">
import { CheckCircle2, XCircle, Info, AlertTriangle, X } from 'lucide-vue-next'
import { useToastStore } from '@/stores/toast'

const toast = useToastStore()

const icons = {
  success: CheckCircle2,
  error: XCircle,
  info: Info,
  warning: AlertTriangle,
}
const accents = {
  success: 'text-success border-success/30',
  error: 'text-danger border-danger/30',
  info: 'text-accent border-accent/30',
  warning: 'text-warn border-warn/30',
}
</script>

<template>
  <div class="pointer-events-none fixed right-4 top-4 z-[60] flex w-full max-w-sm flex-col gap-3">
    <TransitionGroup
      enter-active-class="transition duration-300 ease-out"
      enter-from-class="translate-x-8 opacity-0"
      enter-to-class="translate-x-0 opacity-100"
      leave-active-class="transition duration-200 ease-in"
      leave-from-class="translate-x-0 opacity-100"
      leave-to-class="translate-x-8 opacity-0"
    >
      <div
        v-for="t in toast.toasts"
        :key="t.id"
        class="glass pointer-events-auto flex items-start gap-3 border-l-4 p-4"
        :class="accents[t.kind]"
      >
        <component :is="icons[t.kind]" class="mt-0.5 h-5 w-5 shrink-0" />
        <div class="min-w-0 flex-1">
          <div class="text-sm font-semibold text-fg">{{ t.title }}</div>
          <div v-if="t.message" class="mt-0.5 text-xs text-fg-muted">{{ t.message }}</div>
        </div>
        <button
          class="shrink-0 text-fg-muted transition hover:text-fg"
          @click="toast.dismiss(t.id)"
        >
          <X class="h-4 w-4" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>
