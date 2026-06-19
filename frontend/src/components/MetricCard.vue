<script setup lang="ts">
import type { Component } from 'vue'

defineProps<{
  label: string
  value: string | number
  icon?: Component
  accent?: 'indigo' | 'success' | 'danger' | 'warn'
  loading?: boolean
  pulse?: boolean
}>()

const accentMap: Record<string, string> = {
  indigo: 'text-accent bg-accent/15',
  success: 'text-success bg-success/15',
  danger: 'text-danger bg-danger/15',
  warn: 'text-warn bg-warn/15',
}
</script>

<template>
  <div class="glass glass-hover p-5" :class="pulse ? 'animate-pulse-ring' : ''">
    <div class="flex items-center justify-between">
      <span class="text-xs font-medium uppercase tracking-wide text-fg-muted">{{ label }}</span>
      <span
        v-if="icon"
        class="flex h-8 w-8 items-center justify-center rounded-lg"
        :class="accentMap[accent || 'indigo']"
      >
        <component :is="icon" class="h-4 w-4" />
      </span>
    </div>
    <div v-if="loading" class="skeleton mt-3 h-8 w-24" />
    <div v-else class="mt-2 text-2xl font-extrabold text-fg">{{ value }}</div>
  </div>
</template>
