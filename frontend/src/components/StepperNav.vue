<script setup lang="ts">
import { Check } from 'lucide-vue-next'

defineProps<{
  steps: { label: string; sub: string }[]
  current: number
}>()
</script>

<template>
  <div class="glass p-4 md:p-6">
    <div class="flex items-center">
      <template v-for="(step, i) in steps" :key="i">
        <div class="flex flex-1 flex-col items-center text-center">
          <div
            class="flex h-10 w-10 items-center justify-center rounded-full border-2 text-sm font-bold transition-all duration-300"
            :class="
              i < current
                ? 'border-success bg-success text-white'
                : i === current
                  ? 'border-accent bg-accent text-white shadow-lg shadow-accent/40'
                  : 'border-line bg-surface-2 text-fg-subtle'
            "
          >
            <Check v-if="i < current" class="h-5 w-5" />
            <span v-else>{{ i + 1 }}</span>
          </div>
          <div class="mt-2 hidden sm:block">
            <div
              class="text-sm font-semibold"
              :class="i <= current ? 'text-fg' : 'text-fg-subtle'"
            >
              {{ step.label }}
            </div>
            <div class="text-[11px] text-fg-subtle">{{ step.sub }}</div>
          </div>
        </div>
        <div
          v-if="i < steps.length - 1"
          class="mx-1 h-0.5 flex-1 rounded-full transition-all duration-500"
          :class="i < current ? 'bg-success' : 'bg-surface-2'"
        />
      </template>
    </div>
  </div>
</template>
