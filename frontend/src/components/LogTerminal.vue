<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import type { LogLine } from '@/stores/training'

const props = defineProps<{ lines: LogLine[] }>()
const box = ref<HTMLDivElement | null>(null)
const autoScroll = ref(true)

// Auto-scroll to the bottom as new lines arrive, unless the user scrolled up.
watch(
  () => props.lines.length,
  async () => {
    if (!autoScroll.value) return
    await nextTick()
    if (box.value) box.value.scrollTop = box.value.scrollHeight
  },
)

function onScroll(): void {
  if (!box.value) return
  const { scrollTop, scrollHeight, clientHeight } = box.value
  autoScroll.value = scrollHeight - scrollTop - clientHeight < 40
}

const levelClass: Record<string, string> = {
  INFO: 'text-sky-400',
  WARNING: 'text-amber-400',
  ERROR: 'text-rose-400',
}

function time(ts: number): string {
  try {
    return new Date(ts * 1000).toLocaleTimeString()
  } catch {
    return ''
  }
}
</script>

<template>
  <div
    ref="box"
    class="h-72 overflow-y-auto rounded-xl border border-line bg-slate-900 p-4 font-mono text-xs leading-relaxed"
    @scroll="onScroll"
  >
    <div v-if="!lines.length" class="text-slate-500">No logs yet…</div>
    <div v-for="(line, i) in lines" :key="i" class="whitespace-pre-wrap break-words">
      <span class="text-slate-500">{{ time(line.ts) }}</span>
      <span class="mx-2 font-bold" :class="levelClass[line.level] || 'text-slate-400'">
        {{ line.level }}
      </span>
      <span class="text-slate-200">{{ line.message }}</span>
    </div>
  </div>
</template>
