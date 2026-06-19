<script setup lang="ts">
import { computed } from 'vue'
import { Box, Calendar, Database, TrendingDown, Download, Trash2, MessageSquare, Boxes } from 'lucide-vue-next'
import type { SavedModel } from '@/lib/api'

const props = defineProps<{ model: SavedModel }>()
const emit = defineEmits<{
  (e: 'download'): void
  (e: 'delete'): void
  (e: 'chat'): void
  (e: 'export'): void
}>()

const created = computed(() =>
  props.model.created_at
    ? new Date(props.model.created_at * 1000).toLocaleDateString()
    : '—',
)
const size = computed(() => {
  const b = props.model.size_bytes
  if (!b) return '—'
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`
  return `${(b / (1024 * 1024)).toFixed(1)} MB`
})
</script>

<template>
  <div class="glass glass-hover flex flex-col p-5">
    <div class="flex items-start justify-between gap-3 min-w-0">
      <div class="flex items-center gap-3 min-w-0">
        <div class="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent/15 text-accent">
          <Box class="h-5 w-5" />
        </div>
        <div class="min-w-0">
          <h3 class="truncate font-bold text-fg">{{ model.name }}</h3>
          <p class="truncate text-xs text-fg-muted">base: {{ model.base_model || 'unknown' }}</p>
        </div>
      </div>
      <span class="badge shrink-0 bg-success/15 text-success">{{ size }}</span>
    </div>

    <dl class="mt-4 grid grid-cols-3 gap-3 text-center">
      <div class="rounded-lg bg-surface-2 p-2">
        <Calendar class="mx-auto h-4 w-4 text-fg-muted" />
        <dd class="mt-1 text-xs font-semibold text-fg">{{ created }}</dd>
        <dt class="text-[10px] text-fg-subtle">trained</dt>
      </div>
      <div class="rounded-lg bg-surface-2 p-2">
        <Database class="mx-auto h-4 w-4 text-fg-muted" />
        <dd class="mt-1 text-xs font-semibold text-fg">{{ model.dataset_rows ?? '—' }}</dd>
        <dt class="text-[10px] text-fg-subtle">rows</dt>
      </div>
      <div class="rounded-lg bg-surface-2 p-2">
        <TrendingDown class="mx-auto h-4 w-4 text-fg-muted" />
        <dd class="mt-1 text-xs font-semibold text-fg">
          {{ model.final_loss != null ? model.final_loss.toFixed(3) : '—' }}
        </dd>
        <dt class="text-[10px] text-fg-subtle">loss</dt>
      </div>
    </dl>

    <div class="mt-5 flex gap-2">
      <button class="btn-gradient flex-1 !py-2 text-sm" @click="emit('chat')">
        <MessageSquare class="h-4 w-4" /> Chat
      </button>
      <button class="btn-ghost !px-3 !py-2" title="Download adapter (.zip)" @click="emit('download')">
        <Download class="h-4 w-4" />
      </button>
      <button
        class="btn-ghost !px-3 !py-2 hover:!border-danger/40 hover:!text-danger"
        title="Delete"
        @click="emit('delete')"
      >
        <Trash2 class="h-4 w-4" />
      </button>
    </div>

    <button
      class="btn-ghost mt-2 w-full !py-2 text-sm"
      title="Export a quantized GGUF for Ollama / llama.cpp"
      @click="emit('export')"
    >
      <Boxes class="h-4 w-4" /> Export to GGUF
    </button>
  </div>
</template>
