<script setup lang="ts">
import { ref } from 'vue'
import { UploadCloud, FileText } from 'lucide-vue-next'

const emit = defineEmits<{ (e: 'file', file: File): void }>()

const dragging = ref(false)
const fileName = ref<string | null>(null)
const fileSize = ref<string>('')
const inputRef = ref<HTMLInputElement | null>(null)

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function handleFiles(files: FileList | null): void {
  if (!files || files.length === 0) return
  const file = files[0]
  fileName.value = file.name
  fileSize.value = humanSize(file.size)
  emit('file', file)
}

function onDrop(e: DragEvent): void {
  dragging.value = false
  handleFiles(e.dataTransfer?.files ?? null)
}
</script>

<template>
  <div
    class="relative cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-all duration-300"
    :class="
      dragging
        ? 'border-accent bg-accent/10 scale-[1.01]'
        : 'border-line bg-surface-2 hover:border-accent/50 hover:bg-surface-2'
    "
    @dragover.prevent="dragging = true"
    @dragleave.prevent="dragging = false"
    @drop.prevent="onDrop"
    @click="inputRef?.click()"
  >
    <input
      ref="inputRef"
      type="file"
      accept=".csv,.xlsx,.xls,.pdf,.txt,.md,.json,.jsonl"
      class="hidden"
      @change="handleFiles(($event.target as HTMLInputElement).files)"
    />

    <div class="flex flex-col items-center gap-3">
      <div
        class="flex h-16 w-16 items-center justify-center rounded-2xl bg-accent/15 text-accent"
        :class="dragging ? 'animate-float' : ''"
      >
        <component :is="fileName ? FileText : UploadCloud" class="h-8 w-8" />
      </div>

      <template v-if="!fileName">
        <p class="text-base font-semibold text-fg">Drop your file here or click to browse</p>
        <p class="text-xs text-fg-muted">
          Supports <span class="font-medium text-fg">PDF, Excel (.xlsx), CSV, TXT, Markdown (.md), JSON</span>
        </p>
        <div class="mt-1 flex flex-wrap justify-center gap-1.5 text-[10px]">
          <span class="rounded-md bg-accent/10 px-2 py-0.5 font-medium text-accent">PDF Manuals</span>
          <span class="rounded-md bg-emerald-500/10 px-2 py-0.5 font-medium text-emerald-500">Excel / CSV</span>
          <span class="rounded-md bg-sky-500/10 px-2 py-0.5 font-medium text-sky-500">Web / TXT / MD</span>
          <span class="rounded-md bg-purple-500/10 px-2 py-0.5 font-medium text-purple-500">JSON / JSONL</span>
        </div>
      </template>
      <template v-else>
        <p class="text-base font-semibold text-fg">{{ fileName }}</p>
        <p class="text-sm text-fg-muted">{{ fileSize }} · click to replace</p>
      </template>
    </div>
  </div>
</template>
