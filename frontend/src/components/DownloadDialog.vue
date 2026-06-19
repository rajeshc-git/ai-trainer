<script setup lang="ts">
import { computed } from 'vue'
import {
  Dialog,
  DialogPanel,
  DialogTitle,
  TransitionChild,
  TransitionRoot,
} from '@headlessui/vue'
import { Download, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-vue-next'

const props = defineProps<{
  open: boolean
  name: string
  loaded: number
  total: number | null
  status: 'downloading' | 'done' | 'error'
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'cancel'): void
}>()

const percent = computed(() => {
  if (!props.total || props.total <= 0) return null
  return Math.min(100, Math.round((props.loaded / props.total) * 100))
})

function fmt(b: number): string {
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)} MB`
  return `${(b / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

// Block backdrop close while actively downloading so the request isn't orphaned.
function onClose(): void {
  if (props.status === 'downloading') return
  emit('close')
}
</script>

<template>
  <TransitionRoot :show="open" as="template">
    <Dialog class="relative z-50" @close="onClose">
      <TransitionChild
        as="template"
        enter="duration-300 ease-out" enter-from="opacity-0" enter-to="opacity-100"
        leave="duration-200 ease-in" leave-from="opacity-100" leave-to="opacity-0"
      >
        <div class="fixed inset-0 bg-black/60 backdrop-blur-sm" />
      </TransitionChild>

      <div class="fixed inset-0 flex items-center justify-center p-4">
        <TransitionChild
          as="template"
          enter="duration-300 ease-out" enter-from="opacity-0 scale-95" enter-to="opacity-100 scale-100"
          leave="duration-200 ease-in" leave-from="opacity-100 scale-100" leave-to="opacity-0 scale-95"
        >
          <DialogPanel class="glass w-full max-w-md p-6">
            <div class="flex items-start gap-4">
              <div
                class="flex h-11 w-11 shrink-0 items-center justify-center rounded-full"
                :class="
                  status === 'error'
                    ? 'bg-danger/15 text-danger'
                    : status === 'done'
                      ? 'bg-success/15 text-success'
                      : 'bg-accent/15 text-accent'
                "
              >
                <AlertTriangle v-if="status === 'error'" class="h-6 w-6" />
                <CheckCircle2 v-else-if="status === 'done'" class="h-6 w-6" />
                <Download v-else class="h-6 w-6" />
              </div>
              <div class="min-w-0 flex-1">
                <DialogTitle class="text-lg font-bold text-fg">
                  {{ status === 'error' ? 'Download failed' : status === 'done' ? 'Download ready' : 'Downloading model' }}
                </DialogTitle>
                <p class="mt-1 truncate text-sm text-fg-muted">{{ name }}.zip</p>
              </div>
            </div>

            <!-- Progress -->
            <div v-if="status === 'downloading'" class="mt-5 space-y-2">
              <div class="flex items-center justify-between text-sm">
                <span class="flex items-center gap-2 text-fg-muted">
                  <Loader2 class="h-4 w-4 animate-spin text-accent" />
                  {{ percent !== null ? 'Zipping & transferring…' : 'Preparing archive…' }}
                </span>
                <span class="font-mono text-accent">
                  {{ percent !== null ? `${percent}%` : fmt(loaded) }}
                </span>
              </div>
              <div class="h-2.5 w-full overflow-hidden rounded-full bg-surface-2">
                <!-- Determinate when we know the size, indeterminate sweep otherwise. -->
                <div
                  v-if="percent !== null"
                  class="h-full rounded-full bg-accent transition-all duration-300"
                  :style="{ width: `${percent}%` }"
                />
                <div v-else class="dl-indeterminate h-full w-1/3 rounded-full bg-accent" />
              </div>
              <p v-if="percent !== null && total" class="text-right text-[11px] text-fg-subtle">
                {{ fmt(loaded) }} / {{ fmt(total) }}
              </p>
            </div>

            <p v-else-if="status === 'done'" class="mt-4 text-sm text-fg-muted">
              Saved to your browser's downloads — {{ fmt(loaded) }}.
            </p>
            <p v-else class="mt-4 text-sm text-fg-muted">
              Something went wrong while downloading. Please try again.
            </p>

            <div class="mt-6 flex justify-end gap-3">
              <button v-if="status === 'downloading'" class="btn-ghost" @click="emit('cancel')">
                Cancel
              </button>
              <button v-else class="btn-gradient" @click="emit('close')">
                Done
              </button>
            </div>
          </DialogPanel>
        </TransitionChild>
      </div>
    </Dialog>
  </TransitionRoot>
</template>

<style scoped>
@keyframes dl-sweep {
  0% { transform: translateX(-120%); }
  100% { transform: translateX(420%); }
}
.dl-indeterminate {
  animation: dl-sweep 1.1s ease-in-out infinite;
}
</style>
