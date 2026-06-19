<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  Dialog,
  DialogPanel,
  DialogTitle,
  TransitionChild,
  TransitionRoot,
} from '@headlessui/vue'
import { BookOpen, X, Search } from 'lucide-vue-next'
import { GLOSSARY } from '@/lib/glossary'

defineProps<{ open: boolean }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const query = ref('')
const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return GLOSSARY
  return GLOSSARY.filter(
    (t) =>
      t.term.toLowerCase().includes(q) ||
      t.short.toLowerCase().includes(q) ||
      t.detail.toLowerCase().includes(q),
  )
})
</script>

<template>
  <TransitionRoot :show="open" as="template">
    <Dialog class="relative z-50" @close="emit('close')">
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
          <DialogPanel class="glass flex h-[80vh] w-full max-w-2xl flex-col p-0">
            <!-- Header -->
            <div class="flex items-center justify-between border-b border-line px-5 py-4">
              <div class="flex items-center gap-3">
                <div class="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/15 text-accent">
                  <BookOpen class="h-5 w-5" />
                </div>
                <div>
                  <DialogTitle class="font-bold text-fg">Docs</DialogTitle>
                  <p class="text-xs text-fg-muted">Every AI term in this app, explained simply.</p>
                </div>
              </div>
              <button class="btn-ghost !p-2" @click="emit('close')">
                <X class="h-4 w-4" />
              </button>
            </div>

            <!-- Search -->
            <div class="border-b border-line px-5 py-3">
              <div class="relative">
                <Search class="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-subtle" />
                <input v-model="query" class="input-field pl-9" placeholder="Search terms…" />
              </div>
            </div>

            <!-- Terms -->
            <div class="flex-1 space-y-3 overflow-y-auto p-5">
              <div v-if="!filtered.length" class="py-10 text-center text-sm text-fg-subtle">
                No terms match “{{ query }}”.
              </div>
              <div
                v-for="t in filtered"
                :key="t.term"
                class="rounded-xl border border-line bg-surface-2 p-4"
              >
                <h3 class="font-semibold text-fg">{{ t.term }}</h3>
                <p class="mt-0.5 text-sm font-medium text-accent">{{ t.short }}</p>
                <p class="mt-1 text-sm leading-relaxed text-fg-muted">{{ t.detail }}</p>
              </div>
            </div>
          </DialogPanel>
        </TransitionChild>
      </div>
    </Dialog>
  </TransitionRoot>
</template>
