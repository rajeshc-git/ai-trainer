<script setup lang="ts">
import {
  Dialog,
  DialogPanel,
  DialogTitle,
  TransitionChild,
  TransitionRoot,
} from '@headlessui/vue'
import { AlertTriangle } from 'lucide-vue-next'

defineProps<{
  open: boolean
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
}>()

const emit = defineEmits<{
  (e: 'confirm'): void
  (e: 'cancel'): void
}>()
</script>

<template>
  <TransitionRoot :show="open" as="template">
    <Dialog class="relative z-50" @close="emit('cancel')">
      <TransitionChild
        as="template"
        enter="duration-300 ease-out"
        enter-from="opacity-0"
        enter-to="opacity-100"
        leave="duration-200 ease-in"
        leave-from="opacity-100"
        leave-to="opacity-0"
      >
        <div class="fixed inset-0 bg-black/60 backdrop-blur-sm" />
      </TransitionChild>

      <div class="fixed inset-0 flex items-center justify-center p-4">
        <TransitionChild
          as="template"
          enter="duration-300 ease-out"
          enter-from="opacity-0 scale-95"
          enter-to="opacity-100 scale-100"
          leave="duration-200 ease-in"
          leave-from="opacity-100 scale-100"
          leave-to="opacity-0 scale-95"
        >
          <DialogPanel class="glass w-full max-w-md p-6">
            <div class="flex items-start gap-4">
              <div
                class="flex h-11 w-11 shrink-0 items-center justify-center rounded-full"
                :class="danger ? 'bg-danger/15 text-danger' : 'bg-accent/15 text-accent'"
              >
                <AlertTriangle class="h-6 w-6" />
              </div>
              <div class="flex-1">
                <DialogTitle class="text-lg font-bold text-fg">{{ title }}</DialogTitle>
                <p class="mt-1 text-sm text-fg-muted">{{ message }}</p>
              </div>
            </div>

            <div class="mt-6 flex justify-end gap-3">
              <button class="btn-ghost" @click="emit('cancel')">
                {{ cancelLabel || 'Cancel' }}
              </button>
              <button
                :class="danger ? 'btn-danger' : 'btn-gradient'"
                @click="emit('confirm')"
              >
                {{ confirmLabel || 'Confirm' }}
              </button>
            </div>
          </DialogPanel>
        </TransitionChild>
      </div>
    </Dialog>
  </TransitionRoot>
</template>
