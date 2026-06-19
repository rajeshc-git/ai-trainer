<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import {
  Dialog,
  DialogPanel,
  TransitionChild,
  TransitionRoot,
} from '@headlessui/vue'
import { Send, X, Bot, User, Loader2, Sliders, PowerOff } from 'lucide-vue-next'
import { api } from '@/lib/api'
import type { SavedModel } from '@/lib/api'
import { useToastStore } from '@/stores/toast'

const props = defineProps<{ open: boolean; model: SavedModel | null }>()
const emit = defineEmits<{ (e: 'close'): void }>()
const toast = useToastStore()

interface Msg {
  role: 'user' | 'assistant'
  text: string
}
const messages = ref<Msg[]>([])
const input = ref('')
const sending = ref(false)
const scroller = ref<HTMLDivElement | null>(null)
const showSettings = ref(false)
const temperature = ref(0.7)
const maxNewTokens = ref(256)
const unloading = ref(false)

// Reset the conversation whenever a different model is opened.
watch(
  () => props.model?.job_id,
  () => {
    messages.value = []
    input.value = ''
  },
)

async function send(): Promise<void> {
  const text = input.value.trim()
  if (!text || !props.model || sending.value) return
  messages.value.push({ role: 'user', text })
  input.value = ''
  sending.value = true
  await scrollDown()
  try {
    const { response } = await api.chat(props.model.job_id, text, {
      temperature: temperature.value,
      max_new_tokens: maxNewTokens.value,
    })
    messages.value.push({ role: 'assistant', text: response })
  } catch (e: any) {
    const detail = e?.response?.data?.detail
    messages.value.push({
      role: 'assistant',
      text: `⚠️ ${detail?.error ?? 'The model could not generate a response.'} ${detail?.suggestion ?? ''}`,
    })
    toast.error('Inference failed', detail?.suggestion)
  } finally {
    sending.value = false
    await scrollDown()
  }
}

async function scrollDown(): Promise<void> {
  await nextTick()
  if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight
}

// Free the model's GPU/system memory (like ejecting in LM Studio). The next
// message will transparently reload it.
async function unload(): Promise<void> {
  if (!props.model || unloading.value || sending.value) return
  unloading.value = true
  try {
    const { unloaded } = await api.unloadModel(props.model.job_id)
    toast.success(
      unloaded ? 'Model unloaded — VRAM freed' : 'Model was not loaded',
      unloaded ? 'Your next message will reload it.' : undefined,
    )
  } catch (e: any) {
    toast.error('Could not unload the model', e?.response?.data?.detail?.suggestion)
  } finally {
    unloading.value = false
  }
}
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
                  <Bot class="h-5 w-5" />
                </div>
                <div>
                  <h3 class="font-bold text-fg">Chat with {{ model?.name }}</h3>
                  <p class="text-xs text-fg-muted">base: {{ model?.base_model }}</p>
                </div>
              </div>
              <div class="flex items-center gap-1">
                <button
                  class="btn-ghost !p-2"
                  title="Stop & unload — free GPU/system memory (reloads on next message)"
                  :disabled="unloading || sending"
                  @click="unload"
                >
                  <Loader2 v-if="unloading" class="h-4 w-4 animate-spin" />
                  <PowerOff v-else class="h-4 w-4" />
                </button>
                <button class="btn-ghost !p-2" title="Settings" @click="showSettings = !showSettings">
                  <Sliders class="h-4 w-4" />
                </button>
                <button class="btn-ghost !p-2" @click="emit('close')">
                  <X class="h-4 w-4" />
                </button>
              </div>
            </div>

            <!-- Settings -->
            <div v-if="showSettings" class="grid grid-cols-2 gap-4 border-b border-line bg-surface-2 px-5 py-3">
              <div>
                <label class="text-xs text-fg-muted">Temperature: {{ temperature.toFixed(2) }}</label>
                <input v-model.number="temperature" type="range" min="0" max="1.5" step="0.05" class="w-full accent-accent" />
              </div>
              <div>
                <label class="text-xs text-fg-muted">Max new tokens: {{ maxNewTokens }}</label>
                <input v-model.number="maxNewTokens" type="range" min="16" max="1024" step="16" class="w-full accent-accent" />
              </div>
            </div>

            <!-- Messages -->
            <div ref="scroller" class="flex-1 space-y-4 overflow-y-auto p-5">
              <div v-if="!messages.length" class="flex h-full flex-col items-center justify-center text-center text-fg-subtle">
                <Bot class="h-10 w-10 opacity-40" />
                <p class="mt-3 text-sm">Say hello to your fine-tuned model 👋</p>
              </div>

              <div
                v-for="(m, i) in messages"
                :key="i"
                class="flex gap-3"
                :class="m.role === 'user' ? 'flex-row-reverse' : ''"
              >
                <div
                  class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
                  :class="m.role === 'user' ? 'bg-success/15 text-success' : 'bg-accent/15 text-accent'"
                >
                  <component :is="m.role === 'user' ? User : Bot" class="h-4 w-4" />
                </div>
                <div
                  class="max-w-[75%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm"
                  :class="m.role === 'user' ? 'bg-success/15 text-fg' : 'bg-surface-2 text-fg'"
                >
                  {{ m.text }}
                </div>
              </div>

              <div v-if="sending" class="flex gap-3">
                <div class="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-accent">
                  <Bot class="h-4 w-4" />
                </div>
                <div class="rounded-2xl bg-surface-2 px-4 py-3">
                  <Loader2 class="h-4 w-4 animate-spin text-fg-muted" />
                </div>
              </div>
            </div>

            <!-- Input -->
            <div class="border-t border-line p-4">
              <div class="flex items-end gap-2">
                <textarea
                  v-model="input"
                  rows="1"
                  placeholder="Type a message…"
                  class="input-field max-h-32 flex-1 resize-none"
                  @keydown.enter.exact.prevent="send"
                />
                <button class="btn-gradient !px-4" :disabled="sending || !input.trim()" @click="send">
                  <Send class="h-4 w-4" />
                </button>
              </div>
              <p class="mt-1.5 text-[11px] text-fg-subtle">Enter to send · Shift+Enter for a new line</p>
            </div>
          </DialogPanel>
        </TransitionChild>
      </div>
    </Dialog>
  </TransitionRoot>
</template>
