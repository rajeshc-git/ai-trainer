<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  Dialog,
  DialogPanel,
  TransitionChild,
  TransitionRoot,
} from '@headlessui/vue'
import {
  Boxes,
  X,
  Download,
  Trash2,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Copy,
} from 'lucide-vue-next'
import { api } from '@/lib/api'
import type { SavedModel, GgufFile } from '@/lib/api'
import { useToastStore } from '@/stores/toast'
import { useWebSocket } from '@/composables/useWebSocket'

const props = defineProps<{ open: boolean; model: SavedModel | null }>()
const emit = defineEmits<{ (e: 'close'): void }>()
const toast = useToastStore()

interface QuantOption {
  id: string
  label: string
  hint: string
}
const QUANTS: QuantOption[] = [
  { id: 'Q4_K_M', label: 'Q4_K_M', hint: 'Smallest · ~¼ size · great quality' },
  { id: 'Q5_K_M', label: 'Q5_K_M', hint: 'Recommended · best size/quality balance' },
  { id: 'Q8_0', label: 'Q8_0', hint: 'Near-lossless · largest file' },
]

const quant = ref('Q5_K_M')
const exporting = ref(false)
const done = ref(false)
const failed = ref(false)
const percent = ref(0)
const logs = ref<string[]>([])
const ggufFile = ref<string | null>(null)
const existing = ref<GgufFile[]>([])
const logBox = ref<HTMLDivElement | null>(null)

const isSeq2seq = computed(() => props.model?.architecture === 'seq2seq')

const ws = useWebSocket({
  onMessage: (raw) => {
    try {
      const data = JSON.parse(raw)
      if (data.type === 'status' && data.status) {
        if (typeof data.status.percent === 'number') percent.value = data.status.percent
      } else if (data.type === 'done') {
        finish(data.status)
      } else if (data.level && data.message) {
        pushLog(data.message)
      }
    } catch {
      /* ignore malformed frames */
    }
  },
})

function pushLog(line: string): void {
  logs.value.push(line)
  if (logs.value.length > 500) logs.value.shift()
  requestAnimationFrame(() => {
    if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight
  })
}

async function finish(status: string): Promise<void> {
  exporting.value = false
  ws.close()
  if (status === 'completed') {
    percent.value = 100
    done.value = true
    await refreshExisting()
    ggufFile.value = existing.value[0]?.filename ?? null
    toast.success('GGUF export complete! 🎉', 'Download it or run it with Ollama.')
  } else {
    failed.value = true
    toast.error('GGUF export failed', 'See the log for details.')
  }
}

async function refreshExisting(): Promise<void> {
  if (!props.model) return
  try {
    const { files } = await api.listGguf(props.model.job_id)
    existing.value = files
  } catch {
    /* ignore */
  }
}

async function start(): Promise<void> {
  if (!props.model || exporting.value || isSeq2seq.value) return
  exporting.value = true
  done.value = false
  failed.value = false
  percent.value = 0
  logs.value = []
  ggufFile.value = null
  try {
    const { export_id } = await api.exportGguf(props.model.job_id, quant.value)
    ws.connect(api.wsTrainUrl(export_id))
    pushLog(`Export started (${quant.value}). Merging adapter → converting → quantizing…`)
  } catch (e: any) {
    exporting.value = false
    const detail = e?.response?.data?.detail
    toast.error(detail?.error ?? 'Could not start export', detail?.suggestion)
  }
}

function dl(filename: string): void {
  if (!props.model) return
  window.open(api.ggufDownloadUrl(props.model.job_id, filename), '_blank')
}

const deleting = ref<string | null>(null)

async function del(filename: string): Promise<void> {
  if (!props.model || deleting.value) return
  if (!window.confirm(`Delete ${filename}? This can't be undone.`)) return
  deleting.value = filename
  try {
    await api.deleteGguf(props.model.job_id, filename)
    if (ggufFile.value === filename) {
      done.value = false
      ggufFile.value = null
    }
    await refreshExisting()
    toast.success('Deleted', `${filename} removed.`)
  } catch (e: any) {
    const detail = e?.response?.data?.detail
    toast.error('Could not delete', detail?.error ?? 'Please try again.')
  } finally {
    deleting.value = null
  }
}

const ollamaSnippet = computed(() => {
  const f = ggufFile.value ?? existing.value[0]?.filename ?? 'model.gguf'
  const name = (props.model?.name ?? 'my-model').toLowerCase().replace(/[^a-z0-9_-]+/g, '-')
  return `# Save this as 'Modelfile' next to ${f}\nFROM ./${f}\n\n# then build & run:\n# ollama create ${name} -f Modelfile\n# ollama run ${name}`
})

function copySnippet(): void {
  navigator.clipboard?.writeText(ollamaSnippet.value)
  toast.info('Copied', 'Ollama Modelfile snippet copied to clipboard.')
}

function fmtSize(b: number): string {
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(0)} MB`
  return `${(b / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

// Reset + load existing exports whenever a different model is opened.
watch(
  () => [props.open, props.model?.job_id],
  () => {
    if (props.open) {
      exporting.value = false
      done.value = false
      failed.value = false
      percent.value = 0
      logs.value = []
      ggufFile.value = null
      existing.value = []
      void refreshExisting()
    } else {
      ws.close()
    }
  },
)
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
          <DialogPanel class="glass flex max-h-[88vh] w-full max-w-2xl flex-col p-0">
            <!-- Header -->
            <div class="flex items-center justify-between border-b border-line px-5 py-4">
              <div class="flex items-center gap-3">
                <div class="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/15 text-accent">
                  <Boxes class="h-5 w-5" />
                </div>
                <div>
                  <h3 class="font-bold text-fg">Export to GGUF</h3>
                  <p class="text-xs text-fg-muted">{{ model?.name }} → Ollama / llama.cpp</p>
                </div>
              </div>
              <button class="btn-ghost !p-2" @click="emit('close')">
                <X class="h-4 w-4" />
              </button>
            </div>

            <div class="flex-1 space-y-5 overflow-y-auto p-5">
              <!-- seq2seq gate -->
              <div
                v-if="isSeq2seq"
                class="flex items-start gap-3 rounded-xl border border-warn/40 bg-warn/10 p-4 text-sm"
              >
                <AlertTriangle class="mt-0.5 h-5 w-5 shrink-0 text-warn" />
                <div>
                  <p class="font-semibold text-fg">GGUF export isn't available for this model</p>
                  <p class="mt-0.5 text-fg-muted">
                    GGUF (Ollama / llama.cpp) targets causal (GPT-style) models. This is a
                    seq2seq (T5-style) model — use the <strong>.zip</strong> download instead.
                  </p>
                </div>
              </div>

              <template v-else>
                <!-- Quant picker -->
                <div v-if="!exporting && !done">
                  <p class="text-sm font-medium text-fg-muted">Choose a quantization level</p>
                  <div class="mt-3 grid gap-2 sm:grid-cols-3">
                    <button
                      v-for="q in QUANTS"
                      :key="q.id"
                      type="button"
                      class="rounded-xl border p-3 text-left transition-all duration-200"
                      :class="
                        quant === q.id
                          ? 'border-accent bg-accent/10 ring-1 ring-accent'
                          : 'border-line bg-surface-2 hover:border-accent/40'
                      "
                      @click="quant = q.id"
                    >
                      <div class="font-mono text-sm font-bold text-fg">{{ q.label }}</div>
                      <div class="mt-1 text-[11px] leading-snug text-fg-muted">{{ q.hint }}</div>
                    </button>
                  </div>
                  <p class="mt-3 text-xs text-fg-subtle">
                    The adapter is merged into the base model (fp16 — on your GPU when it has room,
                    otherwise CPU), converted to GGUF, then quantized. The live log shows which
                    device was chosen. This can take several minutes.
                  </p>
                  <button class="btn-gradient mt-4" @click="start">
                    <Boxes class="h-4 w-4" /> Start export ({{ quant }})
                  </button>
                </div>

                <!-- Progress -->
                <div v-if="exporting || done || failed" class="space-y-3">
                  <div class="flex items-center justify-between text-sm">
                    <span class="flex items-center gap-2 text-fg-muted">
                      <Loader2 v-if="exporting" class="h-4 w-4 animate-spin text-accent" />
                      <CheckCircle2 v-else-if="done" class="h-4 w-4 text-success" />
                      <AlertTriangle v-else-if="failed" class="h-4 w-4 text-danger" />
                      {{ exporting ? 'Exporting…' : done ? 'Completed' : 'Failed' }}
                    </span>
                    <span class="font-mono text-accent">{{ percent.toFixed(0) }}%</span>
                  </div>
                  <div class="h-2.5 w-full overflow-hidden rounded-full bg-surface-2">
                    <div
                      class="h-full rounded-full bg-accent transition-all duration-500"
                      :style="{ width: `${percent}%` }"
                    />
                  </div>
                  <div
                    ref="logBox"
                    class="h-48 overflow-y-auto rounded-xl border border-line bg-slate-900 p-3 font-mono text-[11px] leading-relaxed text-slate-200"
                  >
                    <div v-if="!logs.length" class="text-slate-500">Waiting for output…</div>
                    <div v-for="(l, i) in logs" :key="i" class="whitespace-pre-wrap break-words">{{ l }}</div>
                  </div>
                </div>

                <!-- Done: download + Ollama usage -->
                <div v-if="done && ggufFile" class="space-y-3">
                  <button class="btn-gradient w-full" @click="dl(ggufFile)">
                    <Download class="h-4 w-4" /> Download {{ ggufFile }}
                  </button>
                  <div class="rounded-xl border border-line bg-surface-2 p-3">
                    <div class="mb-2 flex items-center justify-between">
                      <span class="text-xs font-semibold uppercase tracking-wide text-fg-subtle">Run it with Ollama</span>
                      <button class="btn-ghost !px-2 !py-1 text-xs" @click="copySnippet">
                        <Copy class="h-3.5 w-3.5" /> Copy
                      </button>
                    </div>
                    <pre class="overflow-x-auto whitespace-pre-wrap text-[11px] text-fg-muted">{{ ollamaSnippet }}</pre>
                  </div>
                </div>

                <!-- Existing exports -->
                <div v-if="existing.length && !exporting">
                  <p class="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-subtle">
                    Previously exported
                  </p>
                  <div class="space-y-2">
                    <div
                      v-for="f in existing"
                      :key="f.filename"
                      class="flex items-center justify-between gap-3 rounded-xl border border-line bg-surface-2 p-3 text-sm"
                    >
                      <div class="min-w-0">
                        <div class="truncate font-mono text-fg">{{ f.filename }}</div>
                        <div class="text-[11px] text-fg-subtle">
                          {{ f.quant ?? 'gguf' }} · {{ fmtSize(f.size_bytes) }}
                        </div>
                      </div>
                      <div class="flex shrink-0 items-center gap-1">
                        <button class="btn-ghost !px-3 !py-2" title="Download" @click="dl(f.filename)">
                          <Download class="h-4 w-4" />
                        </button>
                        <button
                          class="btn-ghost !px-3 !py-2 text-danger hover:bg-danger/10"
                          title="Delete"
                          :disabled="deleting === f.filename"
                          @click="del(f.filename)"
                        >
                          <Loader2 v-if="deleting === f.filename" class="h-4 w-4 animate-spin" />
                          <Trash2 v-else class="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </div>
          </DialogPanel>
        </TransitionChild>
      </div>
    </Dialog>
  </TransitionRoot>
</template>
