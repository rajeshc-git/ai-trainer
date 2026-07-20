<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  Download,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  XCircle,
  Sliders,
  Sparkles,
  Gauge,
  Loader2,
  Ban,
  Check,
  AlertTriangle,
  ExternalLink,
  TrendingUp,
  Search,
  Globe,
  FileText,
  Layers,
  ShieldCheck,
} from 'lucide-vue-next'
import StepperNav from '@/components/StepperNav.vue'
import DropZone from '@/components/DropZone.vue'
import HelpTip from '@/components/HelpTip.vue'
import MetricCard from '@/components/MetricCard.vue'
import LossChart from '@/components/LossChart.vue'
import LogTerminal from '@/components/LogTerminal.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
import { api } from '@/lib/api'
import type { DatasetValidation, HfSearchResult, HfModelInfo } from '@/lib/api'
import {
  CURATED_MODELS,
  TRENDING_MODELS,
  findCurated,
  parseParamCountB,
  estimateTrainVramGb,
} from '@/lib/models'
import type { CuratedModel } from '@/lib/models'
import { useTrainingStore } from '@/stores/training'
import { useToastStore } from '@/stores/toast'
import { useGpuStore } from '@/stores/gpu'
import { onMounted, onUnmounted } from 'vue'

const router = useRouter()
const training = useTrainingStore()
const toast = useToastStore()
const gpu = useGpuStore()

const steps = [
  { label: 'Get Dataset', sub: 'Template' },
  { label: 'Upload', sub: 'Your CSV' },
  { label: 'Configure', sub: 'Model + params' },
  { label: 'Train', sub: 'Monitor live' },
]
const current = ref(0)

// ── Step 1 sample table ─────────────────────────────────────
const sampleRows = [
  { input: 'Hello, how are you?', output: 'Bonjour, comment allez-vous ?' },
  { input: 'I love programming.', output: "J'adore la programmation." },
  { input: 'Where is the train station?', output: 'Où est la gare ?' },
]

function downloadTemplate(): void {
  window.open(api.templateUrl(), '_blank')
  toast.success('Template downloaded', 'Fill it in and upload it in step 2.')
}

// ── Step 2 upload + CEF pipeline validation ──────────────────────────────
const validation = ref<DatasetValidation | null>(null)
const validating = ref(false)
const inputTab = ref<'file' | 'url'>('file')
const urlInput = ref('')
const urlExtracting = ref(false)

async function onFile(file: File): Promise<void> {
  validating.value = true
  validation.value = null
  try {
    // Use CEF engine to clean and extract from any document type (PDF, Excel, CSV, TXT, JSON)
    validation.value = await api.extractAndCleanDataset(file)
    if (validation.value.valid) {
      const count = validation.value.clean_pairs_count ?? validation.value.row_count
      toast.success('CEF Data Extraction Complete!', `${count} clean training pairs extracted.`)
    } else {
      toast.error('Extraction failed', validation.value.error ?? undefined)
    }
  } catch {
    toast.error('Validation failed', 'Could not reach the backend CEF engine.')
  } finally {
    validating.value = false
  }
}

async function extractFromUrl(): Promise<void> {
  if (!urlInput.value.trim()) {
    toast.error('Please enter a URL', 'Provide a valid web page URL starting with http:// or https://')
    return
  }
  urlExtracting.value = true
  validation.value = null
  try {
    validation.value = await api.extractUrlDataset(urlInput.value.trim())
    if (validation.value.valid) {
      toast.success('Web Article Extracted!', `${validation.value.clean_pairs_count} pairs generated.`)
    } else {
      toast.error('URL Extraction failed', validation.value.error ?? undefined)
    }
  } catch {
    toast.error('Extraction failed', 'Could not fetch content from that URL.')
  } finally {
    urlExtracting.value = false
  }
}

const datasetValid = computed(() => validation.value?.valid === true)

// ── Step 3 configure ────────────────────────────────────────
// Beginners pick from a curated list; "advanced" reveals a free-text box.
const useCustomModel = ref(false)
const selectedCurated = ref<string>(
  CURATED_MODELS.find((m) => m.recommended)?.id ?? CURATED_MODELS[0].id,
)
const modelInput = ref('')
const showAdvanced = ref(false)
const epochs = ref(3)
const learningRateExp = ref(-3.7) // 10^x ; default ~2e-4
const batchSize = ref(4)
const maxLength = ref(512)

const learningRate = computed(() => Math.pow(10, learningRateExp.value))

const curatedSelected = computed(() => findCurated(selectedCurated.value))

const modelId = computed(() => {
  if (!useCustomModel.value) return selectedCurated.value
  const m = modelInput.value.trim()
  const match = m.match(/huggingface\.co\/([^/\s]+(?:\/[^/\s?#]+)?)/)
  return match ? match[1] : m
})

const gpuCapacityGb = computed(() => (gpu.totalMb ? gpu.totalMb / 1024 : null))

// Adjust a base per-model VRAM need by the user's batch size / max length.
function adjustVram(baseGb: number): number {
  const lenFactor = maxLength.value / 512
  const batchFactor = batchSize.value / 4
  return baseGb + Math.max(0, baseGb * 0.4) * (lenFactor * batchFactor - 1)
}

// VRAM needed for the *currently chosen* model.
const neededVramGb = computed(() => {
  if (!useCustomModel.value && curatedSelected.value) {
    return adjustVram(curatedSelected.value.vramGb)
  }
  // Unknown custom model — fall back to a rough heuristic.
  const lenFactor = maxLength.value / 512
  const batchFactor = batchSize.value / 4
  return 2.2 + 1.6 * lenFactor * batchFactor
})
const vramEstimate = computed(() => `${neededVramGb.value.toFixed(1)} GB`)

// Does a given VRAM requirement fit the user's GPU? null = unknown (no GPU info).
function fitsGpu(needGb: number): boolean | null {
  if (gpuCapacityGb.value == null) return null
  return needGb <= gpuCapacityGb.value * 0.95
}
const estimateFits = computed(() => fitsGpu(neededVramGb.value))

// Per-curated-model fit, used to badge each card in the picker.
function curatedFits(m: CuratedModel): boolean | null {
  return fitsGpu(adjustVram(m.vramGb))
}

// ── Custom (advanced) model: live size + fit detection from the typed id ──
const showHfHelp = ref(false)
const customParamsB = computed(() =>
  useCustomModel.value && modelId.value ? parseParamCountB(modelId.value) : null,
)
const customVramGb = computed(() =>
  customParamsB.value != null ? estimateTrainVramGb(customParamsB.value) : null,
)
const customFits = computed(() =>
  customVramGb.value != null ? fitsGpu(adjustVram(customVramGb.value)) : null,
)
function pickTrending(id: string): void {
  modelInput.value = id
}

// ── Live Hugging Face / Unsloth search ──────────────────────
const searchQuery = ref('')
const searchSource = ref<'all' | 'unsloth'>('all')
const searchResults = ref<HfSearchResult[]>([])
const searching = ref(false)
const selectedInfo = ref<HfModelInfo | null>(null)
const loadingInfo = ref(false)
let searchTimer: ReturnType<typeof setTimeout> | undefined

async function runSearch(): Promise<void> {
  searching.value = true
  try {
    const { models } = await api.hfSearch(searchQuery.value.trim(), {
      source: searchSource.value,
    })
    searchResults.value = models
  } catch {
    toast.error('Search failed', 'Could not reach Hugging Face. Check the backend connection.')
    searchResults.value = []
  } finally {
    searching.value = false
  }
}

function debouncedSearch(): void {
  if (searchTimer) clearTimeout(searchTimer)
  searchTimer = setTimeout(runSearch, 350)
}

// Load trending on first entry to the search tab, and on source switch.
watch(useCustomModel, (v) => {
  if (v && !searchResults.value.length && !searching.value) void runSearch()
})
watch(searchSource, () => void runSearch())

async function selectSearchModel(id: string): Promise<void> {
  modelInput.value = id
  selectedInfo.value = null
  loadingInfo.value = true
  try {
    selectedInfo.value = await api.hfInfo(id)
  } catch {
    selectedInfo.value = null
  } finally {
    loadingInfo.value = false
  }
}

// Typing in the paste box invalidates the (search-result) exact info.
function onModelInput(): void {
  selectedInfo.value = null
}

function resultParamsB(id: string): number | null {
  return parseParamCountB(id)
}
function resultFit(id: string): boolean | null {
  const p = parseParamCountB(id)
  if (p == null) return null
  return fitsGpu(adjustVram(estimateTrainVramGb(p)))
}

function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return `${n}`
}
function fmtParams(b: number): string {
  return b >= 1 ? `${b}B` : `${Math.round(b * 1000)}M`
}

// Exact params from /info when a search result is selected; else estimate by id.
const selectedParamsB = computed<number | null>(() => {
  if (
    selectedInfo.value &&
    selectedInfo.value.id === modelId.value &&
    selectedInfo.value.params
  ) {
    return selectedInfo.value.params / 1e9
  }
  return modelId.value ? parseParamCountB(modelId.value) : null
})
const selectedVramGb = computed<number | null>(() =>
  selectedParamsB.value != null ? estimateTrainVramGb(selectedParamsB.value) : null,
)
const selectedFits = computed<boolean | null>(() =>
  selectedVramGb.value != null ? fitsGpu(adjustVram(selectedVramGb.value)) : null,
)
const downloadSizeGb = computed<number | null>(() => {
  const i = selectedInfo.value
  if (i && i.id === modelId.value && i.size_bytes) return i.size_bytes / 1024 ** 3
  return null
})

// ── Step 4 training + websocket ─────────────────────────────
// The training store now owns the WebSocket — it persists across page
// navigations so status / fan-spin / loss curve stay live even if the user
// switches to the Dashboard mid-training.
const starting = ref(false)
const showCancel = ref(false)

// Register a callback so *this* page can show toasts when training finishes.
// The callback is cleaned up on unmount so it doesn't fire from the Dashboard.
onMounted(() => {
  if (training.isRunning) {
    current.value = 3
  } else {
    current.value = 0
    training.reset(null)
  }
  training.onDone((terminal: string) => {
    if (terminal === 'completed') toast.success('Training complete! 🎉')
    else if (terminal === 'failed') toast.error('Training failed')
    else if (terminal === 'cancelled') toast.warning('Training cancelled')
  })
})
onUnmounted(() => {
  training.clearOnDone()
})

async function startTraining(): Promise<void> {
  if (!validation.value?.dataset_id) {
    toast.error('No dataset', 'Please upload a valid dataset first.')
    return
  }
  starting.value = true
  try {
    const { job_id } = await api.startTraining({
      model_name: modelId.value,
      dataset_id: validation.value.dataset_id,
      epochs: epochs.value,
      learning_rate: learningRate.value,
      batch_size: batchSize.value,
      max_length: maxLength.value,
    })
    training.reset(job_id)
    training.connectWs(api.wsTrainUrl(job_id))
    toast.info('Training started', 'Live logs are streaming below.')
  } catch (e: any) {
    const detail = e?.response?.data?.detail
    toast.error(detail?.error ?? 'Could not start training', detail?.suggestion)
  } finally {
    starting.value = false
  }
}

async function confirmCancel(): Promise<void> {
  showCancel.value = false
  if (!training.jobId) return
  try {
    await api.cancelJob(training.jobId)
    toast.warning('Cancelling…', 'The job will stop at the next step.')
  } catch {
    toast.error('Could not cancel the job')
  }
}

const etaText = computed(() => {
  const s = training.status?.eta_seconds
  if (s == null) return '—'
  if (s < 60) return `${s.toFixed(0)}s`
  if (s < 3600) return `${(s / 60).toFixed(1)}m`
  return `${(s / 3600).toFixed(1)}h`
})

// ── Live VRAM (device-wide: weights + activations + KV cache + everything) ──
const vramUsedMb = computed(() => training.status?.gpu_memory_mb ?? null)
const vramTotalMb = computed(
  () => training.status?.gpu_total_mb ?? gpu.totalMb ?? null,
)
const vramPercent = computed(() => {
  if (training.status?.gpu_memory_percent != null) return training.status.gpu_memory_percent
  if (vramUsedMb.value != null && vramTotalMb.value)
    return Math.min(100, (100 * vramUsedMb.value) / vramTotalMb.value)
  return null
})
const gb = (mb: number) => (mb / 1024).toFixed(1)
const vramText = computed(() => {
  if (vramUsedMb.value == null) return '—'
  if (vramTotalMb.value) return `${gb(vramUsedMb.value)} / ${gb(vramTotalMb.value)} GB`
  return `${gb(vramUsedMb.value)} GB`
})
const vramBarColor = computed(() => {
  const p = vramPercent.value ?? 0
  if (p >= 92) return 'from-danger to-rose-400'
  if (p >= 75) return 'from-warn to-amber-400'
  return 'from-accent to-success'
})

function next(): void {
  if (current.value < steps.length - 1) current.value++
}
function prev(): void {
  if (current.value > 0) {
    current.value--
  } else {
    router.push('/')
  }
}

const canProceed = computed(() => {
  if (current.value === 1) return datasetValid.value
  if (current.value === 2) return modelId.value.length > 0
  return true
})
</script>

<template>
  <div class="space-y-6">
    <div>
      <h1 class="text-3xl font-extrabold tracking-tight text-fg">Training Wizard</h1>
      <p class="mt-1 text-fg-muted">Four simple steps to your own fine-tuned model.</p>
    </div>

    <StepperNav :steps="steps" :current="current" />

    <!-- ───────── STEP 1 — Get Dataset ───────── -->
    <section v-show="current === 0" class="glass p-6 md:p-8">
      <div class="flex items-center gap-2">
        <h2 class="text-xl font-bold text-fg">Step 1 · Get the dataset template</h2>
        <HelpTip text="A dataset is just a list of example questions (input) and the answers you want the model to learn (output)." />
      </div>

      <button class="btn-gradient mt-6" @click="downloadTemplate">
        <Download class="h-5 w-5" /> Download Dataset Template
      </button>

      <div class="mt-6 rounded-xl border border-accent/30 bg-accent/10 p-4 text-sm text-fg">
        <strong class="text-fg">Format:</strong> Your CSV must have 2 columns:
        <code class="rounded bg-surface-2 px-1.5 py-0.5 text-accent">input</code> and
        <code class="rounded bg-surface-2 px-1.5 py-0.5 text-accent">output</code>.
        Each row is one training example. An optional
        <code class="rounded bg-surface-2 px-1.5 py-0.5 text-accent">instruction</code>
        column is supported for instruction-tuning.
      </div>

      <div class="mt-6 overflow-hidden rounded-xl border border-line">
        <table class="w-full text-left text-sm">
          <thead class="bg-surface-2 text-xs uppercase tracking-wide text-fg-muted">
            <tr>
              <th class="px-4 py-2">input</th>
              <th class="px-4 py-2">output</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(r, i) in sampleRows" :key="i" class="border-t border-line">
              <td class="px-4 py-2 text-fg">{{ r.input }}</td>
              <td class="px-4 py-2 text-fg-muted">{{ r.output }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ───────── STEP 2 — CEF Upload & Extraction ───────── -->
    <section v-show="current === 1" class="glass p-6 md:p-8">
      <div class="flex items-center justify-between gap-4">
        <div class="flex items-center gap-2">
          <h2 class="text-xl font-bold text-fg">Step 2 · Ingest & Clean Dataset (CEF Pipeline)</h2>
          <HelpTip text="Upload any PDF manual, Excel sheet, CSV, TXT, JSON, or enter a web URL. The CEF engine will automatically clean, extract, and convert it into fine-tuning pairs." />
        </div>
        <div class="flex items-center gap-1 rounded-xl border border-line bg-surface-2 p-1 text-xs font-semibold">
          <button
            type="button"
            class="flex items-center gap-1.5 rounded-lg px-3 py-1.5 transition-all"
            :class="inputTab === 'file' ? 'bg-accent text-white shadow-sm' : 'text-fg-muted hover:text-fg'"
            @click="inputTab = 'file'"
          >
            <FileText class="h-3.5 w-3.5" />
            File Ingestion
          </button>
          <button
            type="button"
            class="flex items-center gap-1.5 rounded-lg px-3 py-1.5 transition-all"
            :class="inputTab === 'url' ? 'bg-accent text-white shadow-sm' : 'text-fg-muted hover:text-fg'"
            @click="inputTab = 'url'"
          >
            <Globe class="h-3.5 w-3.5" />
            URL Scraper
          </button>
        </div>
      </div>

      <!-- File Ingestion Mode -->
      <div v-if="inputTab === 'file'" class="mt-6">
        <DropZone @file="onFile" />
      </div>

      <!-- URL Scraper Mode -->
      <div v-else class="mt-6 rounded-2xl border border-line bg-surface-2 p-6">
        <div class="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div class="relative flex-1">
            <Globe class="absolute left-3.5 top-3 h-5 w-5 text-fg-muted" />
            <input
              v-model="urlInput"
              type="url"
              placeholder="https://example.com/article-or-documentation"
              class="w-full rounded-xl border border-line bg-surface py-2.5 pl-11 pr-4 text-sm text-fg outline-none focus:border-accent focus:ring-1 focus:ring-accent"
              @keydown.enter="extractFromUrl"
            />
          </div>
          <button
            class="btn-gradient shrink-0"
            :disabled="urlExtracting || !urlInput.trim()"
            @click="extractFromUrl"
          >
            <Loader2 v-if="urlExtracting" class="h-4 w-4 animate-spin" />
            <Sparkles v-else class="h-4 w-4" />
            <span>{{ urlExtracting ? 'Extracting URL…' : 'Extract Pairs' }}</span>
          </button>
        </div>
        <p class="mt-2 text-xs text-fg-muted">
          Scrapes public web content, strips headers/navigation noise, and converts body text into Q&A fine-tuning pairs.
        </p>
      </div>

      <!-- Validating skeleton -->
      <div v-if="validating || urlExtracting" class="mt-6 space-y-3">
        <div class="flex items-center gap-2 text-sm text-accent">
          <Loader2 class="h-4 w-4 animate-spin" />
          <span>CEF Engine running: cleaning, deduplicating, and extracting prompt-response pairs…</span>
        </div>
        <div class="skeleton h-6 w-40" />
        <div class="skeleton h-32 w-full" />
      </div>

      <!-- Validation & CEF Extraction Result -->
      <div v-else-if="validation" class="mt-6 space-y-4">
        <div
          class="flex items-start justify-between gap-3 rounded-xl border p-4"
          :class="datasetValid ? 'border-success/30 bg-success/10' : 'border-danger/30 bg-danger/10'"
        >
          <div class="flex items-start gap-3">
            <component
              :is="datasetValid ? CheckCircle2 : XCircle"
              class="mt-0.5 h-5 w-5 shrink-0"
              :class="datasetValid ? 'text-success' : 'text-danger'"
            />
            <div class="text-sm">
              <p class="font-semibold text-fg">
                {{
                  datasetValid
                    ? `CEF Extraction Complete — ${validation.clean_pairs_count ?? validation.row_count} Clean Pairs Generated`
                    : validation.error
                }}
              </p>
              <p v-if="validation.suggestion" class="mt-0.5 text-fg-muted">
                {{ validation.suggestion }}
              </p>
              <div v-if="datasetValid" class="mt-2 flex flex-wrap items-center gap-3 text-xs">
                <span class="inline-flex items-center gap-1 text-fg-muted">
                  <Layers class="h-3.5 w-3.5 text-accent" />
                  Raw Items Found: <strong class="text-fg">{{ validation.raw_items_found ?? validation.row_count }}</strong>
                </span>
                <span class="inline-flex items-center gap-1 text-fg-muted">
                  <ShieldCheck class="h-3.5 w-3.5 text-success" />
                  Clean Pairs: <strong class="text-fg">{{ validation.clean_pairs_count ?? validation.row_count }}</strong>
                </span>
              </div>
            </div>
          </div>

          <!-- Quality Score Meter -->
          <div v-if="datasetValid && validation.quality_score != null" class="flex flex-col items-end shrink-0">
            <span class="text-[10px] uppercase tracking-wider text-fg-muted">CEF Quality Score</span>
            <div class="mt-0.5 flex items-center gap-1.5">
              <span class="text-lg font-extrabold text-success">{{ validation.quality_score }}%</span>
              <div class="h-2 w-16 overflow-hidden rounded-full bg-surface-2">
                <div
                  class="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400"
                  :style="{ width: `${validation.quality_score}%` }"
                />
              </div>
            </div>
          </div>
        </div>

        <!-- Preview table -->
        <div v-if="validation.sample_rows.length" class="overflow-x-auto rounded-xl border border-line">
          <div class="flex items-center justify-between border-b border-line bg-surface-2 px-4 py-2 text-xs font-semibold text-fg">
            <span>Extracted Fine-Tuning Sample Pairs (Showing First 10)</span>
            <span class="text-fg-muted">CSV output format ready for trainer</span>
          </div>
          <table class="w-full text-left text-sm">
            <thead class="bg-surface-2/60 text-xs uppercase tracking-wide text-fg-muted">
              <tr>
                <th class="px-4 py-2">Instruction / Context</th>
                <th class="px-4 py-2">Input (Prompt)</th>
                <th class="px-4 py-2">Output (Response)</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, i) in validation.sample_rows" :key="i" class="border-t border-line">
                <td class="max-w-[150px] truncate px-4 py-2 text-xs text-fg-subtle">
                  {{ row['instruction'] || 'Standard Q&A' }}
                </td>
                <td class="max-w-xs truncate px-4 py-2 text-fg">
                  {{ row['input'] || row[Object.keys(row)[0]] }}
                </td>
                <td class="max-w-md truncate px-4 py-2 text-fg-muted">
                  {{ row['output'] || row[Object.keys(row)[1]] }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- ───────── STEP 3 — Configure ───────── -->
    <section v-show="current === 2" class="glass p-6 md:p-8">
      <div class="flex items-center gap-2">
        <h2 class="text-xl font-bold text-fg">Step 3 · Choose the model</h2>
        <HelpTip text="Pick a base model to fine-tune. Each card shows what it's good for and whether it fits your GPU. All are openly licensed — no Hugging Face token needed." />
      </div>

      <!-- Source toggle: curated picks vs. the whole hub -->
      <div class="mt-4 inline-flex w-full rounded-xl border border-line bg-surface-2 p-1 sm:w-auto">
        <button
          type="button"
          class="flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-all duration-200 sm:flex-none"
          :class="!useCustomModel ? 'bg-accent text-white shadow-sm shadow-accent/30' : 'text-fg-muted hover:text-fg'"
          @click="useCustomModel = false"
        >
          <Sparkles class="h-4 w-4" />
          Tested models
        </button>
        <button
          type="button"
          class="flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-all duration-200 sm:flex-none"
          :class="useCustomModel ? 'bg-accent text-white shadow-sm shadow-accent/30' : 'text-fg-muted hover:text-fg'"
          @click="useCustomModel = true"
        >
          <Search class="h-4 w-4" />
          Search any model
          <span
            class="hidden rounded-full px-1.5 py-0.5 text-[10px] font-medium sm:inline"
            :class="useCustomModel ? 'bg-white/20 text-white' : 'bg-surface text-fg-subtle'"
          >
            HF + Unsloth
          </span>
        </button>
      </div>

      <p class="mt-3 text-sm text-fg-muted">
        <template v-if="!useCustomModel">
          Tested models, tuned for a 16&nbsp;GB+ GPU. Each shows its task and estimated VRAM to fine-tune.
        </template>
        <template v-else>
          Search the full Hugging Face hub and Unsloth's optimized checkpoints — pick any base model to fine-tune.
        </template>
      </p>

      <!-- Curated picker -->
      <div v-if="!useCustomModel" class="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
        <button
          v-for="m in CURATED_MODELS"
          :key="m.id"
          type="button"
          class="group relative rounded-xl border p-4 text-left transition-all duration-300"
          :class="
            selectedCurated === m.id
              ? 'border-accent bg-accent/10 shadow-lg shadow-accent/20'
              : 'border-line bg-surface-2 hover:border-accent/40 hover:bg-surface'
          "
          @click="selectedCurated = m.id"
        >
          <div class="flex items-start justify-between gap-2">
            <div class="flex items-center gap-2">
              <h3 class="font-bold text-fg">{{ m.name }}</h3>
              <span v-if="m.recommended" class="badge bg-accent/20 text-accent">Recommended</span>
            </div>
            <!-- Fit badge -->
            <span
              class="badge shrink-0"
              :class="
                curatedFits(m) === false
                  ? 'bg-danger/15 text-danger'
                  : 'bg-success/15 text-success'
              "
            >
              <component :is="curatedFits(m) === false ? AlertTriangle : Check" class="h-3.5 w-3.5" />
              {{ curatedFits(m) === false ? 'Needs more VRAM' : curatedFits(m) === null ? '~' + m.vramGb + ' GB' : 'Fits' }}
            </span>
          </div>

          <p class="mt-1.5 text-sm text-fg-muted">{{ m.blurb }}</p>

          <div class="mt-3 flex flex-wrap items-center gap-2 text-xs">
            <span class="badge bg-surface-2 text-fg-muted">{{ m.task }}</span>
            <span class="badge bg-surface-2 text-fg-muted">{{ m.params }} params</span>
            <span class="badge bg-surface-2 text-fg-muted">
              {{ m.type === 'seq2seq' ? 'seq2seq' : 'causal LM' }}
            </span>
            <span class="badge bg-surface-2 text-fg-muted">~{{ m.vramGb }} GB to train</span>
          </div>
        </button>
      </div>

      <!-- Search: any Hugging Face / Unsloth model -->
      <div v-else class="mt-5 space-y-4">
        <!-- Search bar + source toggle -->
        <div class="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div class="relative flex-1">
            <Search class="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-subtle" />
            <input
              v-model="searchQuery"
              class="input-field !pl-9"
              placeholder="Search models — e.g. llama, qwen instruct, gemma…"
              @input="debouncedSearch"
              @keydown.enter.prevent="runSearch"
            />
          </div>
          <div class="inline-flex rounded-xl border border-line bg-surface-2 p-0.5 text-sm">
            <button
              type="button"
              class="rounded-lg px-3 py-1.5 font-medium transition"
              :class="searchSource === 'all' ? 'bg-accent text-white' : 'text-fg-muted hover:text-fg'"
              @click="searchSource = 'all'"
            >
              All HF
            </button>
            <button
              type="button"
              class="rounded-lg px-3 py-1.5 font-medium transition"
              :class="searchSource === 'unsloth' ? 'bg-accent text-white' : 'text-fg-muted hover:text-fg'"
              @click="searchSource = 'unsloth'"
            >
              Unsloth
            </button>
          </div>
        </div>

        <p class="text-xs text-fg-subtle">
          Public &amp; gated models from Hugging Face. Gated ones (🔒, e.g. Llama) need an
          <code class="rounded bg-surface-2 px-1 text-accent">HF_TOKEN</code> in your
          <code class="rounded bg-surface-2 px-1 text-accent">.env</code>. The
          <strong class="text-fg-muted">Unsloth</strong> tab lists Unsloth's optimized checkpoints.
        </p>

        <!-- Results -->
        <div v-if="searching" class="space-y-2">
          <div v-for="i in 4" :key="i" class="skeleton h-14 w-full" />
        </div>
        <div
          v-else-if="searchResults.length"
          class="max-h-[22rem] space-y-2 overflow-y-auto pr-1"
        >
          <button
            v-for="r in searchResults"
            :key="r.id"
            type="button"
            class="flex w-full items-center justify-between gap-3 rounded-xl border p-3 text-left transition-all duration-200"
            :class="
              modelId === r.id
                ? 'border-accent bg-accent/10 ring-1 ring-accent'
                : 'border-line bg-surface-2 hover:border-accent/40 hover:bg-surface'
            "
            @click="selectSearchModel(r.id)"
          >
            <div class="min-w-0">
              <div class="flex items-center gap-1.5">
                <span class="truncate font-mono text-sm font-semibold text-fg">{{ r.id }}</span>
                <span v-if="r.gated" title="Gated — needs HF_TOKEN">🔒</span>
              </div>
              <div class="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-fg-subtle">
                <span v-if="r.downloads != null">↓ {{ fmtCount(r.downloads) }}</span>
                <span v-if="r.likes != null">♥ {{ fmtCount(r.likes) }}</span>
                <span v-if="resultParamsB(r.id) != null">~{{ fmtParams(resultParamsB(r.id)!) }}</span>
              </div>
            </div>
            <span
              class="badge shrink-0"
              :class="
                resultFit(r.id) === false
                  ? 'bg-danger/15 text-danger'
                  : resultFit(r.id) === true
                    ? 'bg-success/15 text-success'
                    : 'bg-surface-2 text-fg-muted'
              "
            >
              <component
                :is="resultFit(r.id) === false ? AlertTriangle : resultFit(r.id) === true ? Check : Gauge"
                class="h-3.5 w-3.5"
              />
              {{ resultFit(r.id) === false ? 'Too big' : resultFit(r.id) === true ? 'Fits' : '?' }}
            </span>
          </button>
        </div>
        <p v-else class="rounded-xl border border-line bg-surface-2 p-4 text-sm text-fg-muted">
          No models found. Try a different search term.
        </p>

        <!-- Or paste an id directly -->
        <div>
          <label class="block text-xs font-medium uppercase tracking-wide text-fg-subtle">
            …or paste a model id / URL
          </label>
          <input
            v-model="modelInput"
            class="input-field mt-2"
            placeholder="e.g. Qwen/Qwen2.5-7B-Instruct  or  https://huggingface.co/..."
            @input="onModelInput"
          />
        </div>

        <!-- Selected model: exact size + fit + download size -->
        <div
          v-if="modelId"
          class="flex flex-wrap items-center gap-2 rounded-xl border p-3 text-sm"
          :class="
            selectedFits === false
              ? 'border-danger/40 bg-danger/10'
              : selectedFits === true
                ? 'border-success/30 bg-success/10'
                : 'border-line bg-surface-2'
          "
        >
          <Loader2 v-if="loadingInfo" class="h-4 w-4 animate-spin text-fg-muted" />
          <component
            v-else
            :is="selectedFits === false ? AlertTriangle : selectedFits === true ? Check : Gauge"
            class="h-4 w-4"
            :class="selectedFits === false ? 'text-danger' : selectedFits === true ? 'text-success' : 'text-fg-muted'"
          />
          <span class="font-mono text-xs text-accent">{{ modelId }}</span>
          <template v-if="selectedParamsB != null">
            <span class="text-fg-muted">
              · ~{{ fmtParams(selectedParamsB) }} · ~{{ selectedVramGb }} GB to fine-tune
              <template v-if="downloadSizeGb != null"> · {{ downloadSizeGb.toFixed(1) }} GB download</template>
            </span>
            <span v-if="selectedFits === true" class="font-semibold text-success">— fits your GPU ✓</span>
            <span v-else-if="selectedFits === false" class="font-semibold text-danger">
              — too big for your {{ gpuCapacityGb?.toFixed(0) }} GB GPU.
            </span>
            <span v-else class="text-fg-muted">— connect a GPU to check fit.</span>
          </template>
          <span v-else-if="!loadingInfo" class="text-fg-muted">
            Size unknown from the name — we'll confirm when training starts.
          </span>
        </div>
      </div>

      <!-- Advanced settings -->
      <button
        class="btn-ghost mt-6"
        @click="showAdvanced = !showAdvanced"
      >
        <Sliders class="h-4 w-4" />
        {{ showAdvanced ? 'Hide' : 'Show' }} Advanced Settings
      </button>

      <Transition
        enter-active-class="transition duration-300"
        enter-from-class="opacity-0 -translate-y-2"
        leave-active-class="transition duration-200"
        leave-to-class="opacity-0"
      >
        <div v-if="showAdvanced" class="mt-6 space-y-6">
          <!-- Epochs -->
          <div>
            <div class="flex items-center justify-between">
              <label class="flex items-center gap-1.5 text-sm font-medium text-fg-muted">
                Epochs <HelpTip text="How many times the model sees your whole dataset. More epochs = more learning, but too many can overfit." />
              </label>
              <span class="font-mono text-sm text-accent">{{ epochs }}</span>
            </div>
            <input v-model.number="epochs" type="range" min="1" max="10" step="1" class="mt-2 w-full accent-accent" />
          </div>

          <!-- Learning rate -->
          <div>
            <div class="flex items-center justify-between">
              <label class="flex items-center gap-1.5 text-sm font-medium text-fg-muted">
                Learning Rate <HelpTip text="How big each learning step is. Too high and training is unstable; too low and it learns slowly. 2e-4 is a safe default." />
              </label>
              <span class="font-mono text-sm text-accent">{{ learningRate.toExponential(1) }}</span>
            </div>
            <input v-model.number="learningRateExp" type="range" min="-5" max="-3" step="0.05" class="mt-2 w-full accent-accent" />
          </div>

          <!-- Batch size -->
          <div>
            <div class="flex items-center justify-between">
              <label class="flex items-center gap-1.5 text-sm font-medium text-fg-muted">
                Batch Size <HelpTip text="How many examples are processed at once. Higher uses more GPU memory but trains faster." />
              </label>
              <span class="font-mono text-sm text-accent">{{ batchSize }}</span>
            </div>
            <input v-model.number="batchSize" type="range" min="1" max="16" step="1" class="mt-2 w-full accent-accent" />
          </div>

          <!-- Max length -->
          <div>
            <div class="flex items-center justify-between">
              <label class="flex items-center gap-1.5 text-sm font-medium text-fg-muted">
                Max Sequence Length <HelpTip text="The maximum number of tokens per example. Longer sequences need much more GPU memory." />
              </label>
              <span class="font-mono text-sm text-accent">{{ maxLength }}</span>
            </div>
            <input v-model.number="maxLength" type="range" min="64" max="2048" step="64" class="mt-2 w-full accent-accent" />
          </div>
        </div>
      </Transition>

      <!-- VRAM estimator -->
      <div
        class="mt-6 flex items-start gap-3 rounded-xl border p-4"
        :class="
          estimateFits === false
            ? 'border-warn/40 bg-warn/10'
            : 'border-accent/30 bg-accent/10'
        "
      >
        <Gauge class="mt-0.5 h-5 w-5" :class="estimateFits === false ? 'text-warn' : 'text-accent'" />
        <div class="text-sm">
          <div>
            <span class="text-fg-muted">Estimated GPU memory needed:</span>
            <span class="ml-1 font-bold text-fg">{{ vramEstimate }}</span>
            <span class="ml-1 text-xs text-fg-subtle">(4-bit quantization + LoRA)</span>
          </div>
          <div v-if="gpuCapacityGb != null" class="mt-1 text-xs">
            <span class="text-fg-muted">Your GPU has</span>
            <span class="font-semibold text-fg"> {{ gpuCapacityGb.toFixed(1) }} GB</span>
            <span v-if="estimateFits" class="ml-1 text-success">— should fit ✓</span>
            <span v-else class="ml-1 text-warn">— may be tight; reduce batch size / max length if you hit an out-of-memory error.</span>
          </div>
          <div v-else class="mt-1 text-xs text-fg-subtle">
            No GPU detected — training will run on CPU (much slower).
          </div>
          <p class="mt-1 text-[11px] text-fg-subtle">
            This is a rough estimate. The real, live VRAM usage is shown while training in step 4.
          </p>
        </div>
      </div>
    </section>

    <!-- ───────── STEP 4 — Train & Monitor ───────── -->
    <section v-show="current === 3" class="glass p-6 md:p-8">
      <div class="flex items-center gap-2">
        <h2 class="text-xl font-bold text-fg">Step 4 · Train &amp; monitor</h2>
        <HelpTip text="Click start, then watch the loss go down. Lower loss generally means the model is learning your data." />
      </div>

      <button
        v-if="!training.jobId"
        class="btn-gradient mt-6 text-base"
        :disabled="starting"
        @click="startTraining"
      >
        <Loader2 v-if="starting" class="h-5 w-5 animate-spin" />
        <Sparkles v-else class="h-5 w-5" />
        {{ starting ? 'Starting…' : 'Start Fine-Tuning' }}
      </button>

      <!-- Live dashboard -->
      <div v-if="training.jobId" class="mt-6 space-y-6">
        <!-- Run header — model + live status pill -->
        <div class="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-line bg-surface-2 px-4 py-3">
          <div class="flex min-w-0 items-center gap-3">
            <div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <Sparkles class="h-5 w-5" />
            </div>
            <div class="min-w-0">
              <div class="truncate font-mono text-sm font-semibold text-fg">{{ modelId }}</div>
              <div class="text-[11px] text-fg-subtle">Fine-tuning run · {{ etaText }} remaining</div>
            </div>
          </div>
          <span
            class="badge"
            :class="{
              'bg-accent/15 text-accent': training.isRunning,
              'bg-success/15 text-success': training.status?.status === 'completed',
              'bg-danger/15 text-danger': training.status?.status === 'failed',
              'bg-surface text-fg-muted': training.status?.status === 'cancelled',
            }"
          >
            <span
              v-if="training.isRunning"
              class="h-1.5 w-1.5 rounded-full bg-accent animate-pulse-ring"
            />
            {{ training.status?.status === 'running' ? 'Training' : training.status?.status ?? 'starting' }}
          </span>
        </div>

        <!-- Progress bar -->
        <div>
          <div class="mb-2 flex items-center justify-between text-sm">
            <span class="text-fg-muted">Overall progress</span>
            <span class="font-mono text-accent">{{ (training.status?.percent || 0).toFixed(1) }}%</span>
          </div>
          <div class="h-3 w-full overflow-hidden rounded-full bg-surface-2">
            <div
              class="h-full rounded-full bg-accent transition-all duration-500"
              :style="{ width: `${training.status?.percent || 0}%` }"
            />
          </div>
        </div>

        <!-- Metric cards -->
        <div class="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <MetricCard label="Loss" :value="training.status?.loss != null ? training.status.loss.toFixed(4) : '—'" accent="indigo" :pulse="training.isRunning" />
          <MetricCard label="Epoch" :value="`${training.status?.current_epoch || 0}/${training.status?.total_epochs || epochs}`" accent="success" />
          <MetricCard label="Step" :value="`${training.status?.current_step || 0}/${training.status?.total_steps || 0}`" accent="warn" />
          <MetricCard label="GPU VRAM" :value="vramText" accent="indigo" />
          <MetricCard label="ETA" :value="etaText" accent="success" />
        </div>

        <!-- VRAM usage bar — helps you see if the model fits your GPU -->
        <div v-if="vramTotalMb" class="glass p-4">
          <div class="mb-2 flex items-center justify-between text-sm">
            <span class="flex items-center gap-1.5 text-fg-muted">
              GPU VRAM usage
              <HelpTip text="Total memory used on your GPU right now — model weights, activations, the KV cache and anything else. If this nears 100%, the model is too big for your card: pick a smaller model or lower the batch size / max length." />
            </span>
            <span class="font-mono text-accent">
              {{ vramText }}<span v-if="vramPercent != null"> · {{ vramPercent.toFixed(0) }}%</span>
            </span>
          </div>
          <div class="h-3 w-full overflow-hidden rounded-full bg-surface-2">
            <div
              class="h-full rounded-full bg-gradient-to-r transition-all duration-500"
              :class="vramBarColor"
              :style="{ width: `${vramPercent ?? 0}%` }"
            />
          </div>
          <p v-if="(vramPercent ?? 0) >= 92" class="mt-2 text-xs text-danger">
            ⚠ VRAM almost full — risk of an out-of-memory error. Consider a smaller model or lower batch size / max length.
          </p>
        </div>

        <!-- Loss chart -->
        <div class="glass p-4">
          <h3 class="mb-3 text-sm font-semibold text-fg-muted">Loss curve</h3>
          <LossChart :points="training.lossCurve" />
        </div>

        <!-- Log terminal -->
        <div>
          <div class="mb-2 flex items-center justify-between">
            <h3 class="text-sm font-semibold text-fg-muted">Live training logs</h3>
            <span class="text-xs" :class="training.wsConnected ? 'text-success' : 'text-warn'">
              {{ training.wsConnected ? '● live' : training.wsReconnecting ? '○ reconnecting…' : '○ disconnected' }}
            </span>
          </div>
          <LogTerminal :lines="training.logs" />
        </div>

        <!-- Error -->
        <div v-if="training.status?.error" class="rounded-xl border border-danger/30 bg-danger/10 p-4 text-sm">
          <p class="font-semibold text-danger">{{ training.status.error }}</p>
          <p v-if="training.status.suggestion" class="mt-1 text-fg-muted">{{ training.status.suggestion }}</p>
        </div>

        <!-- Actions -->
        <div class="flex flex-wrap gap-3">
          <button
            v-if="training.isRunning"
            class="btn-danger"
            @click="showCancel = true"
          >
            <Ban class="h-4 w-4" /> Cancel Training
          </button>
          <button
            v-if="training.status?.status === 'completed'"
            class="btn-gradient"
            @click="router.push('/models')"
          >
            <CheckCircle2 class="h-4 w-4" /> View in My Models
          </button>
        </div>
      </div>
    </section>

    <!-- Wizard navigation -->
    <div class="flex items-center justify-between">
      <button
        class="btn-ghost"
        :disabled="current === 3 && training.isRunning"
        @click="prev"
      >
        <ArrowLeft class="h-4 w-4" /> Back
      </button>
      <button
        v-if="current < steps.length - 1"
        class="btn-gradient"
        :disabled="!canProceed"
        @click="next"
      >
        Continue <ArrowRight class="h-4 w-4" />
      </button>
    </div>

    <ConfirmDialog
      :open="showCancel"
      danger
      title="Cancel this training run?"
      message="Progress so far will be lost and the partially-trained model will not be saved."
      confirm-label="Yes, cancel"
      cancel-label="Keep training"
      @confirm="confirmCancel"
      @cancel="showCancel = false"
    />
  </div>
</template>
