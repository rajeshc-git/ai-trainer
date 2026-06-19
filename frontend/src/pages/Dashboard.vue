<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useStorage } from '@vueuse/core'
import {
  Activity,
  Boxes,
  Clock,
  MemoryStick,
  Sparkles,
  Plus,
  GraduationCap,
  X,
  Upload,
  Settings2,
  Rocket,
  Trash2,
} from 'lucide-vue-next'
import MetricCard from '@/components/MetricCard.vue'
import GpuCard from '@/components/GpuCard.vue'
import { api } from '@/lib/api'
import type { JobStatus, SavedModel } from '@/lib/api'
import { useToastStore } from '@/stores/toast'
import { useGpuStore } from '@/stores/gpu'
import { useTrainingStore } from '@/stores/training'

const router = useRouter()
const toast = useToastStore()
const gpu = useGpuStore()
const training = useTrainingStore()

const deletingJob = ref<JobStatus | null>(null)

function confirmDeleteJob(job: JobStatus) {
  deletingJob.value = job
}

function handleJobClick(job: JobStatus) {
  if (job.status === 'running' || job.status === 'queued') {
    router.push('/train')
  }
}

async function deleteJobConfirmed() {
  if (!deletingJob.value) return
  const jobId = deletingJob.value.job_id
  const modelName = deletingJob.value.model_name || 'Unnamed model'
  try {
    await api.deleteJob(jobId)
    training.jobs = training.jobs.filter((j) => j.job_id !== jobId)
    toast.success('Job deleted', `Successfully removed training history for ${modelName}`)
  } catch (err: any) {
    const errMsg = err.response?.data?.detail?.error || 'Could not delete the job.'
    toast.error('Failed to delete job', errMsg)
  } finally {
    deletingJob.value = null
  }
}

// Show a friendly intro to first-timers; remember if they dismiss it.
const introDismissed = useStorage('ft-intro-dismissed', false)
const introSteps = [
  { icon: Upload, title: 'Bring your examples', text: 'Upload a CSV of input → output pairs — your training data.' },
  { icon: Settings2, title: 'Pick a model', text: 'Choose a small, tested model that fits your GPU. No jargon required.' },
  { icon: Rocket, title: 'Train & chat', text: 'Watch it learn live, then chat with your own custom model.' },
]

const jobs = computed(() => training.jobs)
const models = ref<SavedModel[]>([])
const loading = ref(true)

/** Fetch latest jobs + models from the API and sync training store. */
async function refreshData(): Promise<void> {
  try {
    const [_, m] = await Promise.all([training.refreshJobs(), api.listModels()])
    models.value = m.models
    // Reconcile the training store so isRunning (GPU fan, status badge)
    // reflects reality even if the user left the Train page mid-run.
    training.syncFromJobs(training.jobs)
  } catch {
    // Silently ignore poll failures (toast only on first load).
  }
}

onMounted(async () => {
  try {
    await refreshData()
  } catch {
    toast.error('Could not reach the backend', 'Is the API container running?')
  } finally {
    loading.value = false
  }
})

// Watch if training completes, and reload jobs/models automatically.
watch(
  () => training.isRunning,
  async (running, wasRunning) => {
    if (!running && wasRunning) {
      await refreshData()
    }
  }
)

const totalRuns = computed(() => jobs.value.length)
const modelsSaved = computed(() => models.value.length)

const lastDuration = computed(() => {
  const done = jobs.value.find((j) => j.started_at && j.finished_at)
  if (!done || !done.started_at || !done.finished_at) return '—'
  const secs = Math.max(0, done.finished_at - done.started_at)
  if (secs < 60) return `${secs.toFixed(0)}s`
  if (secs < 3600) return `${(secs / 60).toFixed(1)}m`
  return `${(secs / 3600).toFixed(1)}h`
})

const gpuMem = computed(() => {
  const toGb = (mb: number) => (mb / 1024).toFixed(1)
  const running = jobs.value.find((j) => j.status === 'running' && j.gpu_memory_mb)
  if (running?.gpu_memory_mb) {
    const total = running.gpu_total_mb ?? gpu.totalMb
    return total
      ? `${toGb(running.gpu_memory_mb)} / ${toGb(total)} GB`
      : `${toGb(running.gpu_memory_mb)} GB`
  }
  // No active job — show the card's total capacity so users know what they have.
  return gpu.totalMb ? `0 / ${toGb(gpu.totalMb)} GB` : '—'
})

const statusStyle: Record<string, string> = {
  running: 'bg-accent/15 text-accent',
  completed: 'bg-success/15 text-success',
  failed: 'bg-danger/15 text-danger',
  cancelled: 'bg-slate-500/15 text-fg-muted',
  queued: 'bg-warn/15 text-warn',
}

function fmtDate(ts: number | null): string {
  return ts ? new Date(ts * 1000).toLocaleString() : '—'
}
</script>

<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
      <div>
        <h1 class="text-3xl font-extrabold tracking-tight text-fg">Dashboard</h1>
        <p class="mt-1 text-fg-muted">Monitor your fine-tuning runs and saved models.</p>
      </div>
      <button class="btn-gradient" @click="router.push('/train')">
        <Plus class="h-4 w-4" /> Start New Training
      </button>
    </div>

    <!-- First-timer intro -->
    <div
      v-if="!introDismissed"
      class="glass relative overflow-hidden p-6"
    >
      <button
        class="absolute right-4 top-4 text-fg-muted transition hover:text-fg"
        title="Dismiss"
        @click="introDismissed = true"
      >
        <X class="h-5 w-5" />
      </button>

      <div class="flex items-center gap-3">
        <div class="flex h-11 w-11 items-center justify-center rounded-xl bg-accent text-white">
          <GraduationCap class="h-6 w-6" />
        </div>
        <div>
          <h2 class="text-xl font-bold text-fg">Fine-tune &amp; serve your own models</h2>
          <p class="text-sm text-fg-muted">From dataset to a deployable, chattable model — end to end.</p>
        </div>
      </div>

      <p class="mt-4 max-w-3xl text-sm leading-relaxed text-fg-muted">
        <strong class="text-fg">Fine-tuning</strong> adapts a pre-trained base model to your
        own task using your examples — QLoRA (4-bit + LoRA) under the hood, so 7B-class models
        train comfortably on a single 16&nbsp;GB GPU. Upload a dataset, pick a base model sized
        to your card, monitor loss/VRAM live, then chat with the result or call it over the
        built-in (OpenAI-compatible) inference API.
      </p>

      <div class="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div
          v-for="(s, i) in introSteps"
          :key="i"
          class="rounded-xl border border-line bg-surface-2 p-4"
        >
          <div class="flex items-center gap-2">
            <span class="flex h-7 w-7 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <component :is="s.icon" class="h-4 w-4" />
            </span>
            <span class="text-xs font-bold uppercase tracking-wide text-fg-muted">Step {{ i + 1 }}</span>
          </div>
          <h3 class="mt-2 font-semibold text-fg">{{ s.title }}</h3>
          <p class="mt-1 text-sm text-fg-muted">{{ s.text }}</p>
        </div>
      </div>

      <div class="mt-5 flex flex-wrap items-center gap-3">
        <button class="btn-gradient" @click="router.push('/train')">
          <Sparkles class="h-4 w-4" /> Try it — start your first training
        </button>
        <span class="text-xs text-fg-subtle">
          Tip: open the <strong class="text-fg-muted">Docs</strong> (top bar) any time a word looks unfamiliar.
        </span>
      </div>
    </div>

    <!-- Detected hardware -->
    <GpuCard v-if="!gpu.checking && gpu.online" />

    <!-- Stat cards -->
    <div class="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
      <MetricCard label="Total Training Runs" :value="totalRuns" :icon="Activity" accent="indigo" :loading="loading" />
      <MetricCard label="Models Saved" :value="modelsSaved" :icon="Boxes" accent="success" :loading="loading" />
      <MetricCard label="Last Training Duration" :value="lastDuration" :icon="Clock" accent="warn" :loading="loading" />
      <MetricCard label="GPU Memory Used" :value="gpuMem" :icon="MemoryStick" accent="indigo" :loading="loading" />
    </div>

    <!-- Recent jobs -->
    <div class="glass overflow-hidden">
      <div class="flex items-center justify-between border-b border-line px-6 py-4">
        <h2 class="text-lg font-bold text-fg">Recent Training Jobs</h2>
        <Sparkles class="h-5 w-5 text-accent" />
      </div>

      <div v-if="loading" class="space-y-3 p-6">
        <div v-for="i in 3" :key="i" class="skeleton h-12 w-full" />
      </div>

      <div v-else-if="!jobs.length" class="px-6 py-12 text-center">
        <p class="text-fg-muted">No training runs yet.</p>
        <button class="btn-gradient mt-4" @click="router.push('/train')">
          <Sparkles class="h-4 w-4" /> Fine-tune your first model
        </button>
      </div>

      <div v-else class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead class="bg-surface-2 text-xs uppercase tracking-wide text-fg-subtle">
            <tr class="border-b border-line">
              <th class="px-6 py-3">Model</th>
              <th class="px-6 py-3">Status</th>
              <th class="px-6 py-3">Progress</th>
              <th class="px-6 py-3">Final Loss</th>
              <th class="px-6 py-3">Started</th>
              <th class="px-6 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="job in jobs"
              :key="job.job_id"
              class="group border-b border-line transition hover:bg-surface-2"
              :class="{ 'cursor-pointer hover:bg-accent/5': job.status === 'running' || job.status === 'queued' }"
              @click="handleJobClick(job)"
            >
              <td class="px-6 py-3 font-medium text-fg">{{ job.model_name || '—' }}</td>
              <td class="px-6 py-3">
                <span class="badge" :class="statusStyle[job.status]">
                  <span
                    v-if="job.status === 'running'"
                    class="h-1.5 w-1.5 rounded-full bg-accent animate-pulse"
                  />
                  {{ job.status }}
                </span>
              </td>
              <td class="px-6 py-3">
                <div class="flex items-center gap-2">
                  <div class="h-1.5 w-24 overflow-hidden rounded-full bg-line">
                    <div
                      class="h-full rounded-full bg-accent transition-all"
                      :style="{ width: `${job.percent || 0}%` }"
                    />
                  </div>
                  <span class="text-xs text-fg-muted">{{ (job.percent || 0).toFixed(0) }}%</span>
                </div>
              </td>
              <td class="px-6 py-3 text-fg-muted">
                {{ job.loss != null ? job.loss.toFixed(4) : '—' }}
              </td>
              <td class="px-6 py-3 text-fg-muted">{{ fmtDate(job.started_at) }}</td>
              <td class="px-6 py-3 text-right">
                <button
                  v-if="job.status === 'running' || job.status === 'queued'"
                  class="text-accent hover:bg-accent/10 p-1.5 rounded-lg transition-all duration-200"
                  title="Monitor active training"
                  @click.stop="router.push('/train')"
                >
                  <Activity class="h-4 w-4 animate-pulse" />
                </button>
                <button
                  v-else
                  class="text-fg-subtle hover:text-danger p-1.5 rounded-lg hover:bg-danger/10 transition-all duration-200 opacity-0 group-hover:opacity-100 focus:opacity-100 animate-fade-in"
                  title="Delete job history"
                  @click.stop="confirmDeleteJob(job)"
                >
                  <Trash2 class="h-4 w-4" />
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Delete Confirmation Modal (Teleported to body to keep single root) -->
    <Teleport to="body">
      <Transition name="fade">
        <div
          v-if="deletingJob"
          class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
          @click.self="deletingJob = null"
        >
          <div class="glass w-full max-w-md overflow-hidden p-6 shadow-2xl">
            <div class="flex items-center gap-3 text-danger mb-4">
              <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-danger/15 text-danger">
                <Trash2 class="h-5 w-5" />
              </div>
              <h3 class="text-lg font-bold text-fg">Delete Job History?</h3>
            </div>
            <p class="text-sm text-fg-muted leading-relaxed">
              Are you sure you want to remove the history for <strong class="text-fg">{{ deletingJob.model_name || 'Unnamed model' }}</strong>? 
              This will only delete the training history and logs from the dashboard. Your fine-tuned model files will <strong class="text-success">not</strong> be deleted.
            </p>
            <div class="mt-6 flex justify-end gap-3">
              <button
                class="px-4 py-2 text-sm font-semibold rounded-lg border border-line bg-surface-2 text-fg hover:bg-surface-3 transition"
                @click="deletingJob = null"
              >
                Cancel
              </button>
              <button
                class="px-4 py-2 text-sm font-semibold rounded-lg bg-danger text-white hover:bg-danger/90 transition flex items-center gap-2"
                @click="deleteJobConfirmed"
              >
                Delete Job
              </button>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>
