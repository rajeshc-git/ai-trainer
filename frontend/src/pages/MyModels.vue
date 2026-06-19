<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { Boxes, Sparkles, PowerOff, Loader2 } from 'lucide-vue-next'
import ModelCard from '@/components/ModelCard.vue'
import ConfirmDialog from '@/components/ConfirmDialog.vue'
import ChatDialog from '@/components/ChatDialog.vue'
import GgufExportDialog from '@/components/GgufExportDialog.vue'
import DownloadDialog from '@/components/DownloadDialog.vue'
import { api } from '@/lib/api'
import type { SavedModel } from '@/lib/api'
import { useToastStore } from '@/stores/toast'
import { useTrainingStore } from '@/stores/training'

const router = useRouter()
const toast = useToastStore()
const training = useTrainingStore()

const models = ref<SavedModel[]>([])
const loading = ref(true)

const toDelete = ref<SavedModel | null>(null)
const chatModel = ref<SavedModel | null>(null)
const chatOpen = ref(false)
const exportModel = ref<SavedModel | null>(null)
const exportOpen = ref(false)
const unloadingAll = ref(false)

async function load(): Promise<void> {
  loading.value = true
  try {
    const { models: list } = await api.listModels()
    models.value = list
  } catch {
    toast.error('Could not load models', 'Is the backend running?')
  } finally {
    loading.value = false
  }
}

onMounted(load)

// Watch if training completes, and reload models automatically.
watch(
  () => training.isRunning,
  async (running, wasRunning) => {
    if (!running && wasRunning) {
      await load()
    }
  }
)

// In-app download with a progress dialog (instead of opening a blank tab).
const dl = ref({
  open: false,
  name: '',
  loaded: 0,
  total: null as number | null,
  status: 'downloading' as 'downloading' | 'done' | 'error',
})
let dlController: AbortController | null = null

function slug(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'model'
}

async function download(m: SavedModel): Promise<void> {
  dlController = new AbortController()
  dl.value = { open: true, name: slug(m.name), loaded: 0, total: null, status: 'downloading' }
  try {
    const blob = await api.downloadModelZip(
      m.job_id,
      (loaded, total) => {
        dl.value.loaded = loaded
        dl.value.total = total
      },
      dlController.signal,
    )
    // Hand the finished blob to the browser to save at full disk speed.
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${slug(m.name)}.zip`
    document.body.appendChild(a)
    a.click()
    a.remove()
    // Revoke after a tick so the save has started, then drop the blob ref so
    // the browser can reclaim the memory immediately.
    setTimeout(() => URL.revokeObjectURL(url), 1000)
    dl.value.status = 'done'
  } catch (e: any) {
    if (e?.code === 'ERR_CANCELED' || e?.name === 'CanceledError') {
      dl.value.open = false
      return
    }
    dl.value.status = 'error'
    toast.error('Download failed', 'Is the backend running? Please try again.')
  } finally {
    dlController = null
  }
}

function cancelDownload(): void {
  dlController?.abort()
}

function openChat(m: SavedModel): void {
  chatModel.value = m
  chatOpen.value = true
}

function openExport(m: SavedModel): void {
  exportModel.value = m
  exportOpen.value = true
}

// Free the GPU/system memory held by every loaded model at once.
async function unloadAll(): Promise<void> {
  if (unloadingAll.value) return
  unloadingAll.value = true
  try {
    const { unloaded_count } = await api.unloadAllModels()
    if (unloaded_count > 0) {
      toast.success(`Freed memory — unloaded ${unloaded_count} model(s)`, 'They reload on next chat.')
    } else {
      toast.info('No models were loaded', 'Nothing to free right now.')
    }
  } catch {
    toast.error('Could not free model memory')
  } finally {
    unloadingAll.value = false
  }
}

async function confirmDelete(): Promise<void> {
  if (!toDelete.value) return
  const m = toDelete.value
  toDelete.value = null
  try {
    await api.deleteModel(m.job_id)
    models.value = models.value.filter((x) => x.job_id !== m.job_id)
    toast.success('Model deleted', m.name)
  } catch {
    toast.error('Could not delete the model')
  }
}
</script>

<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-extrabold tracking-tight text-fg">My Models</h1>
        <p class="mt-1 text-fg-muted">Your fine-tuned models — chat, download or delete.</p>
      </div>
      <div class="flex items-center gap-2">
        <button
          v-if="models.length"
          class="btn-ghost"
          title="Free GPU/system memory held by any loaded models (they reload on next chat)"
          :disabled="unloadingAll"
          @click="unloadAll"
        >
          <Loader2 v-if="unloadingAll" class="h-4 w-4 animate-spin" />
          <PowerOff v-else class="h-4 w-4" />
          Free memory
        </button>
        <button class="btn-gradient" @click="router.push('/train')">
          <Sparkles class="h-4 w-4" /> New Training
        </button>
      </div>
    </div>

    <!-- Loading skeletons -->
    <div v-if="loading" class="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
      <div v-for="i in 3" :key="i" class="glass p-5">
        <div class="skeleton h-11 w-11 rounded-xl" />
        <div class="skeleton mt-4 h-5 w-32" />
        <div class="skeleton mt-4 h-16 w-full" />
        <div class="skeleton mt-4 h-9 w-full" />
      </div>
    </div>

    <!-- Empty state -->
    <div v-else-if="!models.length" class="glass flex flex-col items-center px-6 py-16 text-center">
      <div class="flex h-16 w-16 items-center justify-center rounded-2xl bg-accent/15 text-accent">
        <Boxes class="h-8 w-8" />
      </div>
      <h2 class="mt-4 text-lg font-bold text-fg">No models yet</h2>
      <p class="mt-1 max-w-sm text-sm text-fg-muted">
        Fine-tune your first model and it will show up here, ready to chat with or download.
      </p>
      <button class="btn-gradient mt-6" @click="router.push('/train')">
        <Sparkles class="h-4 w-4" /> Start Training
      </button>
    </div>

    <!-- Model grid -->
    <div v-else class="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
      <ModelCard
        v-for="m in models"
        :key="m.job_id"
        :model="m"
        @download="download(m)"
        @delete="toDelete = m"
        @chat="openChat(m)"
        @export="openExport(m)"
      />
    </div>

    <ConfirmDialog
      :open="toDelete !== null"
      danger
      title="Delete this model?"
      :message="`This permanently removes “${toDelete?.name}” from disk. This cannot be undone.`"
      confirm-label="Delete"
      @confirm="confirmDelete"
      @cancel="toDelete = null"
    />

    <ChatDialog :open="chatOpen" :model="chatModel" @close="chatOpen = false" />

    <GgufExportDialog :open="exportOpen" :model="exportModel" @close="exportOpen = false" />

    <DownloadDialog
      :open="dl.open"
      :name="dl.name"
      :loaded="dl.loaded"
      :total="dl.total"
      :status="dl.status"
      @cancel="cancelDownload"
      @close="dl.open = false"
    />
  </div>
</template>
