<script setup lang="ts">
import { computed } from 'vue'
import { Cpu, Zap, WifiOff } from 'lucide-vue-next'
import { useGpuStore } from '@/stores/gpu'

const gpu = useGpuStore()

const state = computed(() => {
  if (gpu.checking) return 'checking'
  if (!gpu.online) return 'offline'
  return gpu.hasGpu ? 'gpu' : 'cpu'
})

// Total VRAM in GB, used to tell the user their card's capacity.
const totalGb = computed(() =>
  gpu.totalMb ? `${(gpu.totalMb / 1024).toFixed(0)} GB` : '',
)

</script>

<template>
  <div
    class="badge"
    :class="{
      'bg-success/15 text-success': state === 'gpu',
      'bg-warn/15 text-warn': state === 'cpu',
      'bg-danger/15 text-danger': state === 'offline',
      'bg-surface-2 text-fg-muted': state === 'checking',
    }"
  >
    <span
      v-if="state === 'gpu'"
      class="h-2 w-2 rounded-full bg-success animate-pulse-ring"
    />
    <template v-if="state === 'gpu'">
      <Zap class="h-3.5 w-3.5" />
      <span class="hidden sm:inline">{{ gpu.gpuName || 'GPU' }}</span>
      <span class="sm:hidden">GPU</span>
      <span v-if="totalGb" class="rounded-full bg-success/20 px-1.5 py-0.5 text-[10px] font-bold">
        {{ totalGb }}
      </span>
    </template>
    <template v-else-if="state === 'cpu'">
      <Cpu class="h-3.5 w-3.5" />
      <span>CPU only</span>
    </template>
    <template v-else-if="state === 'offline'">
      <WifiOff class="h-3.5 w-3.5" />
      <span>Backend offline</span>
    </template>
    <template v-else>
      <span class="h-2 w-2 rounded-full bg-slate-500 animate-pulse" />
      <span>Checking…</span>
    </template>
  </div>
</template>
