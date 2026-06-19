<script setup lang="ts">
/**
 * A hand-drawn (SVG) graphics card that shows the detected GPU's name and VRAM.
 * Purely decorative — when a CUDA GPU is detected we render an actual little
 * dual-fan card with spinning fans and an NVIDIA-green glow instead of a plain
 * line of text. Falls back to a CPU illustration when there's no GPU.
 *
 * The fans spin only while a training job is actively running and stop on idle.
 */
import { computed } from 'vue'
import { useGpuStore } from '@/stores/gpu'
import { useTrainingStore } from '@/stores/training'

const gpu = useGpuStore()
const training = useTrainingStore()

const totalGb = computed(() =>
  gpu.totalMb ? `${(gpu.totalMb / 1024).toFixed(0)} GB` : null,
)

// Vendor-tinted accent: green for NVIDIA, red-ish for AMD, else our indigo.
const accent = computed(() => {
  const n = (gpu.gpuName || '').toLowerCase()
  if (n.includes('nvidia') || n.includes('geforce') || n.includes('rtx') || n.includes('gtx') || n.includes('tesla') || n.includes('quadro')) {
    return { stroke: '#76b900', glow: '#76b900', label: 'NVIDIA' }
  }
  if (n.includes('amd') || n.includes('radeon')) {
    return { stroke: '#ed1c24', glow: '#ed1c24', label: 'AMD' }
  }
  return { stroke: '#6366f1', glow: '#6366f1', label: 'GPU' }
})
</script>

<template>
  <div class="glass relative overflow-hidden p-6">
    <!-- ambient glow -->
    <div
      class="pointer-events-none absolute -left-8 -top-10 h-40 w-40 rounded-full blur-3xl"
      :style="{ backgroundColor: accent.glow, opacity: gpu.hasGpu ? 0.18 : 0.06 }"
    />

    <div class="flex items-center gap-6">
      <!-- ── Drawn graphics card ─────────────────────────── -->
      <svg
        v-if="gpu.hasGpu"
        viewBox="0 0 200 110"
        class="h-28 w-48 shrink-0 drop-shadow-lg"
        role="img"
        aria-label="Graphics card"
      >
        <defs>
          <linearGradient id="pcb" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#1c2530" />
            <stop offset="1" stop-color="#0f141b" />
          </linearGradient>
          <linearGradient id="shroud" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#2b3543" />
            <stop offset="1" stop-color="#1a2129" />
          </linearGradient>
          <radialGradient id="fanHub" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" :stop-color="accent.stroke" stop-opacity="0.35" />
            <stop offset="0.6" stop-color="#0f141b" />
            <stop offset="1" stop-color="#0f141b" />
          </radialGradient>
        </defs>

        <!-- PCB base -->
        <rect x="6" y="20" width="188" height="74" rx="6" fill="url(#pcb)" stroke="#000" stroke-opacity="0.4" />

        <!-- bracket / IO plate -->
        <rect x="2" y="16" width="8" height="82" rx="2" fill="#3a4654" />
        <rect x="4" y="30" width="3" height="10" rx="1" :fill="accent.stroke" opacity="0.7" />
        <rect x="4" y="46" width="3" height="10" rx="1" fill="#4b5563" />

        <!-- shroud -->
        <rect x="14" y="14" width="178" height="66" rx="8" fill="url(#shroud)" stroke="#000" stroke-opacity="0.35" />

        <!-- accent stripe -->
        <rect x="14" y="14" width="178" height="4" rx="2" :fill="accent.stroke" opacity="0.85" />

        <!-- two fans -->
        <g>
          <circle cx="62" cy="47" r="27" fill="#0c1117" stroke="#3a4654" stroke-width="1.5" />
          <circle cx="138" cy="47" r="27" fill="#0c1117" stroke="#3a4654" stroke-width="1.5" />

          <!-- fan blades — spin only during training -->
          <g :class="['fan', { 'fan--active': training.isRunning }]" style="transform-origin: 62px 47px">
            <g v-for="i in 7" :key="`l${i}`" :transform="`rotate(${i * 51.4} 62 47)`">
              <path d="M62 47 Q70 30 56 26 Q66 38 62 47 Z" fill="#22303d" />
            </g>
            <circle cx="62" cy="47" r="9" fill="url(#fanHub)" :stroke="accent.stroke" stroke-width="1" />
          </g>
          <g :class="['fan fan-rev', { 'fan--active': training.isRunning }]" style="transform-origin: 138px 47px">
            <g v-for="i in 7" :key="`r${i}`" :transform="`rotate(${i * 51.4} 138 47)`">
              <path d="M138 47 Q146 30 132 26 Q142 38 138 47 Z" fill="#22303d" />
            </g>
            <circle cx="138" cy="47" r="9" fill="url(#fanHub)" :stroke="accent.stroke" stroke-width="1" />
          </g>
        </g>

        <!-- PCIe gold connector -->
        <g fill="#caa23a">
          <rect x="40" y="94" width="44" height="9" rx="1" />
          <rect x="96" y="94" width="24" height="9" rx="1" />
        </g>
        <g fill="#0f141b">
          <rect v-for="i in 14" :key="`p${i}`" :x="42 + i * 3" y="94" width="1.2" height="9" />
        </g>

        <!-- power LED -->
        <circle cx="180" cy="26" r="2.4" :fill="training.isRunning ? '#f59e0b' : accent.glow" :class="['led', { 'led--training': training.isRunning }]" />
      </svg>

      <!-- CPU fallback -->
      <svg v-else viewBox="0 0 110 110" class="h-24 w-24 shrink-0 opacity-80" role="img" aria-label="CPU">
        <rect x="28" y="28" width="54" height="54" rx="6" fill="url(#pcb)" stroke="#4b5563" />
        <rect x="40" y="40" width="30" height="30" rx="3" fill="#2b3543" />
        <g stroke="#4b5563" stroke-width="3">
          <line v-for="i in 5" :key="`t${i}`" :x1="34 + i * 8" y1="20" :x2="34 + i * 8" y2="28" />
          <line v-for="i in 5" :key="`b${i}`" :x1="34 + i * 8" y1="82" :x2="34 + i * 8" y2="90" />
          <line v-for="i in 5" :key="`l${i}`" x1="20" :y1="34 + i * 8" x2="28" :y2="34 + i * 8" />
          <line v-for="i in 5" :key="`rr${i}`" x1="82" :y1="34 + i * 8" x2="90" :y2="34 + i * 8" />
        </g>
      </svg>

      <!-- ── Text ────────────────────────────────────────── -->
      <div class="min-w-0">
        <div class="flex items-center gap-2">
          <span
            class="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider"
            :style="{ backgroundColor: `${accent.stroke}22`, color: accent.stroke }"
          >
            <span class="h-1.5 w-1.5 rounded-full" :style="{ backgroundColor: accent.stroke }" />
            {{ gpu.hasGpu ? accent.label : 'CPU' }}
          </span>
          <span v-if="gpu.hasGpu && !training.isRunning" class="text-[11px] font-medium text-success">● ready</span>
          <span v-if="gpu.hasGpu && training.isRunning" class="text-[11px] font-medium text-amber-400 animate-pulse">● training</span>
        </div>

        <h3 class="mt-2 truncate text-lg font-bold text-fg" :title="gpu.gpuName || ''">
          {{ gpu.hasGpu ? (gpu.gpuName || 'Graphics Card') : 'No GPU detected' }}
        </h3>

        <p v-if="gpu.hasGpu && totalGb" class="mt-1 text-sm text-fg-muted">
          <span class="font-mono font-semibold text-fg">{{ totalGb }}</span> VRAM
        </p>
        <p v-else-if="!gpu.hasGpu" class="mt-1 text-sm text-fg-muted">
          Training runs on CPU — much slower.
        </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* ── Fans: idle by default, spin only during training ──────── */
.fan {
  animation: none;
}
.fan-rev {
  animation: none;
}

/* Active state — fans spin */
.fan.fan--active {
  animation: spin 1.6s linear infinite;
}
.fan-rev.fan--active {
  animation: spin 1.6s linear infinite reverse;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* ── LED: gentle blink normally, fast pulse during training ── */
.led {
  animation: blink 2s ease-in-out infinite;
}
.led--training {
  animation: blink-fast 0.8s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}
@keyframes blink-fast {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.2; }
}

@media (prefers-reduced-motion: reduce) {
  .fan, .fan-rev, .led { animation: none; }
}
</style>
