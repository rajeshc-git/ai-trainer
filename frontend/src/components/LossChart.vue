<script setup lang="ts">
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Filler,
} from 'chart.js'
import type { LossPoint } from '@/stores/training'

ChartJS.register(
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Filler,
)

const props = defineProps<{ points: LossPoint[] }>()

const chartData = computed(() => ({
  labels: props.points.map((p) => p.step),
  datasets: [
    {
      label: 'Training loss',
      data: props.points.map((p) => p.loss),
      borderColor: '#6366F1',
      backgroundColor: (ctx: { chart: ChartJS }) => {
        const { ctx: c, chartArea } = ctx.chart
        if (!chartArea) return 'rgba(99,102,241,0.15)'
        const g = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom)
        g.addColorStop(0, 'rgba(99,102,241,0.35)')
        g.addColorStop(1, 'rgba(99,102,241,0)')
        return g
      },
      borderWidth: 2,
      fill: true,
      tension: 0.35,
      pointRadius: 0,
      pointHoverRadius: 4,
    },
  ],
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 250 },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#1E293B',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1,
      titleColor: '#fff',
      bodyColor: '#cbd5e1',
      callbacks: {
        title: (items: { label: string }[]) => `Step ${items[0]?.label}`,
        label: (item: { parsed: { y: number } }) => `Loss: ${item.parsed.y.toFixed(4)}`,
      },
    },
  },
  scales: {
    x: {
      grid: { color: 'rgba(255,255,255,0.05)' },
      ticks: { color: '#64748b', maxTicksLimit: 8 },
      title: { display: true, text: 'Step', color: '#64748b' },
    },
    y: {
      grid: { color: 'rgba(255,255,255,0.05)' },
      ticks: { color: '#64748b' },
      title: { display: true, text: 'Loss', color: '#64748b' },
    },
  },
}
</script>

<template>
  <div class="h-64 w-full">
    <Line v-if="points.length" :data="chartData" :options="chartOptions as any" />
    <div v-else class="flex h-full items-center justify-center text-sm text-fg-subtle">
      Waiting for the first training step…
    </div>
  </div>
</template>
