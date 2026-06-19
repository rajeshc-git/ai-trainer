<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { Cpu, LayoutDashboard, Sparkles, Boxes, BookOpen, Sun, Moon } from 'lucide-vue-next'
import GpuBadge from '@/components/GpuBadge.vue'
import ToastContainer from '@/components/ToastContainer.vue'
import GlossaryModal from '@/components/GlossaryModal.vue'
import { useGpuStore } from '@/stores/gpu'
import { useThemeStore } from '@/stores/theme'
import { useTrainingStore } from '@/stores/training'

const gpu = useGpuStore()
const theme = useThemeStore()
const training = useTrainingStore()
const route = useRoute()
const glossaryOpen = ref(false)

onMounted(() => gpu.startPolling())
onUnmounted(() => gpu.stopPolling())

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/train', label: 'New Training', icon: Sparkles },
  { to: '/models', label: 'My Models', icon: Boxes },
]
</script>

<template>
  <div class="min-h-screen">
    <!-- Navbar -->
    <header class="sticky top-0 z-40 border-b border-line bg-surface/80 backdrop-blur-xl">
      <div class="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 md:px-8">
        <RouterLink to="/" class="flex items-center gap-3">
          <div
            class="flex h-10 w-10 items-center justify-center rounded-xl bg-accent shadow-sm"
          >
            <Cpu class="h-5 w-5 text-white" />
          </div>
          <div class="leading-tight">
            <div class="text-lg font-extrabold tracking-tight text-fg">AI Fine-Tuner</div>
            <div class="text-[11px] text-fg-muted">Fine-tune models, zero code</div>
          </div>
        </RouterLink>

        <nav class="hidden items-center gap-1 md:flex">
          <RouterLink
            v-for="item in navItems"
            :key="item.to"
            :to="item.to"
            class="flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium text-fg-muted transition-all duration-300 hover:bg-surface-2 hover:text-fg"
            :class="route.path === item.to ? 'bg-surface-2 text-fg' : ''"
          >
            <component
              :is="item.icon"
              class="h-4 w-4 transition-colors duration-300"
              :class="{ 'text-amber-500 animate-pulse': item.to === '/train' && training.isRunning }"
            />
            <span>{{ item.to === '/train' && training.isRunning ? 'Live Training' : item.label }}</span>
            <span
              v-if="item.to === '/train' && training.isRunning"
              class="relative flex h-2 w-2"
            >
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
            </span>
          </RouterLink>
        </nav>

        <div class="flex items-center gap-2">
          <button
            class="btn-ghost !px-3 !py-2 text-sm"
            :title="theme.isDark ? 'Switch to light theme' : 'Switch to dark theme'"
            @click="theme.toggle()"
          >
            <Sun v-if="theme.isDark" class="h-4 w-4" />
            <Moon v-else class="h-4 w-4" />
          </button>
          <button
            class="btn-ghost !px-3 !py-2 text-sm"
            title="Docs — plain-English explanations of the AI terms"
            @click="glossaryOpen = true"
          >
            <BookOpen class="h-4 w-4" />
            <span class="hidden md:inline">Docs</span>
          </button>
          <GpuBadge />
        </div>
      </div>

      <!-- Mobile/tablet nav -->
      <nav class="flex items-center gap-1 overflow-x-auto px-4 pb-2 md:hidden">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="flex items-center gap-2 whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-medium text-fg-muted"
          :class="route.path === item.to ? 'bg-surface-2 text-fg' : ''"
        >
          <component
            :is="item.icon"
            class="h-4 w-4"
            :class="{ 'text-amber-500 animate-pulse': item.to === '/train' && training.isRunning }"
          />
          <span>{{ item.to === '/train' && training.isRunning ? 'Live Training' : item.label }}</span>
          <span
            v-if="item.to === '/train' && training.isRunning"
            class="relative flex h-2 w-2 shrink-0"
          >
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
          </span>
        </RouterLink>
      </nav>
    </header>

    <!-- Page content -->
    <main class="mx-auto max-w-7xl px-4 py-8 md:px-8">
      <RouterView v-slot="{ Component }">
        <Transition name="fade" mode="out-in">
          <component :is="Component" />
        </Transition>
      </RouterView>
    </main>

    <ToastContainer />
    <GlossaryModal :open="glossaryOpen" @close="glossaryOpen = false" />
  </div>
</template>
