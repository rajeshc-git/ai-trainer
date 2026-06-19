<script setup lang="ts">
import { ref } from 'vue'
import { HelpCircle } from 'lucide-vue-next'

defineProps<{ text: string }>()
const show = ref(false)
</script>

<template>
  <span class="relative inline-flex">
    <button
      type="button"
      class="text-fg-subtle transition hover:text-accent"
      @mouseenter="show = true"
      @mouseleave="show = false"
      @focus="show = true"
      @blur="show = false"
      aria-label="Help"
    >
      <HelpCircle class="h-4 w-4" />
    </button>
    <Transition
      enter-active-class="transition duration-200"
      enter-from-class="opacity-0 translate-y-1"
      leave-active-class="transition duration-150"
      leave-to-class="opacity-0"
    >
      <span
        v-if="show"
        class="absolute bottom-full left-1/2 z-50 mb-2 w-60 -translate-x-1/2 rounded-xl border border-line bg-surface-2 p-3 text-xs leading-relaxed text-fg shadow-2xl"
      >
        {{ text }}
      </span>
    </Transition>
  </span>
</template>
