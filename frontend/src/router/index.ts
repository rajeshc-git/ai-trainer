import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'dashboard',
    component: () => import('@/pages/Dashboard.vue'),
    meta: { title: 'Dashboard' },
  },
  {
    path: '/train',
    name: 'train',
    component: () => import('@/pages/TrainWizard.vue'),
    meta: { title: 'New Training' },
  },
  {
    path: '/models',
    name: 'models',
    component: () => import('@/pages/MyModels.vue'),
    meta: { title: 'My Models' },
  },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 }
  },
})

router.afterEach((to) => {
  document.title = `${(to.meta.title as string) ?? 'AI Fine-Tuner'} · AI Fine-Tuner`
})

export default router
