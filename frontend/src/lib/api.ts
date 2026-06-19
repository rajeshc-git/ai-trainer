/**
 * Centralised API client.
 *
 * Resolves the backend base URL from the Vite env (falling back to localhost),
 * and exposes typed helpers for every endpoint the UI needs. WebSocket URLs
 * are derived from the same base.
 */
import axios from 'axios'

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string) || 'http://localhost:8000'

export const http = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
})

// ── Types ───────────────────────────────────────────────────
export interface HealthResponse {
  status: string
  gpu: boolean
  gpu_name: string | null
  gpu_total_mb: number | null
}

export interface ColumnPreview {
  name: string
  present: boolean
  null_count: number
}

export interface DatasetValidation {
  valid: boolean
  row_count: number
  columns: string[]
  detected: ColumnPreview[]
  sample_rows: Record<string, unknown>[]
  file_size_bytes: number
  error: string | null
  suggestion: string | null
  dataset_id: string | null
}

export interface TrainParams {
  model_name: string
  dataset_id: string
  epochs: number
  learning_rate: number
  batch_size: number
  max_length: number
}

export interface JobStatus {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  percent: number
  current_epoch: number
  total_epochs: number
  current_step: number
  total_steps: number
  loss: number | null
  learning_rate: number | null
  eta_seconds: number | null
  gpu_memory_mb: number | null
  gpu_total_mb: number | null
  gpu_memory_percent: number | null
  model_name: string | null
  started_at: number | null
  finished_at: number | null
  error: string | null
  suggestion: string | null
}

export interface SavedModel {
  job_id: string
  name: string
  base_model: string | null
  created_at: number | null
  dataset_rows: number | null
  final_loss: number | null
  architecture?: string | null
  size_bytes: number
}

export interface HfSearchResult {
  id: string
  downloads: number | null
  likes: number | null
  pipeline_tag: string | null
  gated: boolean
  private: boolean
}

export interface HfModelInfo {
  id: string
  params: number | null
  gated: boolean
  private: boolean
  downloads: number | null
  likes: number | null
  pipeline_tag: string | null
  size_bytes: number | null
}

export interface GgufFile {
  filename: string
  quant: string | null
  size_bytes: number
  created: number
}

// ── Endpoints ───────────────────────────────────────────────
export const api = {
  async health(): Promise<HealthResponse> {
    const { data } = await http.get<HealthResponse>('/health')
    return data
  },

  templateUrl(): string {
    return `${API_BASE}/api/dataset/template`
  },

  async validateDataset(file: File): Promise<DatasetValidation> {
    const form = new FormData()
    form.append('file', file)
    const { data } = await http.post<DatasetValidation>(
      '/api/dataset/validate',
      form,
      { headers: { 'Content-Type': 'multipart/form-data' } },
    )
    return data
  },

  async startTraining(params: TrainParams): Promise<{ job_id: string }> {
    const { data } = await http.post('/api/train', params)
    return data
  },

  async jobStatus(jobId: string): Promise<JobStatus> {
    const { data } = await http.get<JobStatus>(`/api/train/${jobId}/status`)
    return data
  },

  async cancelJob(jobId: string): Promise<void> {
    await http.get(`/api/train/${jobId}/cancel`)
  },

  async deleteJob(jobId: string): Promise<void> {
    await http.delete(`/api/train/${jobId}`)
  },

  async listJobs(): Promise<{ jobs: JobStatus[] }> {
    const { data } = await http.get('/api/train')
    return data
  },

  async listModels(): Promise<{ models: SavedModel[] }> {
    const { data } = await http.get('/api/models/list')
    return data
  },

  downloadModelUrl(jobId: string): string {
    return `${API_BASE}/api/models/${jobId}/download`
  },

  /**
   * Download a model's .zip as a Blob, reporting progress. Pass an
   * AbortSignal to allow cancellation. Timeout is disabled — archives
   * can be large and take a while.
   */
  async downloadModelZip(
    jobId: string,
    onProgress?: (loaded: number, total: number | null) => void,
    signal?: AbortSignal,
  ): Promise<Blob> {
    const { data } = await http.get(`/api/models/${jobId}/download`, {
      responseType: 'blob',
      timeout: 0,
      signal,
      onDownloadProgress: (e) => onProgress?.(e.loaded, e.total ?? null),
    })
    return data as Blob
  },

  // ── Hugging Face / Unsloth model search ──
  async hfSearch(
    q: string,
    opts: { source?: 'all' | 'unsloth'; limit?: number } = {},
  ): Promise<{ models: HfSearchResult[] }> {
    const { data } = await http.get('/api/hf/search', {
      params: { q, source: opts.source ?? 'all', limit: opts.limit ?? 24 },
    })
    return data
  },

  async hfInfo(modelId: string): Promise<HfModelInfo> {
    const { data } = await http.get<HfModelInfo>('/api/hf/info', {
      params: { model_id: modelId },
    })
    return data
  },

  // ── GGUF export ──
  async exportGguf(jobId: string, quant: string): Promise<{ export_id: string; quant: string }> {
    const { data } = await http.post(`/api/models/${jobId}/export/gguf`, { quant })
    return data
  },

  async listGguf(jobId: string): Promise<{ files: GgufFile[] }> {
    const { data } = await http.get(`/api/models/${jobId}/gguf`)
    return data
  },

  ggufDownloadUrl(jobId: string, filename: string): string {
    return `${API_BASE}/api/models/${jobId}/gguf/${encodeURIComponent(filename)}/download`
  },

  async deleteGguf(jobId: string, filename: string): Promise<void> {
    await http.delete(`/api/models/${jobId}/gguf/${encodeURIComponent(filename)}`)
  },

  async deleteModel(jobId: string): Promise<void> {
    await http.delete(`/api/models/${jobId}`)
  },

  async chat(
    jobId: string,
    message: string,
    opts: { max_new_tokens?: number; temperature?: number } = {},
  ): Promise<{ response: string }> {
    const { data } = await http.post(
      `/api/inference/${jobId}/chat`,
      { message, ...opts },
      { timeout: 120000 },
    )
    return data
  },

  async unloadModel(jobId: string): Promise<{ unloaded: boolean; message: string }> {
    const { data } = await http.post(`/api/inference/${jobId}/unload`)
    return data
  },

  async unloadAllModels(): Promise<{ unloaded_count: number; message: string }> {
    const { data } = await http.post('/api/inference/unload-all')
    return data
  },

  wsTrainUrl(jobId: string): string {
    const base = API_BASE.replace(/^http/, 'ws')
    return `${base}/ws/train/${jobId}`
  },
}
