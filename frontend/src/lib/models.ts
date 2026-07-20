/**
 * Curated catalog of production-grade Hugging Face models for fine-tuning.
 *
 * Baseline assumption: the user has a capable modern GPU with **≥16 GB VRAM**
 * (e.g. RTX 4080/4090/5080-class). The list therefore leads with real 7B-class
 * models that people actually ship, plus a few lighter options for fast
 * iteration. All entries are openly licensed (no gating / token required).
 *
 * `vramGb` is a conservative estimate of the VRAM needed to *fine-tune* the
 * model with 4-bit + LoRA (QLoRA) at batch size 4 / max length 512, so the UI
 * can tell the user up-front whether it fits their card.
 */
export interface CuratedModel {
  /** Hugging Face model id. */
  id: string
  /** Friendly display name. */
  name: string
  /** One-line, plain-English description of what it's good for. */
  blurb: string
  /** Short task tag shown as a chip. */
  task: string
  /** Architecture family. */
  type: 'causal' | 'seq2seq'
  /** Human-readable parameter count. */
  params: string
  /** Conservative VRAM needed to fine-tune (GB), 4-bit + LoRA. */
  vramGb: number
  /** Marks the recommended default for most users. */
  recommended?: boolean
  /** Indicates if the model requires license acceptance / HF_TOKEN. */
  gated?: boolean
}

export const CURATED_MODELS: CuratedModel[] = [
  {
    id: 'Qwen/Qwen2.5-7B-Instruct',
    name: 'Qwen2.5 7B Instruct',
    blurb: 'Top-tier 7B for chat, reasoning and code. A strong default on a 16 GB card.',
    task: 'Chat / reasoning',
    type: 'causal',
    params: '7B',
    vramGb: 10,
    recommended: true,
  },
  {
    id: 'mistralai/Mistral-7B-v0.1',
    name: 'Mistral 7B',
    blurb: 'The popular 7B base — an excellent starting point for instruction tuning.',
    task: 'Instruction tuning',
    type: 'causal',
    params: '7B',
    vramGb: 10,
  },
  {
    id: 'HuggingFaceH4/zephyr-7b-beta',
    name: 'Zephyr 7B',
    blurb: 'A chat-optimized 7B that already behaves like a helpful assistant.',
    task: 'Chat',
    type: 'causal',
    params: '7B',
    vramGb: 10,
  },
  {
    id: 'microsoft/Phi-3-mini-4k-instruct',
    name: 'Phi-3 Mini',
    blurb: 'Compact 3.8B that punches well above its size. Fast to train, easy to serve.',
    task: 'Chat / general',
    type: 'causal',
    params: '3.8B',
    vramGb: 6,
  },
  {
    id: 'Qwen/Qwen2.5-1.5B-Instruct',
    name: 'Qwen2.5 1.5B',
    blurb: 'Lightweight 1.5B for rapid iteration and quick experiments.',
    task: 'Fast iteration',
    type: 'causal',
    params: '1.5B',
    vramGb: 4,
  },
  {
    id: 'google/flan-t5-large',
    name: 'FLAN-T5 Large',
    blurb: 'Best-in-class encoder-decoder for translation and summarizing.',
    task: 'Translate / summarize',
    type: 'seq2seq',
    params: '780M',
    vramGb: 6,
  },
  {
    id: 'google/flan-t5-xl',
    name: 'FLAN-T5 XL',
    blurb: 'Larger seq2seq for higher-quality structured generation.',
    task: 'Translate / summarize',
    type: 'seq2seq',
    params: '3B',
    vramGb: 11,
  },
  {
    id: 'sshleifer/tiny-gpt2',
    name: 'Tiny GPT-2 (smoke test)',
    blurb: 'Trains in seconds — use it to verify your dataset and pipeline before a real run.',
    task: 'Pipeline check',
    type: 'causal',
    params: '~5M',
    vramGb: 1,
  },
]

export function findCurated(id: string): CuratedModel | undefined {
  return CURATED_MODELS.find((m) => m.id === id)
}

/**
 * Trending, openly-licensed models worth fine-tuning, kept loosely in line with
 * what the community is using. Each is a quick-pick for the "advanced" flow.
 * `gated` ones require accepting a license on Hugging Face + an HF_TOKEN.
 */
export interface TrendingModel {
  id: string
  label: string
  gated?: boolean
}

export const TRENDING_MODELS: TrendingModel[] = [
  { id: 'Qwen/Qwen2.5-7B-Instruct', label: 'Qwen2.5 7B' },
  { id: 'mistralai/Mistral-7B-Instruct-v0.3', label: 'Mistral 7B' },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', label: 'Llama 3.1 8B', gated: true },
  { id: 'google/gemma-2-9b-it', label: 'Gemma 2 9B', gated: true },
  { id: 'microsoft/Phi-3.5-mini-instruct', label: 'Phi-3.5 Mini' },
  { id: '01-ai/Yi-1.5-9B-Chat', label: 'Yi 1.5 9B' },
  { id: 'Qwen/Qwen2.5-14B-Instruct', label: 'Qwen2.5 14B' },
]

/**
 * Best-effort parse of a model's parameter count (in billions) from its id,
 * e.g. "Mistral-7B" → 7, "Phi-3-mini... 3.8b" → 3.8, "pythia-160m" → 0.16,
 * "Mixtral-8x7B" → ~47 (rough, treated as a×b). Returns null if unknown.
 */
export function parseParamCountB(id: string): number | null {
  const s = id.toLowerCase()
  const moe = s.match(/(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*b/)
  if (moe) return parseFloat(moe[1]) * parseFloat(moe[2])
  const b = s.match(/(\d+(?:\.\d+)?)\s*b(?:[^a-z]|$)/)
  if (b) return parseFloat(b[1])
  const m = s.match(/(\d+(?:\.\d+)?)\s*m(?:[^a-z]|$)/)
  if (m) return parseFloat(m[1]) / 1000
  return null
}

/**
 * Estimate the VRAM (GB) needed to fine-tune a model of `paramsB` billion
 * parameters with 4-bit + LoRA (QLoRA), targeting all linear layers.
 * Roughly: 4-bit weights + LoRA/grad/8-bit-optimizer/activations overhead.
 */
export function estimateTrainVramGb(paramsB: number): number {
  return Math.round((1.25 * paramsB + 2.5) * 10) / 10
}
