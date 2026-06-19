/**
 * Plain-English glossary of the AI terms that show up around the app.
 * Surfaced in the Glossary modal and reused for inline tooltips.
 */
export interface GlossaryTerm {
  term: string
  short: string
  detail: string
}

export const GLOSSARY: GlossaryTerm[] = [
  {
    term: 'Fine-tuning',
    short: 'Teaching an existing AI model your own examples.',
    detail:
      'Instead of training a model from scratch (which needs huge data and money), you take a model that already knows language and show it a few hundred of your own input→output examples so it picks up your specific task or style.',
  },
  {
    term: 'Model',
    short: 'The pre-trained AI brain you start from.',
    detail:
      'A neural network that has already learned language from the internet. You pick one from Hugging Face (a free model library) and fine-tune it on your data.',
  },
  {
    term: 'Dataset',
    short: 'Your training examples, as a CSV.',
    detail:
      'A spreadsheet where each row is one example: an "input" (the question/prompt) and the "output" you want the model to learn to produce.',
  },
  {
    term: 'Token',
    short: 'A chunk of text — roughly ¾ of a word.',
    detail:
      'Models read text in small pieces called tokens. "Hello there" is about 2–3 tokens. Longer text = more tokens = more memory and time.',
  },
  {
    term: 'Epoch',
    short: 'One full pass through your dataset.',
    detail:
      'If you set 3 epochs, the model sees every example 3 times. More epochs can mean more learning, but too many can cause it to memorize instead of generalize (overfitting).',
  },
  {
    term: 'Batch size',
    short: 'How many examples are processed at once.',
    detail:
      'Bigger batches train faster but use more GPU memory. If you run out of memory, lowering the batch size is the first thing to try.',
  },
  {
    term: 'Learning rate',
    short: 'How big each learning step is.',
    detail:
      'Too high and training becomes unstable; too low and it learns very slowly. 0.0002 (2e-4) is a safe default for LoRA fine-tuning.',
  },
  {
    term: 'Loss',
    short: 'How wrong the model currently is.',
    detail:
      'A number that should generally go down as training progresses. Lower loss usually means the model is learning your data well.',
  },
  {
    term: 'LoRA',
    short: 'A cheap way to fine-tune by training tiny add-on layers.',
    detail:
      'Low-Rank Adaptation freezes the big original model and only trains a small set of extra weights. This is far faster and uses far less memory than training the whole model — and the result is a small file.',
  },
  {
    term: '4-bit quantization',
    short: 'Shrinking the model so it fits in less GPU memory.',
    detail:
      'The model’s numbers are stored using 4 bits instead of 16/32, cutting memory use roughly 4–8×. Combined with LoRA (this is called QLoRA), it lets large models train on consumer GPUs.',
  },
  {
    term: 'Causal LM (GPT-style)',
    short: 'Predicts the next word; great for chat & text generation.',
    detail:
      'Decoder-only models like GPT-2 generate text left-to-right. Good for chat, completion and open-ended generation. Detected automatically from the model.',
  },
  {
    term: 'Seq2seq (T5-style)',
    short: 'Reads an input and writes an output; great for translation.',
    detail:
      'Encoder-decoder models like T5 take a full input and produce a transformed output. Good for translation, summarizing and structured tasks. Detected automatically.',
  },
  {
    term: 'VRAM',
    short: 'Your GPU’s memory.',
    detail:
      'The working memory on your graphics card. The model weights, activations and the KV cache all live here during training. If a model needs more VRAM than you have, you get an out-of-memory (OOM) error.',
  },
  {
    term: 'KV cache',
    short: 'Memory the model uses to remember the conversation so far.',
    detail:
      'When generating text, the model caches its past computations (keys & values) to avoid redoing work. It grows with longer text and adds to VRAM usage.',
  },
  {
    term: 'OOM (out of memory)',
    short: 'The model needed more GPU memory than you have.',
    detail:
      'Fix it by choosing a smaller model, lowering the batch size, or reducing the max sequence length.',
  },
  {
    term: 'GPU',
    short: 'The graphics chip that makes training fast.',
    detail:
      'NVIDIA GPUs accelerate the math behind training. This app detects your GPU automatically; without one it falls back to the CPU, which is much slower.',
  },
]
