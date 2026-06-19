# 🧠 AI Trainer

Fine-tune any Hugging Face model on **your own dataset**, using **your local NVIDIA GPU**, with **real-time training progress** and **zero coding required** — then chat with your trained model right in the browser or call it from your own apps via a built-in API.

![screenshot placeholder](docs/screenshot-dashboard.png)
<!-- Add screenshots to ./docs and they will render here -->

---

## ✨ Features

- 🎛 **4-step wizard** — get template → upload CSV → configure → train & monitor
- ⚡ **Automatic NVIDIA GPU detection** — uses your GPU if present, falls back to CPU with a clear warning
- 🪶 **Memory-efficient** — 4-bit quantization (QLoRA-style) + LoRA adapters so even large models fit on a single consumer GPU
- 🔁 **Both model families** — causal LM (GPT-style) and seq2seq (T5-style), auto-detected
- 📈 **Live dashboard** — streaming logs, loss curve, ETA, GPU memory, per-step metrics over WebSocket (auto-reconnects)
- 💬 **Chat with your model** — talk to any trained model from the *My Models* page
- 🔌 **External inference API** — including an **OpenAI-compatible** endpoint so existing tools can use your fine-tuned model
- 🎨 Polished dark-mode UI (Vue 3 + Tailwind, glassmorphism)

---

## 📦 Prerequisites

| Requirement | Notes |
|---|---|
| **Docker Desktop** (or Docker Engine + Compose v2) | https://www.docker.com/products/docker-desktop |
| **NVIDIA GPU** | Optional but strongly recommended. Without one, training runs on CPU and is *very* slow. |
| **NVIDIA GPU drivers** | Latest stable driver for your card. |
| **NVIDIA Container Toolkit** | Lets Docker access your GPU. Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html |

> 💡 **Windows users:** install Docker Desktop with the **WSL 2** backend and a recent NVIDIA driver — GPU passthrough works out of the box, no extra toolkit needed inside WSL.

Verify GPU access works in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If you do, this app will detect it automatically.

---

## 🚀 Setup (3 steps)

```bash
# 1. Clone
git clone <your-repo-url> ai-fine-tuner
cd ai-fine-tuner

# 2. (Optional) create your env file
cp .env.example .env      # edit JWT_SECRET / HF_TOKEN if you like

# 3. Build & run everything
docker compose up --build -d
```

Then open:

- **Frontend (the app):** http://localhost:5173
- **Backend API docs (Swagger):** http://localhost:8000/docs

The first run downloads the CUDA base image and Python ML wheels — it can take several minutes. Subsequent runs are fast.

> The GPU is detected automatically. The navbar shows a **green pulsing badge** with your GPU name when a GPU is found, or an **orange “CPU only”** badge otherwise.

To stop:

```bash
docker compose down
```

---

## 🗂 Dataset format

Your dataset is a **CSV** with two required columns and one optional column:

| Column | Required | Meaning |
|---|---|---|
| `input` | ✅ | The example input / question / prompt |
| `output` | ✅ | The response you want the model to learn |
| `instruction` | optional | A task instruction (for instruction-tuning) |

Each **row** is one training example. Download a ready-to-edit template from **Step 1** of the wizard, or from:

```
GET http://localhost:8000/api/dataset/template
```

### Full example (`my_data.csv`)

```csv
instruction,input,output
Translate the English sentence to French.,Hello, how are you?,Bonjour, comment allez-vous ?
Translate the English sentence to French.,I love programming.,J'adore la programmation.
Answer the question concisely.,What is the capital of Japan?,Tokyo.
```

Internally each row is rendered into this prompt format before training:

```
### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}
```

(The `### Instruction` block is omitted automatically if you don't provide that column.)

---

## ✅ Tested Hugging Face models

Start small — these train quickly and fit comfortably in memory. Great for learning the workflow:

| Model | Type | Why |
|---|---|---|
| `sshleifer/tiny-gpt2` | causal | Tiny — trains in seconds, perfect for a first test (default) |
| `distilgpt2` | causal | Small, fast, decent quality |
| `gpt2` | causal | Classic baseline |
| `EleutherAI/pythia-160m` | causal | Small modern architecture |
| `google/flan-t5-small` | seq2seq | Great for instruction / translation tasks |
| `google/flan-t5-base` | seq2seq | A step up in quality |

Larger models (e.g. `mistralai/Mistral-7B-v0.1`, `meta-llama/Llama-2-7b-hf`) also work thanks to 4-bit + LoRA, but need a GPU with **~6–8 GB+ VRAM** and may require a Hugging Face token (set `HF_TOKEN` in `.env`) for gated models.

---

## 💬 Using your fine-tuned model

### In the browser
Go to **My Models → Chat** on any trained model and start talking to it. Adjust temperature and max tokens from the settings (⚙️) menu.

### From your own code (external API)
Every trained model is exposed at a stable URL keyed by its `job_id`:

**Simple endpoint**

```bash
curl -X POST http://localhost:8000/api/inference/<JOB_ID>/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Translate: Good morning", "max_new_tokens": 64, "temperature": 0.7}'
```

```json
{ "response": "Bonjour", "job_id": "<JOB_ID>" }
```

**OpenAI-compatible endpoint** (drop-in for OpenAI SDKs):

```bash
curl -X POST http://localhost:8000/api/inference/<JOB_ID>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [
          {"role": "system", "content": "Translate the English sentence to French."},
          {"role": "user", "content": "Good morning"}
        ],
        "temperature": 0.7,
        "max_tokens": 64
      }'
```

```python
# Python, using the openai SDK pointed at your local model
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/api/inference/<JOB_ID>/v1",
                api_key="not-needed")
resp = client.chat.completions.create(
    model="<JOB_ID>",
    messages=[{"role": "user", "content": "Good morning"}],
)
print(resp.choices[0].message.content)
```

> You can find a model's `JOB_ID` on the **My Models** page (it's the model folder name) or from `GET /api/models/list`.

### Download the raw weights
Use the **Download** button on a model card (or `GET /api/models/<JOB_ID>/download`) to get a zip of the LoRA adapter + tokenizer. Load it in your own Python:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("path/to/unzipped/model")
tok = AutoTokenizer.from_pretrained("path/to/unzipped/model")
```

---

## 🔧 Configuration reference

| Setting | Range | Default | What it does |
|---|---|---|---|
| Epochs | 1–10 | 3 | How many passes over your data |
| Learning rate | 1e-5 – 1e-3 | 2e-4 | Step size per update |
| Batch size | 1–16 | 4 | Examples processed at once (↑ = more VRAM) |
| Max sequence length | 64–2048 | 512 | Tokens per example (↑ = much more VRAM) |

Models are saved to `./models/<job_id>/`, datasets to `./datasets/`. Both are bind-mounted, so they persist across restarts.

---

## 🩺 Troubleshooting

### "CPU only" badge — GPU not detected
- Confirm `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` works.
- Ensure the **NVIDIA Container Toolkit** is installed and the Docker daemon was restarted afterwards.
- On Windows, use the **WSL 2** backend in Docker Desktop and update your GPU driver.
- Check backend logs: `docker compose logs backend` — it prints GPU status at startup.

### "Not enough GPU memory" (CUDA OOM)
- Lower **Batch Size** (try 1–2).
- Lower **Max Sequence Length** (try 128–256).
- Use a smaller base model from the tested list.
- Close other GPU-hungry apps.

### "Could not load model from Hugging Face"
- Double-check the model name/URL (the wizard auto-extracts the id from a full URL).
- For **gated** models (e.g. Llama), set `HF_TOKEN` in your `.env` and `docker compose up` again.
- Confirm the container has internet access.

### Training is very slow
- You're likely on CPU — check the GPU badge. CPU training is expected to be slow.
- Reduce dataset size / epochs / max length while experimenting.

### Frontend can't reach the backend
- Make sure all three containers are healthy: `docker compose ps`.
- The browser talks to `http://localhost:8000`; if you changed ports, update `VITE_API_BASE` in `.env`.

### Live logs stopped updating
- The WebSocket auto-reconnects. The log header shows ● live / ○ reconnecting. If it stays disconnected, check `docker compose logs backend`.

---

## 🏗 Architecture

```
┌──────────────┐     REST + WebSocket     ┌───────────────────────┐
│  Vue 3 / Vite│ ───────────────────────▶ │  FastAPI backend       │
│  (port 5173) │ ◀─────────────────────── │  (port 8000, GPU)      │
└──────────────┘    live logs / metrics   │  ├─ trainer (TRL+PEFT) │
                                           │  └─ inference engine   │
                                           └──────────┬────────────┘
                                                      │ pub/sub + job state
                                                ┌─────▼─────┐
                                                │  Redis    │
                                                │ (port 6379)│
                                                └───────────┘
```

- **backend** — FastAPI + Transformers/TRL/PEFT/bitsandbytes. Runs training jobs in background threads (concurrent jobs supported) and streams logs to Redis pub/sub.
- **frontend** — Vue 3 + Pinia + Tailwind. Subscribes to the WebSocket for live updates.
- **redis** — job state store + log pub/sub channel.

---

## 📁 Project structure

```
.
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py            # FastAPI app + /health
│   ├── trainer.py         # fine-tuning engine (LoRA + 4-bit)
│   ├── inference.py       # load adapter + generate (chat)
│   ├── schemas.py         # Pydantic models
│   ├── schemas_inference.py
│   ├── utils.py           # GPU detection, dataset helpers, job store
│   └── routers/
│       ├── dataset.py     # template + validation
│       ├── training.py    # start/status/cancel + WebSocket
│       ├── models.py      # list/download/delete
│       └── inference.py   # chat + external (OpenAI-compatible) API
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    └── src/
        ├── main.ts, App.vue, style.css
        ├── router/index.ts
        ├── lib/api.ts
        ├── stores/        # training, gpu, toast (Pinia)
        ├── composables/   # useWebSocket (auto-reconnect)
        ├── pages/         # Dashboard, TrainWizard, MyModels
        └── components/    # DropZone, LossChart, LogTerminal, MetricCard,
                           # ModelCard, StepperNav, GpuBadge, ChatDialog, ...
```

---

## 📜 License

MIT — use it, learn from it, ship it.
