"""Inference endpoints — chat with a trained model + an external API.

Exposes two surfaces:

* ``POST /api/inference/{job_id}/chat`` — the simple endpoint the built-in
  chat UI uses.
* ``POST /api/inference/{job_id}/generate`` and an OpenAI-compatible
  ``POST /api/inference/{job_id}/v1/chat/completions`` — stable external APIs
  that other apps / scripts can call to use the fine-tuned model.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import inference
from schemas_inference import (
    ChatRequest,
    ChatResponse,
    OpenAIChatRequest,
    OpenAIChatResponse,
)
from utils import now

router = APIRouter(prefix="/api/inference", tags=["inference"])


async def _run_generate(job_id: str, message: str, instruction, max_new_tokens, temperature) -> str:
    """Run the (blocking) generation in a worker thread to free the event loop."""
    try:
        return await asyncio.to_thread(
            inference.generate,
            job_id,
            message,
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except inference.InferenceError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": exc.message, "suggestion": exc.suggestion},
        ) from exc


@router.post("/{job_id}/chat", response_model=ChatResponse)
async def chat(job_id: str, req: ChatRequest) -> JSONResponse:
    """Generate a single chat response from a fine-tuned model."""
    text = await _run_generate(
        job_id, req.message, req.instruction, req.max_new_tokens, req.temperature
    )
    return JSONResponse(status_code=200, content={"response": text, "job_id": job_id})


@router.post("/{job_id}/generate", response_model=ChatResponse)
async def generate(job_id: str, req: ChatRequest) -> JSONResponse:
    """Alias of /chat with a more API-like name for external integrations."""
    text = await _run_generate(
        job_id, req.message, req.instruction, req.max_new_tokens, req.temperature
    )
    return JSONResponse(status_code=200, content={"response": text, "job_id": job_id})


@router.post("/{job_id}/unload")
async def unload_model(job_id: str) -> JSONResponse:
    """Unload one model from memory, freeing its VRAM/RAM (the Stop button).

    Like ejecting a model in LM Studio: the next chat turn will reload it.
    """
    unloaded = await asyncio.to_thread(inference.unload, job_id)
    return JSONResponse(
        status_code=200,
        content={
            "job_id": job_id,
            "unloaded": unloaded,
            "message": "Model unloaded." if unloaded else "Model was not loaded.",
        },
    )


@router.post("/unload-all")
async def unload_all_models() -> JSONResponse:
    """Unload every resident model, freeing all inference memory at once."""
    count = await asyncio.to_thread(inference.unload_all)
    return JSONResponse(
        status_code=200,
        content={"unloaded_count": count, "message": f"Unloaded {count} model(s)."},
    )


@router.get("/loaded")
async def list_loaded_models() -> JSONResponse:
    """List the models currently held in memory (for the UI's load indicator)."""
    return JSONResponse(status_code=200, content={"loaded": inference.loaded_models()})


@router.post("/{job_id}/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_compatible(job_id: str, req: OpenAIChatRequest) -> JSONResponse:
    """OpenAI-compatible chat-completions endpoint.

    Lets you point existing OpenAI-style SDKs/clients at a locally fine-tuned
    model by using ``{job_id}`` as the model. Only the fields needed for simple
    chat are honoured. The last user message is used as the prompt, and any
    leading ``system`` message becomes the instruction.
    """
    system = next((m.content for m in req.messages if m.role == "system"), None)
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No user message provided.",
                "suggestion": "Include at least one message with role 'user'.",
            },
        )

    text = await _run_generate(
        job_id,
        user_msgs[-1],
        system,
        req.max_tokens or 256,
        req.temperature if req.temperature is not None else 0.7,
    )

    return JSONResponse(
        status_code=200,
        content={
            "id": f"chatcmpl-{job_id[:12]}",
            "object": "chat.completion",
            "created": int(now()),
            "model": job_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        },
    )
