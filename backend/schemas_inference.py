"""Pydantic models for the inference / chat API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body of the simple chat / generate endpoints."""

    message: str = Field(..., description="The user's input text.")
    instruction: Optional[str] = Field(
        None, description="Optional system/instruction prompt."
    )
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """Response from the simple chat / generate endpoints."""

    response: str
    job_id: str


class OpenAIMessage(BaseModel):
    """One message in an OpenAI-style chat request."""

    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    """Subset of the OpenAI chat-completions request body."""

    messages: list[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256

    model_config = {"protected_namespaces": ()}


class OpenAIChoice(BaseModel):
    """One choice in an OpenAI-style chat response."""

    index: int
    message: OpenAIMessage
    finish_reason: str


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat-completions response body."""

    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]

    model_config = {"protected_namespaces": ()}
