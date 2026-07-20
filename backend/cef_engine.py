"""CEF (Clean, Extract, Feed) Engine for AI Model Fine-Tuner.

Intelligent ingestion layer that converts raw, unstructured, or heterogeneous files
(PDF, Excel, TXT, Markdown, JSON, Web URLs) into standardized instruction-response
training pairs without changing the underlying training pipeline.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────
# Core Data Models / Types
# ─────────────────────────────────────────────────────────────
class ExtractedPair:
    """One extracted instruction-tuning pair."""

    def __init__(self, input_text: str, output_text: str, instruction: str = ""):
        self.instruction = instruction.strip()
        self.input = input_text.strip()
        self.output = output_text.strip()

    def to_dict(self) -> dict[str, str]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def is_valid(self) -> bool:
        return bool(self.input and self.output)


# ─────────────────────────────────────────────────────────────
# Text Cleaning & Normalization Helpers
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip noise, excessive whitespace, control characters, and HTML artifacts."""
    if not text:
        return ""
    # Strip HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # Collapse horizontal spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def deduplicate_pairs(pairs: list[ExtractedPair]) -> list[ExtractedPair]:
    """Remove exact duplicate input/output pairs while preserving order."""
    seen = set()
    unique = []
    for pair in pairs:
        key = (pair.input.lower(), pair.output.lower())
        if key not in seen and pair.is_valid():
            seen.add(key)
            unique.append(pair)
    return unique


def calculate_quality_score(pairs: list[ExtractedPair], total_raw_items: int) -> float:
    """Compute a quality score between 0.0 and 100.0%."""
    if not pairs:
        return 0.0
    valid_count = sum(1 for p in pairs if len(p.input) >= 3 and len(p.output) >= 3)
    ratio = valid_count / max(len(pairs), 1)
    # Penalize if many items were dropped during cleaning
    retention = len(pairs) / max(total_raw_items, 1) if total_raw_items > 0 else 1.0
    score = round((ratio * 0.7 + min(retention, 1.0) * 0.3) * 100.0, 1)
    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────
# Tabular & Excel Smart Column Autofit
# ─────────────────────────────────────────────────────────────
PROMPT_COLUMN_CANDIDATES = [
    "input", "prompt", "question", "query", "user", "user_message",
    "context", "text", "source", "src", "instruction_input"
]

OUTPUT_COLUMN_CANDIDATES = [
    "output", "completion", "answer", "response", "reply", "target",
    "assistant", "assistant_message", "result", "ground_truth"
]

INSTRUCTION_COLUMN_CANDIDATES = [
    "instruction", "system", "task", "system_prompt", "desc", "description"
]


def extract_from_dataframe(df: pd.DataFrame) -> list[ExtractedPair]:
    """Auto-detect columns from any DataFrame and extract fine-tuning pairs."""
    cols_lower = {str(c).strip().lower(): str(c) for c in df.columns}

    # Find best match for input
    input_col = None
    for cand in PROMPT_COLUMN_CANDIDATES:
        if cand in cols_lower:
            input_col = cols_lower[cand]
            break

    # Find best match for output
    output_col = None
    for cand in OUTPUT_COLUMN_CANDIDATES:
        if cand in cols_lower:
            output_col = cols_lower[cand]
            break

    # Find optional instruction col
    inst_col = None
    for cand in INSTRUCTION_COLUMN_CANDIDATES:
        if cand in cols_lower:
            inst_col = cols_lower[cand]
            break

    # If explicit input/output columns aren't found, pick first two non-empty text columns
    if not input_col or not output_col:
        text_cols = [c for c in df.columns if df[c].dtype == object or df[c].dtype == str]
        if len(text_cols) >= 2:
            input_col = input_col or text_cols[0]
            output_col = output_col or text_cols[1]
        elif len(text_cols) == 1:
            input_col = input_col or text_cols[0]

    pairs: list[ExtractedPair] = []
    if input_col and output_col:
        for _, row in df.iterrows():
            inp = clean_text(str(row.get(input_col) or ""))
            out = clean_text(str(row.get(output_col) or ""))
            inst = clean_text(str(row.get(inst_col) or "")) if inst_col else ""
            if inp and out and inp.lower() != "nan" and out.lower() != "nan":
                pairs.append(ExtractedPair(inp, out, inst))
    return pairs


# ─────────────────────────────────────────────────────────────
# Unstructured Text & Document Chunking Parser
# ─────────────────────────────────────────────────────────────
def extract_from_raw_text(raw_text: str, default_instruction: str = "Answer the prompt accurately based on context.") -> list[ExtractedPair]:
    """Parse unstructured text/markdown into multiple structured instruction-response pairs."""
    cleaned = clean_text(raw_text)
    if not cleaned:
        return []

    pairs: list[ExtractedPair] = []

    # 1. Try Q&A pattern extraction (all Q: / A: blocks)
    qa_blocks = re.findall(
        r"(?:Q|Question|Prompt|Input):\s*(.*?)\n+(?:A|Answer|Response|Output):\s*(.*?)(?=\n+(?:Q|Question|Prompt|Input):|\Z)",
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if qa_blocks and len(qa_blocks) >= 2:
        for q, a in qa_blocks:
            q_clean, a_clean = clean_text(q), clean_text(a)
            if q_clean and a_clean:
                pairs.append(ExtractedPair(q_clean, a_clean, default_instruction))
        if len(pairs) >= 2:
            return pairs

    # 2. Split into blocks/sections by headings or blank lines
    # Split on Markdown headers (#, ##, ###), HTML tags, or lines that look like headings
    raw_sections = re.split(r"\n+(?=#+\s+|\b[A-Z0-9\.\-\_\s]{3,60}\b(?:\:\n|\n))", cleaned)
    if len(raw_sections) >= 2:
        for sec in raw_sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue
            lines = [l.strip() for l in sec_clean.split("\n") if l.strip()]
            if len(lines) >= 2:
                heading = lines[0].strip("# ").strip()
                body = " ".join(lines[1:])
                heading_clean, body_clean = clean_text(heading), clean_text(body)
                if len(heading_clean) >= 3 and len(body_clean) >= 30:
                    pairs.append(
                        ExtractedPair(
                            input_text=f"Explain or summarize: {heading_clean}",
                            output_text=body_clean,
                            instruction=default_instruction,
                        )
                    )

    # 3. If heading split produced 0 or 1 pair, perform robust paragraph & sentence chunking
    if len(pairs) < 2:
        pairs.clear()
        # Normalize newlines and split by paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if len(p.strip()) >= 30]
        if len(paragraphs) <= 1:
            # Split long text by sentences into ~350 character chunks
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) >= 15]
            paragraphs = []
            buf = []
            curr_len = 0
            for sent in sentences:
                buf.append(sent)
                curr_len += len(sent)
                if curr_len >= 300:
                    paragraphs.append(" ".join(buf))
                    buf = []
                    curr_len = 0
            if buf:
                paragraphs.append(" ".join(buf))

        # Create multiple instruction-response pairs from paragraph chunks
        for i, para in enumerate(paragraphs):
            para_clean = clean_text(para)
            if len(para_clean) < 30:
                continue
            # Extract first sentence or summary fragment for prompt input
            first_sentence = para_clean.split(".")[0].strip()
            if len(first_sentence) >= 10 and len(first_sentence) <= 120:
                prompt_input = f"What details are provided about: {first_sentence}?"
            else:
                prompt_input = f"Summarize the key information in section {i + 1}."

            pairs.append(
                ExtractedPair(
                    input_text=prompt_input,
                    output_text=para_clean,
                    instruction=default_instruction,
                )
            )

    return pairs


# ─────────────────────────────────────────────────────────────
# PDF Extractor
# ─────────────────────────────────────────────────────────────
def extract_from_pdf_bytes(pdf_bytes: bytes) -> list[ExtractedPair]:
    """Extract text from PDF file bytes and parse into training pairs."""
    text_content = ""
    # Try pdfplumber first
    try:
        # pyrefly: ignore [missing-import]
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
            text_content = "\n\n".join(pages_text)
    except Exception:
        # Fallback to pypdf / PyPDF2 if pdfplumber fails or is not available
        try:
            # pyrefly: ignore [missing-import]
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
            text_content = "\n\n".join(pages_text)
        except Exception as exc:
            raise RuntimeError(f"Could not parse PDF document: {exc}") from exc

    return extract_from_raw_text(text_content, default_instruction="Answer based on the PDF document content.")


# ─────────────────────────────────────────────────────────────
# Web Page URL Extractor
# ─────────────────────────────────────────────────────────────
async def extract_from_url(url: str) -> list[ExtractedPair]:
    """Fetch URL content, clean HTML body, and extract training pairs."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "AI-FineTuner-CEF/1.0"})
        resp.raise_for_status()
        html_content = resp.text

    main_text = ""
    # Try trafilatura first
    try:
        # pyrefly: ignore [missing-import]
        import trafilatura

        extracted = trafilatura.extract(html_content)
        if extracted:
            main_text = extracted
    except Exception:
        pass

    # Fallback to BeautifulSoup4
    if not main_text:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, "html.parser")
            # Remove scripts, styles, navs, footers
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            main_text = soup.get_text(separator="\n\n")
        except Exception as exc:
            raise RuntimeError(f"Failed to extract text from URL: {exc}") from exc

    return extract_from_raw_text(main_text, default_instruction=f"Answer based on the web article from {url}.")


# ─────────────────────────────────────────────────────────────
# Unified CEF Main Ingestion Entry Point
# ─────────────────────────────────────────────────────────────
def process_cef_file(filename: str, contents: bytes) -> dict[str, Any]:
    """Process any uploaded file (PDF, Excel, CSV, TXT, JSON) via CEF pipeline.

    Returns dict containing validation status, extracted pairs, quality score,
    deduplicated row count, and dataset preview.
    """
    ext = Path(filename).suffix.lower()
    raw_pairs: list[ExtractedPair] = []

    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        try:
            df = pd.read_csv(io.BytesIO(contents), sep=sep)
            raw_pairs = extract_from_dataframe(df)
        except Exception:
            raw_pairs = extract_from_raw_text(contents.decode("utf-8", errors="ignore"))

    elif ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(io.BytesIO(contents))
            raw_pairs = extract_from_dataframe(df)
        except Exception as exc:
            raise ValueError(f"Could not read Excel file: {exc}") from exc

    elif ext == ".pdf":
        raw_pairs = extract_from_pdf_bytes(contents)

    elif ext in (".json", ".jsonl"):
        try:
            str_data = contents.decode("utf-8", errors="ignore")
            if ext == ".jsonl" or "\n" in str_data.strip():
                lines = [json.loads(line) for line in str_data.strip().split("\n") if line.strip()]
                df = pd.DataFrame(lines)
            else:
                data = json.loads(str_data)
                df = pd.DataFrame(data if isinstance(data, list) else [data])
            raw_pairs = extract_from_dataframe(df)
        except Exception:
            raw_pairs = extract_from_raw_text(contents.decode("utf-8", errors="ignore"))

    else:
        # Default text / markdown / unknown format
        text = contents.decode("utf-8", errors="ignore")
        raw_pairs = extract_from_raw_text(text)

    # Deduplicate & Clean
    cleaned_pairs = deduplicate_pairs(raw_pairs)
    quality_score = calculate_quality_score(cleaned_pairs, len(raw_pairs))

    return {
        "filename": filename,
        "raw_items_found": len(raw_pairs),
        "clean_pairs_count": len(cleaned_pairs),
        "quality_score": quality_score,
        "sample_pairs": [p.to_dict() for p in cleaned_pairs[:10]],
        "pairs": [p.to_dict() for p in cleaned_pairs],
    }
