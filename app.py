from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import FileResponse

# ---------------------------
# Config
# ---------------------------

BASE_DIR = Path("rag_store")
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_PATH = BASE_DIR / "index.faiss"
CHUNKS_PATH = BASE_DIR / "chunks.json"
META_PATH = BASE_DIR / "meta.json"

BASE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
GEMINI_API_MODEL = None

# For multilingual text (including Persian), this model is a solid default.
embedder = SentenceTransformer(EMBED_MODEL_NAME)

app = FastAPI(title="RAG Agent for Textbook QA", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# ---------------------------
# Data Models
# ---------------------------

class IngestResponse(BaseModel):
    ok: bool
    message: str
    chunks: int
    pages: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class SourceItem(BaseModel):
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


# ---------------------------
# Utilities
# ---------------------------

def clean_text(text: str):

    import unicodedata

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")

    return text

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = clean_text(text)
    return text


def chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 40) -> List[str]:
    """
    Word-based chunking. Works reasonably well for Persian and mixed text.
    """
    text = normalize_text(text)
    text = clean_text(text)
    if not text:
        return []

    words = text.split()
    if len(words) <= chunk_size_words:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - overlap_words)

    return chunks


def extract_pdf_text(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Returns a list of page objects:
    [
        {"page": 1, "text": "..."},
        ...
    ]
    """
    reader = PdfReader(str(pdf_path))
    pages: List[Dict[str, Any]] = []

    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = normalize_text(raw_text)
        text = clean_text(text)
        pages.append({"page": i, "text": text})

    return pages


def build_chunks_from_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []

    for page_obj in pages:
        page_num = page_obj["page"]
        text = page_obj["text"]
        if not text:
            continue

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "page": page_num,
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

    return all_chunks


def save_chunks(chunks: List[Dict[str, Any]]) -> None:
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_chunks() -> List[Dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        return []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta: Dict[str, Any]) -> None:
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta() -> Dict[str, Any]:
    if not META_PATH.exists():
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """
    E5-style prompting:
    - query: ...
    - passage: ...
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    texts = [clean_text(t) for t in texts]
    prefix = "query: " if is_query else "passage: "
    prefixed = [prefix + t for t in texts]

    vectors = embedder.encode(
        prefixed,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return vectors.astype(np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No vectors to index.")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity after normalization
    index.add(vectors)
    return index


def save_index(index: faiss.Index) -> None:
    faiss.write_index(index, str(INDEX_PATH))


def load_index() -> Optional[faiss.Index]:
    if not INDEX_PATH.exists():
        return None
    return faiss.read_index(str(INDEX_PATH))


def search_index(
    question: str,
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    q_vec = embed_texts([question], is_query=True)
    if q_vec.size == 0:
        return []

    scores, ids = index.search(q_vec, top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0 or idx >= len(chunks):
            continue
        item = chunks[idx]
        results.append(
            {
                "page": item["page"],
                "score": float(score),
                "text": item["text"],
            }
        )
    return results


def build_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    context_lines = []
    for i, s in enumerate(sources, start=1):
        context_lines.append(f"[منبع {i} | صفحه {s['page']}] {s['text']}")

    context = "\n\n".join(context_lines) if context_lines else "هیچ منبعی پیدا نشد."

    return f"""
    تو یک دستیار آموزشی دقیق هستی.
    فقط بر اساس منابع زیر پاسخ بده.
    اگر پاسخ در منابع نبود، صریح بگو «در منابع موجود پیدا نشد».
    از حدس زدن خودداری کن.
    منابع:
    {context}

    سؤال:
    {question}
    """.strip()


def extract_keywords(question: str) -> List[str]:
    """
    تمام کلمات سوال را استخراج می‌کند.
    """
    return list(set(
        word.strip()
        for word in re.findall(r"\w+", question)
        if word.strip()
    ))


def highlight_text(text: str, keywords: List[str]) -> str:
    """
    کلمات پیدا شده را داخل <strong> قرار می‌دهد.
    """

    text = clean_text(text)

    # بلندترین کلمات اول جایگزین شوند
    keywords = sorted(keywords, key=len, reverse=True)

    for kw in keywords:
        text = re.sub(
            rf"({re.escape(kw)})",
            r"<strong style=\"background-color: yellow;\">\1</strong>",
            text,
            flags=re.IGNORECASE
        )

    return text


def fallback_keyword_search(
    question: str,
    sources: List[Dict[str, Any]]
):
    keywords = extract_keywords(question)

    results = []

    for source in sources:

        text = clean_text(source["text"])

        matched_keywords = [
            kw
            for kw in keywords
            if kw.lower() in text.lower()
        ]

        if matched_keywords:

            results.append({
                "page": source.get("page", -1),
                "matches": len(matched_keywords),
                "text": highlight_text(
                    text,
                    matched_keywords
                )
            })

    results.sort(
        key=lambda x: x["matches"],
        reverse=True
    )

    return results[:5]

def generate_answer(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    Uses OpenAI if API key exists.
    Falls back to keyword + highlighted retrieval if it fails.
    """

    prompt = build_prompt(question, sources)

    if GEMINI_API_MODEL:
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=GEMINI_API_MODEL
                )

            resp = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise Persian educational assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            )

            return resp.choices[0].message.content.strip()

        except Exception as e:

            # =========================
            # SMART FALLBACK (NEW)
            # =========================

            fallback_results = fallback_keyword_search(question, sources)

            if fallback_results:

                formatted = "\n\n".join(
                    f"📄 صفحه {r['page']}\n{r['text']}"
                    for r in fallback_results
                )

                return (
                    "⚠️ خطا در تولید پاسخ هوش مصنوعی.\n\n"
                    "📌 بخش‌های مرتبط از کتاب:\n\n"
                    f"{formatted}"
                    f"\n\n <details><summery>جزئیات خطا</summery>جرئیات خطا: {e}</details>"
                )

            # اگر هیچ چیزی پیدا نشد
            return (
                "⚠️ خطا در تولید پاسخ و هیچ متن مرتبطی هم پیدا نشد.\n\n",
                "<details><summery>جزئیات خطا</summery>",
                str(e),
                "</details>"
            )

    # اگر API key نبود
    fallback_results = fallback_keyword_search(question, sources)

    if fallback_results:
        return "\n\n".join(
            f"📄 صفحه {r['page']}\n{r['text']}"
            for r in fallback_results
        )

    return "هیچ پاسخی در منابع موجود پیدا نشد."


def fallback_answer_from_sources(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "در منابع موجود چیزی پیدا نشد."

    answer_parts = ["پاسخ بر اساس نزدیک‌ترین بخش‌های کتاب:\n"]
    for i, s in enumerate(sources[:3], start=1):
        answer_parts.append(f"{i}) صفحه {s['page']}: {s['text']}")
    return "\n\n".join(answer_parts)


def load_or_fail() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    index = load_index()
    chunks = load_chunks()
    if index is None or not chunks:
        raise HTTPException(
            status_code=400,
            detail="ایندکس هنوز ساخته نشده است. ابتدا /ingest را اجرا کن.",
        )
    return index, chunks


# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/")
def home():
    return FileResponse("templates/index.html")

@app.get("/health")
def health() -> Dict[str, Any]:
    index_exists = INDEX_PATH.exists()
    chunks_exists = CHUNKS_PATH.exists()
    meta = load_meta()

    return {
        "ok": True,
        "index_exists": index_exists,
        "chunks_exists": chunks_exists,
        "meta": meta,
        "embed_model": EMBED_MODEL_NAME,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="فقط فایل PDF پذیرفته می‌شود.")

    pdf_path = UPLOAD_DIR / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pages = extract_pdf_text(pdf_path)
    chunks = build_chunks_from_pages(pages)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="متنی از PDF استخراج نشد. احتمالاً فایل اسکن‌شده است و OCR لازم دارد.",
        )

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, is_query=False)
    index = build_faiss_index(vectors)

    save_index(index)
    save_chunks(chunks)
    save_meta(
        {
            "source_file": file.filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "embed_model": EMBED_MODEL_NAME,
        }
    )

    return IngestResponse(
        ok=True,
        message="PDF با موفقیت ingest شد و ایندکس ساخته شد.",
        chunks=len(chunks),
        pages=len(pages),
    )
# Debug Mode 👇: More verbose logging and no response model for easier debugging. 
# @app.post("/ingest")
# async def ingest_pdf(file: UploadFile = File(...)):

#     start = time.time()

#     print("=" * 50)
#     print("INGEST STARTED")
#     print(f"File name: {file.filename}")

#     try:
#         print("[1] Saving file...")

#         pdf_path = UPLOAD_DIR / file.filename

#         with open(pdf_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         print("[OK] File saved")
#         print(f"Time: {time.time() - start:.2f}s")

#         print("[2] Extracting PDF text...")

#         pages = extract_pdf_text(pdf_path)

#         print(f"[OK] Pages extracted: {len(pages)}")
#         print(f"Time: {time.time() - start:.2f}s")

#         print("[3] Creating chunks...")

#         chunks = build_chunks_from_pages(pages)

#         print(f"[OK] Chunks created: {len(chunks)}")
#         print(f"Time: {time.time() - start:.2f}s")

#         if not chunks:
#             print("[ERROR] No chunks found")
#             raise HTTPException(
#                 status_code=400,
#                 detail="No text extracted from PDF"
#             )

#         print("[4] Creating embeddings...")

#         texts = [c["text"] for c in chunks]

#         vectors = embed_texts(
#             texts,
#             is_query=False
#         )

#         print("[OK] Embeddings created")
#         print(f"Shape: {vectors.shape}")
#         print(f"Time: {time.time() - start:.2f}s")

#         print("[5] Building FAISS index...")

#         index = build_faiss_index(vectors)

#         print("[OK] FAISS index built")
#         print(f"Time: {time.time() - start:.2f}s")

#         print("[6] Saving files...")

#         save_index(index)
#         save_chunks(chunks)

#         print("[OK] Files saved")
#         print(f"Time: {time.time() - start:.2f}s")

#         print("INGEST FINISHED")
#         print("=" * 50)

#         return {
#             "ok": True,
#             "pages": len(pages),
#             "chunks": len(chunks)
#         }

#     except Exception as e:

#         import traceback

#         print("\nERROR OCCURRED:")
#         traceback.print_exc()

#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )

@app.post("/setapikey")
def set_api_key(apikey: str):
    GEMINI_API_MODEL = apikey

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    index, chunks = load_or_fail()

    sources = search_index(
        question=req.question,
        index=index,
        chunks=chunks,
        top_k=req.top_k,
    )

    answer = generate_answer(req.question, sources)

    return AskResponse(
        answer=answer,
        sources=[
            SourceItem(page=s["page"], score=s["score"], text=s["text"])
            for s in sources
        ],
    )


@app.post("/reset")
def reset_index() -> Dict[str, Any]:
    """
    Deletes the stored index and metadata.
    """
    for path in [INDEX_PATH, CHUNKS_PATH, META_PATH]:
        if path.exists():
            path.unlink()

    return {"ok": True, "message": "ایندکس و متادیتا حذف شدند."}


# ---------------------------
# Run with:
# uvicorn app:app --reload
# ---------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=2626, reload=True)