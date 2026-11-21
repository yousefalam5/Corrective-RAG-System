"""
Corrective Retrieval-Augmented Generation (CRAG) Engine
Implements an end-to-end CRAG pipeline:
1. Document Ingestion (PDF with OCR fallback, PPTX, DOCX, TXT, CSV)
2. Semantic Chunking & Vector Indexing via ChromaDB
3. Retrieval Evaluator / Document Grader
4. Query Rewriter & Re-retrieval
5. Hallucination-free Grounded Generation with Source Citations
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import torch
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".csv", ".txt"}


@dataclass
class CRAGConfig:
    data_dir: Path = field(default_factory=lambda: Path("sample_docs"))
    chroma_dir: Path = field(default_factory=lambda: Path("data/chroma_db"))
    collection_name: str = "crag_knowledge_base"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 4
    relevance_threshold: float = 0.35
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.chroma_dir = Path(self.chroma_dir).expanduser().resolve()


class CRAGEngine:
    def __init__(self, config: CRAGConfig | None = None) -> None:
        self.config = config or CRAGConfig()
        self.config.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(self.config.embedding_model, device=self.device)
        self._ocr_engine = None
        self._gemini_client = None

        if self.config.gemini_api_key:
            self._init_gemini(self.config.gemini_api_key)

    @property
    def collection(self):
        return self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @collection.setter
    def collection(self, val):
        pass

    def _init_gemini(self, api_key: str) -> None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._gemini_client = genai.GenerativeModel(self.config.gemini_model)
            self.config.gemini_api_key = api_key
        except Exception:
            self._gemini_client = None

    def update_gemini_key(self, api_key: str) -> bool:
        if not api_key:
            self._gemini_client = None
            return False
        try:
            self._init_gemini(api_key.strip())
            return True
        except Exception:
            return False

    @staticmethod
    def _clean_text(value: str) -> str:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", value)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    def _read_file(self, path: Path) -> list[tuple[str, dict[str, Any]]]:
        suffix = path.suffix.lower()
        output: list[tuple[str, dict[str, Any]]] = []

        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = self._clean_text(page.extract_text() or "")
                if text:
                    output.append((text, {"page": page_number}))
            if not output:
                # OCR fallback for scanned PDFs
                try:
                    import pypdfium2 as pdfium
                    from rapidocr_onnxruntime import RapidOCR

                    if self._ocr_engine is None:
                        self._ocr_engine = RapidOCR()

                    pdf = pdfium.PdfDocument(str(path))
                    for page_number, page in enumerate(pdf, start=1):
                        image = page.render(scale=1.5).to_pil()
                        result, _ = self._ocr_engine(np.array(image))
                        if result:
                            page_text = "\n".join(item[1] for item in result if item and len(item) > 1 and item[1])
                            page_text = self._clean_text(page_text)
                            if page_text:
                                output.append((page_text, {"page": page_number}))
                except Exception:
                    pass

        elif suffix == ".pptx":
            presentation = Presentation(str(path))
            for slide_number, slide in enumerate(presentation.slides, start=1):
                pieces = []
                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False):
                        text = "\n".join(paragraph.text for paragraph in shape.text_frame.paragraphs)
                        if text.strip():
                            pieces.append(text)
                    if getattr(shape, "has_table", False):
                        for row in shape.table.rows:
                            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                            if cells:
                                pieces.append(" | ".join(cells))
                slide_text = self._clean_text("\n".join(pieces))
                if slide_text:
                    output.append((slide_text, {"slide": slide_number}))

        elif suffix == ".docx":
            document = Document(str(path))
            pieces = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        pieces.append(" | ".join(cells))
            text = self._clean_text("\n".join(pieces))
            if text:
                output.append((text, {"section": "document"}))

        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.reader(handle)
                rows = [", ".join(cell.strip() for cell in row if cell.strip()) for row in reader]
            text = self._clean_text("\n".join(row for row in rows if row))
            if text:
                output.append((text, {"type": "csv_data"}))

        elif suffix == ".txt":
            text = self._clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                output.append((text, {"type": "plain_text"}))

        return output

    def ingest_files(self, file_paths: list[Path] | None = None) -> dict[str, Any]:
        """Ingests specified files or scans data_dir if none provided."""
        paths_to_process = []
        if file_paths:
            paths_to_process = [Path(p) for p in file_paths if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS]
        else:
            if self.config.data_dir.exists():
                for root, _, files in os.walk(self.config.data_dir):
                    for file in files:
                        p = Path(root) / file
                        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                            paths_to_process.append(p)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[dict[str, Any]] = []
        warnings: list[str] = []
        processed_files = 0

        for path in paths_to_process:
            filename = path.name
            try:
                segments = self._read_file(path)
            except Exception as exc:
                warnings.append(f"Could not read {filename}: {exc}")
                continue

            if not segments:
                warnings.append(f"No extractable text found in {filename}.")
                continue

            processed_files += 1
            for text, meta in segments:
                docs = splitter.split_text(text)
                for idx, doc in enumerate(docs):
                    chunk_id = hashlib.sha1(f"{filename}_{idx}_{doc[:40]}".encode("utf-8")).hexdigest()
                    metadata = {"source": filename, **meta}
                    chunks.append({"id": chunk_id, "text": doc, "metadata": metadata})

        if not chunks:
            return {"files": processed_files, "chunks": 0, "warnings": warnings}

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        try:
            self.client.delete_collection(self.config.collection_name)
        except Exception:
            pass

        new_collection = self.client.create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            new_collection.upsert(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                embeddings=embeddings[start : start + len(batch)].tolist(),
                metadatas=[item["metadata"] for item in batch],
            )

        return {"files": processed_files, "chunks": len(chunks), "warnings": warnings}

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Performs raw vector search in ChromaDB."""
        k = top_k or self.config.top_k
        if self.collection.count() == 0:
            return []

        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return output

        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            similarity = max(0.0, 1.0 - float(dist))
            output.append({"text": doc, "metadata": meta, "similarity": round(similarity, 4)})
        return output

    def grade_retrieval(self, query: str, chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, float]:
        """
        Retrieval Evaluator (Document Grader)
        Evaluates the relevance of each retrieved chunk against the user query.
        Returns:
            - graded_chunks: list of chunks with evaluation scores and 'RELEVANT' / 'IRRELEVANT' labels
            - overall_verdict: 'CORRECT' | 'WEAK' | 'INCORRECT'
            - avg_score: average relevance score
        """
        if not chunks:
            return [], "INCORRECT", 0.0

        graded_chunks = []
        relevant_count = 0
        scores = []

        query_terms = set(re.findall(r"\w+", query.lower()))

        for item in chunks:
            text = item["text"]
            sim = item["similarity"]

            # Lexical term overlap ratio
            chunk_terms = set(re.findall(r"\w+", text.lower()))
            overlap = len(query_terms.intersection(chunk_terms)) / max(1, len(query_terms))

            # Composite grading score (embedding similarity + term overlap)
            grade_score = round((sim * 0.7) + (overlap * 0.3), 4)
            scores.append(grade_score)

            is_relevant = (grade_score >= self.config.relevance_threshold) and (sim >= 0.40 or overlap >= 0.25)
            if is_relevant:
                relevant_count += 1

            graded_chunks.append({
                **item,
                "grade_score": grade_score,
                "verdict": "RELEVANT" if is_relevant else "IRRELEVANT",
                "explanation": f"Cosine Sim: {sim:.2f}, Term Overlap: {overlap:.2f}",
            })

        avg_score = round(sum(scores) / len(scores), 4)

        if relevant_count >= max(1, len(chunks) // 2) and avg_score >= self.config.relevance_threshold:
            verdict = "CORRECT"
        elif relevant_count > 0:
            verdict = "WEAK"
        else:
            verdict = "INCORRECT"

        return graded_chunks, verdict, avg_score

    def rewrite_query(self, query: str) -> str:
        """
        Query Rewriter / Transformation:
        Reformulates weak or ambiguous queries into clear, keyword-rich search terms.
        """
        if self._gemini_client:
            try:
                prompt = (
                    "You are a search query optimizer for a technical document retrieval system. "
                    "Analyze the user query, remove conversational noise, and produce a concise, "
                    "highly specific search phrase containing domain keywords.\n"
                    f"User Query: {query}\n"
                    "Output ONLY the rewritten search query with no explanations or punctuation:"
                )
                response = self._gemini_client.generate_content(prompt)
                rewritten = response.text.strip().replace('"', "")
                if rewritten:
                    return rewritten
            except Exception:
                pass

        # Smart algorithmic query rewriting fallback
        stopwords = {
            "what", "is", "the", "are", "of", "in", "a", "an", "and", "how", "does", "to",
            "tell", "me", "about", "can", "you", "explain", "i", "want", "know", "please",
            "ما", "هو", "هي", "عن", "في", "من", "إلى", "على", "اشرح", "اريد", "معرفة"
        }
        words = re.findall(r"\w+", query)
        filtered = [w for w in words if w.lower() not in stopwords]
        if filtered:
            return " ".join(filtered)
        return query

    def generate_grounded_answer(self, query: str, verified_chunks: list[dict[str, Any]]) -> str:
        """
        Generates an answer strictly grounded in verified context chunks.
        Prevents hallucination by gracefully declining if verified context is absent.
        """
        if not verified_chunks:
            return (
                "⚠️ **No verified information found.**\n\n"
                "The system evaluated the retrieved document context and determined that there is "
                "no sufficiently relevant information in the uploaded materials to answer this question accurately. "
                "To prevent hallucination, an ungrounded answer was not generated."
            )

        context_blocks = []
        for idx, chunk in enumerate(verified_chunks, start=1):
            source = chunk["metadata"].get("source", "Unknown file")
            page = chunk["metadata"].get("page") or chunk["metadata"].get("slide") or "N/A"
            context_blocks.append(f"[Source {idx}: {source} (Page/Slide {page})]\n{chunk['text']}")

        full_context = "\n\n---\n\n".join(context_blocks)

        if self._gemini_client:
            try:
                prompt = (
                    "You are a strict, factual academic assistant. Answer the user question solely "
                    "based on the verified context passages provided below. "
                    "Do NOT extrapolate or invent facts not present in the text. "
                    "Cite the sources explicitly using [Source X: filename] notation.\n\n"
                    f"CONTEXT:\n{full_context}\n\n"
                    f"USER QUESTION: {query}\n\n"
                    "GROUNDED ANSWER:"
                )
                response = self._gemini_client.generate_content(prompt)
                return response.text.strip()
            except Exception as exc:
                pass

        # Local synthesis fallback
        summary_intro = f"Based on {len(verified_chunks)} verified document passages:\n\n"
        passages = []
        for idx, chunk in enumerate(verified_chunks, start=1):
            source = chunk["metadata"].get("source", "Document")
            page_info = f" (Page/Slide {chunk['metadata'].get('page') or chunk['metadata'].get('slide')})" if 'page' in chunk['metadata'] or 'slide' in chunk['metadata'] else ""
            passages.append(f"**From `{source}`{page_info}:**\n> {chunk['text']}\n")
        return summary_intro + "\n".join(passages)

    def run_crag_pipeline(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Executes the complete Corrective RAG pipeline:
        1. Retrieve initial chunks.
        2. Evaluate / Grade chunks.
        3. If Weak or Incorrect -> Rewrite query and Re-retrieve.
        4. Select verified relevant chunks.
        5. Generate grounded answer.
        """
        k = top_k or self.config.top_k
        query = query.strip()
        if not query:
            return {"status": "ERROR", "message": "Query cannot be empty."}

        # Step 1: Initial Retrieval
        initial_chunks = self.retrieve(query, top_k=k)
        if not initial_chunks:
            return {
                "status": "UNANSWERED",
                "original_query": query,
                "initial_chunks": [],
                "retrieval_grade": "INCORRECT",
                "was_rewritten": False,
                "rewritten_query": None,
                "verified_chunks": [],
                "answer": (
                    "⚠️ **Knowledge base is empty.** Please upload or ingest documents first."
                ),
                "sources": [],
            }

        # Step 2: Retrieval Evaluation
        initial_graded, verdict, initial_avg_score = self.grade_retrieval(query, initial_chunks)

        was_rewritten = False
        rewritten_query = None
        second_chunks = []
        second_graded = []

        # Step 3: Corrective Decision
        if verdict in ("WEAK", "INCORRECT"):
            was_rewritten = True
            rewritten_query = self.rewrite_query(query)
            second_chunks = self.retrieve(rewritten_query, top_k=k)
            second_graded, second_verdict, second_avg_score = self.grade_retrieval(rewritten_query, second_chunks)

            # Combine verified chunks from both attempts, prioritizing second if improved
            verified_chunks = [c for c in second_graded if c["verdict"] == "RELEVANT"]
            if not verified_chunks:
                # Fall back to any initial relevant chunks
                verified_chunks = [c for c in initial_graded if c["verdict"] == "RELEVANT"]
        else:
            verified_chunks = [c for c in initial_graded if c["verdict"] == "RELEVANT"]

        # Deduplicate verified chunks
        unique_verified: list[dict[str, Any]] = []
        seen_texts = set()
        for c in verified_chunks:
            normalized = c["text"][:100]
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_verified.append(c)

        # Step 4: Grounded Generation
        answer = self.generate_grounded_answer(query, unique_verified)

        # Sources summary
        sources = []
        for c in unique_verified:
            source_file = c["metadata"].get("source", "Document")
            page = c["metadata"].get("page") or c["metadata"].get("slide")
            sources.append({
                "source": source_file,
                "page": page,
                "similarity": c["similarity"],
                "grade_score": c["grade_score"],
            })

        status = "SUCCESS" if unique_verified else "UNANSWERED"
        if was_rewritten and unique_verified:
            status = "CORRECTED"

        return {
            "status": status,
            "original_query": query,
            "initial_chunks": initial_graded,
            "retrieval_grade": verdict,
            "initial_avg_score": initial_avg_score,
            "was_rewritten": was_rewritten,
            "rewritten_query": rewritten_query,
            "second_chunks": second_graded,
            "verified_chunks": unique_verified,
            "answer": answer,
            "sources": sources,
        }
