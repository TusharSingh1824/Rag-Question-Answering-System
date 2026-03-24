#!/usr/bin/env python3
"""
rag_pipeline.py — Foundational RAG (Retrieval-Augmented Generation) Q&A System
================================================================================

Pipeline Flow:
    Document (.txt) → Chunking (3-5 lines) → Embedding (all-MiniLM-L6-v2)
    → FAISS Index → Similarity Search → Prompt Construction → LLM Answer

Supported LLM Backends:
    - gemini   : Google Gemini API (free key from https://aistudio.google.com/apikey)
    - openai   : OpenAI GPT API
    - ollama   : Local LLMs via Ollama (no API key needed)
    - huggingface : Local HuggingFace model (no API key needed, that I used)

Usage:
    python rag_pipeline.py
    python rag_pipeline.py --file data/document.txt --llm gemini
    python rag_pipeline.py --file data/document.txt --llm gemini --api-key YOUR_KEY
    python rag_pipeline.py --llm ollama
    python rag_pipeline.py --llm huggingface
"""

import os
import sys
import time
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

DEFAULT_DOCUMENT_PATH = os.path.join("data", "document.txt")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"          
CHUNK_SIZE_LINES = 4                                 
TOP_K = 3                                            
FALLBACK_RESPONSE = "I could not find this in the provided document."

# Gemini retry settings
MAX_RETRIES = 3
BASE_RETRY_DELAY = 20  # seconds


# ============================================================================
# STAGE 1: DOCUMENT INGESTION & CHUNKING
# ============================================================================

def load_document(file_path: str) -> List[str]:
    """
    Load a .txt file and return all lines.

    Args:
        file_path (str): Path to the .txt document.

    Returns:
        List[str]: List of all lines (stripped of trailing whitespace).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a .txt file or is empty.
    """
    # Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at: '{file_path}'")

    # Validate file extension
    if not file_path.lower().endswith('.txt'):
        raise ValueError(f"Only .txt files are supported. Got: '{file_path}'")

    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Strip trailing whitespace/newlines from each line
    lines = [line.rstrip() for line in lines]

    # Validate non-empty content
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        raise ValueError(f"Document is empty: '{file_path}'")

    print(f"  [✓] Loaded document: {file_path}")
    print(f"      Total lines: {len(lines)} | Non-empty lines: {len(non_empty)}")
    return lines


def create_chunks(lines: List[str], chunk_size: int = CHUNK_SIZE_LINES) -> List[Dict]:
    """
    Split document lines into chunks of `chunk_size` non-empty lines each,
    preserving metadata (chunk ID, original line range).

    If the last chunk has fewer than 2 lines, it is merged with the previous chunk
    to maintain meaningful chunk sizes.

    Args:
        lines (List[str]): All lines from the document.
        chunk_size (int): Number of non-empty lines per chunk (default: 4).

    Returns:
        List[Dict]: List of chunk dictionaries, each containing:
            - chunk_id (int): Unique sequential identifier.
            - text (str): Concatenated text content of the chunk.
            - start_line (int): Starting line number (1-indexed).
            - end_line (int): Ending line number (1-indexed).
            - num_lines (int): Number of non-empty lines in the chunk.
    """
    # Collect non-empty lines with their original 1-indexed line numbers
    indexed_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped:
            indexed_lines.append((i + 1, stripped))  # (line_number, text)

    if not indexed_lines:
        return []

    # Group into chunks of `chunk_size` lines
    raw_chunks = []
    for i in range(0, len(indexed_lines), chunk_size):
        group = indexed_lines[i : i + chunk_size]
        raw_chunks.append(group)

    # Merge last chunk if it has fewer than 2 lines (for quality)
    if len(raw_chunks) > 1 and len(raw_chunks[-1]) < 2:
        raw_chunks[-2].extend(raw_chunks[-1])
        raw_chunks.pop()

    # Build chunk dictionaries with metadata
    chunks = []
    for chunk_id, group in enumerate(raw_chunks):
        chunk_text = " ".join(text for _, text in group)
        start_line = group[0][0]
        end_line = group[-1][0]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_line": start_line,
            "end_line": end_line,
            "num_lines": len(group),
        })

    print(f"  [✓] Created {len(chunks)} chunks (target size: {chunk_size} lines/chunk)")
    return chunks


def display_chunks(chunks: List[Dict]) -> None:
    """Print a formatted summary table of all chunks."""
    print(f"\n  {'ID':>4}  {'Lines':>10}  {'Size':>5}  {'Preview'}")
    print(f"  {'—'*4}  {'—'*10}  {'—'*5}  {'—'*50}")
    for c in chunks:
        preview = c['text'][:50] + "..." if len(c['text']) > 50 else c['text']
        print(f"  {c['chunk_id']:>4}  {c['start_line']:>4}-{c['end_line']:<5}  "
              f"{c['num_lines']:>5}  {preview}")


# ============================================================================
# STAGE 2: TEXT EMBEDDING
# ============================================================================

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Load a sentence-transformers model for generating text embeddings.

    Args:
        model_name (str): HuggingFace model identifier.

    Returns:
        SentenceTransformer: Loaded embedding model.
    """
    print(f"  [...] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"  [✓] Model loaded — Embedding dimension: {dim}")
    return model


def generate_embeddings(
    model: SentenceTransformer, chunks: List[Dict]
) -> np.ndarray:
    """
    Generate normalized embeddings for all text chunks.

    Embeddings are L2-normalized so that inner product equals cosine similarity,
    which is required for FAISS IndexFlatIP.

    Args:
        model (SentenceTransformer): Loaded embedding model.
        chunks (List[Dict]): List of chunk dictionaries.

    Returns:
        np.ndarray: Normalized embeddings of shape (num_chunks, embedding_dim).
    """
    texts = [chunk["text"] for chunk in chunks]

    # Encode all chunk texts into dense vectors
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We'll normalize manually for clarity
    )

    # L2-normalize so that inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    print(f"  [✓] Generated & normalized embeddings — Shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


# ============================================================================
# STAGE 3: VECTOR STORAGE (FAISS)
# ============================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS flat index using Inner Product (cosine similarity on
    L2-normalized vectors).

    The mapping from FAISS index position → chunk is positional:
        FAISS index position i  ↔  chunks[i]

    Args:
        embeddings (np.ndarray): Normalized embedding matrix.

    Returns:
        faiss.IndexFlatIP: Populated FAISS index.
    """
    dimension = embeddings.shape[1]

    # IndexFlatIP computes inner product; on normalized vectors this equals
    # cosine similarity, giving values in [-1, 1] where 1 = most similar.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"  [✓] FAISS index built — {index.ntotal} vectors, dimension {dimension}")
    return index


# ============================================================================
# STAGE 4: SEARCH & RETRIEVAL
# ============================================================================

def retrieve_relevant_chunks(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    top_k: int = TOP_K,
) -> List[Tuple[Dict, float]]:
    """
    Retrieve the top-k most relevant chunks for a given query using
    cosine similarity search.

    Args:
        query (str): User's natural language question.
        model (SentenceTransformer): Embedding model (same one used for chunks).
        index (faiss.IndexFlatIP): Populated FAISS index.
        chunks (List[Dict]): Original chunk list (for positional mapping).
        top_k (int): Number of top results to return.

    Returns:
        List[Tuple[Dict, float]]: List of (chunk_dict, cosine_similarity_score)
                                   tuples, sorted by descending relevance.
    """
    # Encode and normalize the query using the same model
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # Perform similarity search against the FAISS index
    scores, indices = index.search(query_embedding.astype(np.float32), top_k)

    # Map FAISS positions back to chunk dictionaries
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(chunks):
            results.append((chunks[idx], float(score)))

    return results


def display_retrieval_results(results: List[Tuple[Dict, float]]) -> None:
    """Print formatted retrieval results."""
    for rank, (chunk, score) in enumerate(results, 1):
        preview = (chunk['text'][:120] + "..."
                   if len(chunk['text']) > 120 else chunk['text'])
        print(f"    [{rank}] Chunk {chunk['chunk_id']} | "
              f"Lines {chunk['start_line']}-{chunk['end_line']} | "
              f"Score: {score:.4f}")
        print(f"        \"{preview}\"")


# ============================================================================
# STAGE 5: ANSWER GENERATION (LLM)
# ============================================================================

def build_prompt(query: str, retrieved_chunks: List[Tuple[Dict, float]]) -> str:
    """
    Construct a grounded prompt that instructs the LLM to answer based ONLY
    on the retrieved context.

    Args:
        query (str): The user's question.
        retrieved_chunks (List[Tuple[Dict, float]]): Retrieved (chunk, score) pairs.

    Returns:
        str: The complete prompt for the LLM.
    """
    # Format each retrieved chunk with its metadata
    context_sections = []
    for i, (chunk, score) in enumerate(retrieved_chunks, 1):
        context_sections.append(
            f"[Source {i} — Chunk {chunk['chunk_id']}, "
            f"Lines {chunk['start_line']}-{chunk['end_line']}]\n"
            f"{chunk['text']}"
        )

    context_block = "\n\n".join(context_sections)

    prompt = (
        "You are a precise Q&A assistant. Your task is to answer the user's question "
        "using ONLY the context provided below.\n\n"
        "RULES:\n"
        "1. Base your answer strictly on the provided context.\n"
        "2. If the context does not contain sufficient information to answer the "
        "question, respond EXACTLY with: "
        "\"I could not find this in the provided document.\"\n"
        "3. Do NOT use any prior knowledge or make assumptions beyond the context.\n"
        "4. Keep your answer concise and directly relevant.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )
    return prompt


# --------------- LLM Backend: Google Gemini (NEW SDK) ---------------

def generate_answer_gemini(prompt: str, api_key: str) -> str:
    """
    Generate an answer using Google Gemini API with the NEW google-genai SDK.
    Includes automatic retry logic with exponential backoff for rate limits.

    Args:
        prompt (str): The fully constructed prompt.
        api_key (str): Google API key.

    Returns:
        str: LLM-generated answer or fallback response.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  [!] Install the new SDK: pip install google-genai")
        print("  [!] Also uninstall the old one: pip uninstall google-generativeai")
        return FALLBACK_RESPONSE

    # Initialize the Gemini client with API key
    client = genai.Client(api_key=api_key)

    # List of models to try (fallback chain)
    models_to_try = ["gemini-2.0-flash-exp"]

    for model_name in models_to_try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  [...] Calling {model_name} (attempt {attempt}/{MAX_RETRIES})")

                # Generate content using the new SDK
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                    ),
                )

                # Extract the answer text
                answer = response.text.strip() if response.text else ""

                if answer:
                    print(f"  [✓] Response received from {model_name}")
                    return answer
                else:
                    print(f"  [!] Empty response from {model_name}")
                    return FALLBACK_RESPONSE

            except Exception as e:
                error_str = str(e)

                # Handle rate limit errors (429) with retry
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = BASE_RETRY_DELAY * attempt
                    print(f"  [!] Rate limited on {model_name}. "
                          f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                # Handle other errors — try next model
                else:
                    print(f"  [!] Error with {model_name}: {e}")
                    break  # Move to next model

        # If all retries for this model failed, continue to next model
        print(f"  [!] All retries exhausted for {model_name}, trying next model...")

    print("  [!] All Gemini models failed. Consider using --llm ollama or --llm huggingface")
    return FALLBACK_RESPONSE


# --------------- LLM Backend: OpenAI ---------------

def generate_answer_openai(prompt: str, api_key: str) -> str:
    """Generate an answer using the OpenAI Chat API."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions based only on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()
        return answer if answer else FALLBACK_RESPONSE

    except ImportError:
        print("  [!] Install openai: pip install openai")
        return FALLBACK_RESPONSE
    except Exception as e:
        print(f"  [!] OpenAI API error: {e}")
        return FALLBACK_RESPONSE


# --------------- LLM Backend: Ollama (Local) ---------------

def generate_answer_ollama(prompt: str, model_name: str = "llama3") -> str:
    """Generate an answer using a local Ollama model (no API key needed)."""
    try:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return answer if answer else FALLBACK_RESPONSE

    except ImportError:
        print("  [!] Install requests: pip install requests")
        return FALLBACK_RESPONSE
    except Exception as e:
        print(f"  [!] Ollama error (is Ollama running?): {e}")
        return FALLBACK_RESPONSE


# --------------- LLM Backend: HuggingFace Local ---------------

def generate_answer_huggingface(prompt: str) -> str:
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        # Load once
        if not hasattr(generate_answer_huggingface, "model"):
            print("  [...] Loading HuggingFace model (flan-t5-base)...")
            generate_answer_huggingface.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            generate_answer_huggingface.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        tokenizer = generate_answer_huggingface.tokenizer
        model = generate_answer_huggingface.model

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not answer or len(answer.split()) < 3:
            return "I could not find this in the provided document."

        return answer.strip()

    except Exception as e:
        print(f"  [!] HuggingFace error: {e}")
        return "I could not find this in the provided document."

# --------------- LLM Router ---------------

def generate_answer(
    prompt: str,
    llm_backend: str = "gemini",
    api_key: Optional[str] = None,
) -> str:
    """
    Route the prompt to the selected LLM backend and return the answer.

    Args:
        prompt (str): Fully constructed prompt with context and question.
        llm_backend (str): One of 'gemini', 'openai', 'ollama', 'huggingface'.
        api_key (Optional[str]): API key (needed for gemini/openai).

    Returns:
        str: The LLM-generated answer, or the fallback response.
    """
    if llm_backend == "gemini":
        key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            print("  [!] GOOGLE_API_KEY not set.")
            print("      Get a free key → https://aistudio.google.com/apikey")
            print("      Then: export GOOGLE_API_KEY=your_key  OR  use --api-key flag")
            return FALLBACK_RESPONSE
        return generate_answer_gemini(prompt, key)

    elif llm_backend == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            print("  [!] OPENAI_API_KEY not set.")
            return FALLBACK_RESPONSE
        return generate_answer_openai(prompt, key)

    elif llm_backend == "ollama":
        return generate_answer_ollama(prompt)

    elif llm_backend == "huggingface":
        return generate_answer_huggingface(prompt)

    else:
        print(f"  [!] Unknown LLM backend: '{llm_backend}'")
        return FALLBACK_RESPONSE


# ============================================================================
# PIPELINE ORCHESTRATION
# ============================================================================

def initialize_pipeline(
    file_path: str, model_name: str = EMBEDDING_MODEL_NAME
) -> Tuple[SentenceTransformer, faiss.IndexFlatIP, List[Dict]]:
    """
    Run the full initialization pipeline: load → chunk → embed → index.

    Args:
        file_path (str): Path to the .txt document.
        model_name (str): Sentence-transformer model name.

    Returns:
        Tuple containing:
            - SentenceTransformer model
            - FAISS index
            - List of chunk dictionaries
    """
    print("\n" + "=" * 65)
    print("   🔧  RAG PIPELINE — INITIALIZATION")
    print("=" * 65)

    # Stage 1: Document Ingestion & Chunking
    print("\n📄 STAGE 1: Document Ingestion & Chunking")
    lines = load_document(file_path)
    chunks = create_chunks(lines, chunk_size=CHUNK_SIZE_LINES)
    display_chunks(chunks)

    # Stage 2: Text Embedding
    print("\n🔢 STAGE 2: Text Embedding")
    embedding_model = load_embedding_model(model_name)
    embeddings = generate_embeddings(embedding_model, chunks)

    # Stage 3: Vector Storage
    print("\n📦 STAGE 3: FAISS Vector Storage")
    faiss_index = build_faiss_index(embeddings)

    print("\n" + "=" * 65)
    print("   ✅  PIPELINE READY — Ask your questions!")
    print("=" * 65)

    return embedding_model, faiss_index, chunks


def ask_question(
    question,
    embedding_model,
    faiss_index,
    chunks,
    llm_backend="huggingface",
    api_key=None,
    top_k=3,
):
    # Retrieval
    results = retrieve_relevant_chunks(
        question, embedding_model, faiss_index, chunks, top_k
    )

    if not results:
        return "I could not find this in the provided document."

    # Generate answer
    prompt = build_prompt(question, results)
    answer = generate_answer(prompt, llm_backend, api_key)

    # -------- PROFESSIONAL OUTPUT --------
    print("\n" + "="*60)
    print("RAG SYSTEM RESPONSE")
    print("="*60)

    print("\nQuestion:")
    print(question)

    print("\nRetrieved Context:")
    for i, (chunk, score) in enumerate(results, 1):
        preview = chunk["text"][:120] + "..."
        print(f"[{i}] Lines {chunk['start_line']}-{chunk['end_line']} | Score: {score:.4f}")
        print(f"    {preview}")

    print("\nFinal Answer:")
    print(answer)

    print("\n" + "="*60 + "\n")

    return answer

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Parse CLI arguments, initialize the pipeline, and run interactive Q&A loop."""

    parser = argparse.ArgumentParser(
        description="RAG Q&A Pipeline — Ask questions grounded in your document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python rag_pipeline.py\n"
            "  python rag_pipeline.py --file data/document.txt --llm gemini\n"
            "  python rag_pipeline.py --llm ollama\n"
            "  python rag_pipeline.py --llm huggingface   # No API key needed\n"
        ),
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=DEFAULT_DOCUMENT_PATH,
        help=f"Path to the .txt document (default: {DEFAULT_DOCUMENT_PATH})",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gemini",
        choices=["gemini", "openai", "ollama", "huggingface"],
        help="LLM backend for answer generation (default: gemini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM (or set GOOGLE_API_KEY / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve per query (default: {TOP_K})",
    )

    args = parser.parse_args()

    # ---- Initialize Pipeline ----
    try:
        embedding_model, faiss_index, chunks = initialize_pipeline(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # ---- Interactive Query Loop ----
    print(f"\n  LLM Backend : {args.llm}")
    print(f"  Top-K       : {args.top_k}")
    print("  Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("📝 Your Question: ").strip()

            if not question:
                print("  Please enter a question.\n")
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break

            answer = ask_question(
                question=question,
                embedding_model=embedding_model,
                faiss_index=faiss_index,
                chunks=chunks,
                llm_backend=args.llm,
                api_key=args.api_key,
                top_k=args.top_k,
            )

            print(f"\n💡 ANSWER: {answer}")
            print("—" * 65 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            continue


if __name__ == "__main__":
    main()
