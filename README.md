# RAG-Based Question Answering System

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers user queries based on a given document.
The system retrieves relevant information using vector similarity search and generates grounded answers using a language model.

---

## Features

* Document ingestion and chunking
* Text embeddings using SentenceTransformers
* Vector storage and similarity search using FAISS
* Top-K retrieval of relevant context
* Prompt-based answer generation using HuggingFace
* Fallback mechanism for unknown queries

---

## Project Structure

```
project-root/
├── rag_pipeline.py
├── requirements.txt
├── README.md
└── data/
    └── document.txt
```

---

## Required Libraries

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries used:

* sentence-transformers
* faiss-cpu
* transformers
* torch
* numpy

---

## How to Run the Project

### Option 1: Run on Google Colab

1. Upload:

   * `rag_pipeline.py`
   * `requirements.txt`
   * `document.txt`

2. Create folder structure:

```python
import os, shutil
os.makedirs("data", exist_ok=True)
shutil.move("document.txt", "data/document.txt")
```

3. Install dependencies:

```python
!pip install -r requirements.txt
```

4. Run:

```python
!python rag_pipeline.py --llm huggingface
```

---

### Option 2: Run on Local System

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
python rag_pipeline.py --llm huggingface
```

---

## System Architecture (Data Flow)

```
Document → Chunking → Embedding → FAISS → Retrieval → Generation
```

### Explanation:

* **Chunking**: Splits document into smaller parts
* **Embedding**: Converts text into vector form
* **FAISS**: Stores vectors for fast similarity search
* **Retrieval**: Finds most relevant chunks
* **Generation**: LLM generates answer using retrieved context

---

## Proof of Execution

### Example 1

**Question:**
What is the Solar System?

**Answer:**
The Solar System consists of the Sun and the celestial objects bound to it by gravity.

---

### Example 2

**Question:**
Why is Venus the hottest planet?

**Answer:**
Venus is the hottest planet due to its thick carbon dioxide atmosphere causing a runaway greenhouse effect.

---

### Example 3 (Fallback)

**Question:**
Who is the President of India?

**Answer:**
I could not find this in the provided document.

---

## Discussion Questions

### 1. Why are text embeddings necessary for an effective retrieval system?

Text embeddings convert textual data into numerical vector representations that capture semantic meaning.
This allows the system to perform similarity search based on meaning rather than exact keyword matching, enabling more accurate and relevant retrieval.

---

### 2. Why use a RAG approach instead of simply passing the entire document to the LLM?

Passing the entire document can exceed token limits and introduce irrelevant information.
RAG retrieves only the most relevant context, making the system more efficient, scalable, and accurate while reducing hallucinations.

---

### 3. How does chunk size impact retrieval quality?

* Smaller chunks → more precise retrieval but may lose context
* Larger chunks → more context but less precise

An optimal chunk size balances context and relevance for accurate retrieval.

---

### 4. What are the limitations of this RAG system?

* Depends heavily on retrieval quality
* Smaller LLM may produce less accurate answers
* No reranking mechanism
* Limited to single-document knowledge
* Cannot handle very complex reasoning queries

---

## Conclusion

This project demonstrates a complete implementation of a RAG pipeline, combining retrieval and generation to produce grounded and context-aware answers. It reflects real-world techniques used in modern AI systems.

---

## Submission Contents

* Source Code (`rag_pipeline.py`)
* Documentation (`README.md`)
* Requirements (`requirements.txt`)
* Data (`document.txt`)
