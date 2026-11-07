# mini-rag

A small retrieval-augmented generation pipeline that combines dense embeddings (fastembed) with BM25 keyword search. It loads a local text corpus, retrieves the most relevant chunks for a question, and optionally drafts an answer from them.

## Setup

```powershell
. "C:\Users\davea\Documents\Avan local projects\activate.ps1"
cd "C:\Users\davea\Documents\Avan local projects\projects\mini-rag"
uv sync
. .\scripts\env.ps1
uv run pytest
```

## Status

The skeleton is in place; the logic is being built step by step.
