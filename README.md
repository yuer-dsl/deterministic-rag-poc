# deterministic-rag-poc

A minimal, fully deterministic "global + local" search pipeline.

This repo is **not** a full RAG framework.  
It’s a small, self-contained script to explore one question:

> How far can we go with reproducible, planner-free, multi-hop-free retrieval  
> before we need complex graph-based graph systems?

## What this PoC does

- Builds a deterministic index over a tiny corpus:
  - TF-IDF document vectors
  - KMeans clustering with a fixed random_state
  - One document nearest to the cluster center is used as a frozen *community summary*
- At query time:
  - Encodes the query with the same TF-IDF model
  - Routes the query to the closest cluster center
  - Performs exact similarity search **only inside that community**
  - Formats a purely symbolic “answer” (no LLM calls in this PoC)

There is:

- no planner
- no multi-hop traversal
- no temperature / sampling
- no hidden randomness

Same corpus + same query → same route → same result.

## Why share this

This PoC is meant to be a tiny, reproducible baseline for discussions about:

- determinism vs. planner-driven orchestration
- stability of global+local search
- how much structure we can get *without* graph expansions

It is intentionally simple and incomplete.  
Real systems may swap TF-IDF for better encoders, add LLMs with strict `temperature=0, top_p=1`,  
or plug this into larger pipelines.

## How to run

```bash
python deterministic_rag_poc.py
