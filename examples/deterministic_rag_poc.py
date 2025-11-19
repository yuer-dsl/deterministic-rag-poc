"""
deterministic_rag_poc.py

Minimal deterministic RAG-style pipeline for discussion & experiments.

Goal:
- Show how to build a tiny, fully reproducible "global + local" search pipeline
  without dynamic planners or multi-hop graph traversal.
- This is intentionally simplified and not meant as a production system.

Notes:
- No LLM calls inside this PoC.
- No external "OS", "agent framework" or private protocol involved.
- All randomness is fixed via random_state to keep behavior stable.

License suggestion (you can change this in your repo):
- Educational / non-commercial use only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 0. Global deterministic setup
# -----------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# -----------------------------
# 1. Example documents
#    (Replace with your own)
# -----------------------------

DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "doc-1",
        "text": "Graph-based retrieval connects entities as nodes and edges to support multi-hop search.",
    },
    {
        "id": "doc-2",
        "text": "Multi-hop reasoning can suffer from semantic drift when expansions are not bounded.",
    },
    {
        "id": "doc-3",
        "text": "Deterministic pipelines are important for compliance and financial workloads.",
    },
    {
        "id": "doc-4",
        "text": "Clustering can group similar documents into stable communities for later retrieval.",
    },
    {
        "id": "doc-5",
        "text": "Sampling in large language models introduces randomness into the generated answers.",
    },
]


# -----------------------------
# 2. Simple data structures
# -----------------------------

@dataclass
class Community:
    label: int
    summary: str
    doc_ids: List[str]
    texts: List[str]
    tfidf_vectors: np.ndarray


@dataclass
class DeterministicIndex:
    vectorizer: TfidfVectorizer
    communities: Dict[int, Community]
    centers: np.ndarray


# -----------------------------
# 3. Build deterministic index
# -----------------------------

def build_deterministic_index(
    documents: List[Dict[str, str]],
    n_clusters: int = 2,
) -> DeterministicIndex:
    """
    Global step:
    - TF-IDF for document vectors (deterministic for a fixed corpus).
    - KMeans with fixed random_state → deterministic clustering.
    - For each cluster, we pick the document closest to its center as
      a frozen "community summary".
    """
    texts = [d["text"] for d in documents]
    doc_ids = [d["id"] for d in documents]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init="auto",
    )
    labels = kmeans.fit_predict(tfidf_matrix)
    centers = kmeans.cluster_centers_

    communities: Dict[int, Community] = {}

    for label in sorted(set(labels)):
        # Collect docs belonging to this cluster
        indices = [i for i, lab in enumerate(labels) if lab == label]
        if not indices:
            continue

        sub_vectors = tfidf_matrix[indices]
        sub_ids = [doc_ids[i] for i in indices]
        sub_texts = [texts[i] for i in indices]

        center = centers[label].reshape(1, -1)
        sims = cosine_similarity(center, sub_vectors)[0]
        best_idx = int(np.argmax(sims))

        summary_text = sub_texts[best_idx]

        communities[label] = Community(
            label=label,
            summary=summary_text,
            doc_ids=sub_ids,
            texts=sub_texts,
            tfidf_vectors=sub_vectors,
        )

    return DeterministicIndex(
        vectorizer=vectorizer,
        communities=communities,
        centers=centers,
    )


# -----------------------------
# 4. Deterministic local search
# -----------------------------

def route_to_community(
    query: str,
    index: DeterministicIndex,
) -> Community:
    """
    Route query to a single community by:
    - encoding query with the same TF-IDF vectorizer
    - selecting the cluster center with highest cosine similarity
    """
    q_vec = index.vectorizer.transform([query]).toarray()
    sims = cosine_similarity(q_vec, index.centers)[0]
    label = int(np.argmax(sims))
    return index.communities[label]


def retrieve_in_community(
    query: str,
    community: Community,
    index: DeterministicIndex,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Exact retrieval limited to a single community:
    - no multi-hop
    - no planner
    - no sampling
    """
    q_vec = index.vectorizer.transform([query]).toarray()
    sims = cosine_similarity(q_vec, community.tfidf_vectors)[0]
    order = np.argsort(-sims)

    results: List[Dict[str, Any]] = []
    for i in order[:top_k]:
        results.append(
            {
                "doc_id": community.doc_ids[i],
                "text": community.texts[i],
                "score": float(sims[i]),
            }
        )
    return results


def format_answer(
    query: str,
    community: Community,
    results: List[Dict[str, Any]],
) -> str:
    """
    Minimal, fully deterministic "answer" formatter.
    Real systems may plug an LLM here with strict temperature=0, top_p=1,
    but this PoC keeps it purely symbolic to avoid any hidden randomness.
    """
    lines: List[str] = []
    lines.append("[community_summary] " + community.summary)
    lines.append("")
    lines.append("Top matches:")
    for r in results:
        lines.append(f"- ({r['score']:.3f}) {r['doc_id']}: {r['text']}")
    lines.append("")
    lines.append("User query: " + query)
    return "\n".join(lines)


# -----------------------------
# 5. End-to-end demo
# -----------------------------

def demo() -> None:
    print(">>> Building deterministic index...")
    index = build_deterministic_index(DOCUMENTS, n_clusters=2)

    print("\n>>> Frozen community summaries:")
    for label, com in index.communities.items():
        print(f"  [Cluster {label}] {com.summary}")

    query = "Why do multi-hop graph systems drift over long chains?"
    print("\n>>> Query:", query)

    community = route_to_community(query, index)
    print(f"\n>>> Routed to cluster: {community.label}")
    print(f"    Summary: {community.summary}")

    results = retrieve_in_community(query, community, index, top_k=3)
    answer = format_answer(query, community, results)

    print("\n=== Deterministic Result ===")
    print(answer)


if __name__ == "__main__":
    demo()
