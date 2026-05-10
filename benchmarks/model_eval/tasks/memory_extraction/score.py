"""Memory extraction scoring.

Looser than entity F1 because content paraphrasing is expected. We score:
- Type accuracy: per-item, did the model assign the right memory type?
- Coverage: percentage of ground-truth items that have a corresponding
  predicted item with semantic match (cosine similarity > 0.6 on content)
- Hallucination rate: predicted items with no semantic match in truth
"""
from __future__ import annotations

import json

from ...metrics import label_similarity


def _parse(predicted_text: str) -> list[dict]:
    try:
        data = json.loads(predicted_text)
    except json.JSONDecodeError:
        return []
    items = data.get("memories", []) if isinstance(data, dict) else []
    return [m for m in items if isinstance(m, dict)]


def score(
    predicted_text: str,
    ground_truth: list[dict],
    similarity_threshold: float = 0.6,
    embed_model: str = "nomic-embed-text",
    endpoint: str = "http://localhost:11434",
) -> dict:
    pred = _parse(predicted_text)
    truth = [m for m in ground_truth if isinstance(m, dict)]

    if not truth:
        return {
            "coverage": 0.0,
            "hallucination_rate": 1.0 if pred else 0.0,
            "type_accuracy": 0.0,
            "predicted_count": len(pred),
            "truth_count": 0,
            "valid_json": _is_valid(predicted_text),
        }

    matched_truth_indices: set[int] = set()
    matched_pred_indices: set[int] = set()
    type_correct = 0
    type_total = 0

    for pi, p in enumerate(pred):
        p_content = p.get("content", "")
        p_type = p.get("type", "").strip().lower()
        if not isinstance(p_content, str) or not p_content:
            continue
        best_sim = 0.0
        best_ti = None
        for ti, t in enumerate(truth):
            if ti in matched_truth_indices:
                continue
            t_content = t.get("content", "")
            if not isinstance(t_content, str):
                continue
            sim = label_similarity(p_content, t_content, embed_model=embed_model, endpoint=endpoint)
            if sim > best_sim:
                best_sim = sim
                best_ti = ti
        if best_ti is not None and best_sim >= similarity_threshold:
            matched_truth_indices.add(best_ti)
            matched_pred_indices.add(pi)
            type_total += 1
            t_type = truth[best_ti].get("type", "").strip().lower()
            if p_type == t_type:
                type_correct += 1

    coverage = len(matched_truth_indices) / len(truth)
    hallucination_rate = (len(pred) - len(matched_pred_indices)) / max(1, len(pred))
    type_accuracy = type_correct / type_total if type_total else 0.0

    return {
        "coverage": coverage,
        "hallucination_rate": hallucination_rate,
        "type_accuracy": type_accuracy,
        "predicted_count": len(pred),
        "truth_count": len(truth),
        "matched_count": len(matched_pred_indices),
        "valid_json": _is_valid(predicted_text),
    }


def _is_valid(text: str) -> bool:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and isinstance(data.get("memories", None), list)
