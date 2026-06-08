from typing import Dict, List

import torch

# Global SBERT model cache (lazy-loaded in sbert_similarity)
_SBERT_MODEL = None


def vqa_accuracy(
    pred_answer: str,
    ground_truth_answers: List[Dict[str, str]],
) -> float:
    """
    Compute VizWiz VQA accuracy for a single prediction.

    The formula: min(1, count_of_pred_in_gt / 3)

    Only ground-truth answers with answer_confidence != "no" are counted.

    Args:
        pred_answer: Predicted answer string.
        ground_truth_answers: List of 10 dicts with keys "answer" and "answer_confidence" (values: "yes"/"maybe"/"no").

    Returns:
        Accuracy score in [0.0, 1.0].
    """
    # Filter out "no" confidence answers
    valid = [a for a in ground_truth_answers if a.get("answer_confidence", "yes") != "no"]
    if not valid:
        return 0.0

    count = sum(1 for a in valid if a["answer"] == pred_answer)
    return min(1.0, count / 3.0)


def sbert_similarity(
    pred_answer: str,
    ground_truth_answers: List[Dict[str, str]],
) -> float:
    """
    Compute SBERT semantic similarity between prediction and ground truth answers.

    Uses `all-MiniLM-L6-v2` (384-dim sentence embeddings, fast on GPU).
    Follows the same evaluation schema as vqa_accuracy:
    1. Filter out "no" confidence answers
    2. Encode pred and all valid gt answers
    3. Return mean of top-3 cosine similarities (analogous to min(1, count/3))

    Args:
        pred_answer: Predicted answer string.
        ground_truth_answers: List of 10 dicts with keys "answer" and "answer_confidence".

    Returns:
        Similarity score in [0.0, 1.0].
    """
    # Lazy-import inside function to avoid import failure breaking metrics.py
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for SBERT similarity. "
            "Install: pip install sentence-transformers"
        )

    # Global singleton for SBERT model (lazy-loaded)
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    # Filter out "no" confidence answers (same as vqa_accuracy)
    valid = [a for a in ground_truth_answers if a.get("answer_confidence", "yes") != "no"]
    if not valid:
        return 0.0

    gt_texts = [a["answer"] for a in valid]

    # Encode all texts
    pred_emb = _SBERT_MODEL.encode(pred_answer, convert_to_tensor=True)
    gt_embs = _SBERT_MODEL.encode(gt_texts, convert_to_tensor=True)

    # Cosine similarity = normalize -> dot product
    pred_norm = torch.nn.functional.normalize(pred_emb.unsqueeze(0), p=2, dim=1)
    gt_norm = torch.nn.functional.normalize(gt_embs, p=2, dim=1)
    similarities = (pred_norm @ gt_norm.T).squeeze(0).cpu().numpy()

    # Average top-3 similarities (analogous to VQA accuracy's min(1, count/3) schema)
    top_3 = sorted(similarities, reverse=True)[:3]
    return float(np.mean(top_3))