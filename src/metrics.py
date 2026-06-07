from typing import Dict, List


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