import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.configs.vgg_lstm_concat_config import (
    TRAIN_ANN, VAL_ANN, TEST_ANN,
    TRAIN_IMG_ROOT, VAL_IMG_ROOT, TEST_IMG_ROOT,
    MAX_QUSETION_VOCAB, MAX_QUESTION, ANSWER_TOP_K,
    PAD_TOKEN, UNK_TOKEN,
)

# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Lower-case, split on non-alphanumeric (keep a-z, 0-9, ').
    """
    return re.findall(r"[a-z0-9']+", text.lower())


def build_question_vocab(
    annotations: List[dict],
    max_vocab: int = MAX_QUSETION_VOCAB,
) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Build question word vocabulary from a list of annotation records.

    Returns:
        word_to_idx: mapping word -> index (0=pad, 1=unk)
        idx_to_word: mapping index -> word
        vocab_size: number of words in vocabulary
    """
    counter: Counter = Counter()
    for ann in annotations:
        tokens = tokenize(ann["question"])
        counter.update(tokens)

    # keep top max_vocab-2 to leave room for pad/unk
    most_common = counter.most_common(max_vocab - 2)
    word_to_idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in most_common:
        word_to_idx[word] = len(word_to_idx)

    idx_to_word: Dict[int, str] = {v: k for k, v in word_to_idx.items()}
    return word_to_idx, idx_to_word, len(word_to_idx)


def encode_question(
    question: str,
    word_to_idx: Dict[str, int],
    max_len: int = MAX_QUESTION,
) -> Tuple[torch.LongTensor, int]:
    """
    Tokenize, encode, and pad/truncate a question.

    Returns:
        indices: LongTensor of shape (max_len,) with pad token filling tail.
        length:  original token count (before padding).
    """
    tokens = tokenize(question)
    unk_id = word_to_idx.get(UNK_TOKEN, 1)
    ids = [word_to_idx.get(t, unk_id) for t in tokens[:max_len]]
    length = len(ids)
    # pad
    pad_id = word_to_idx.get(PAD_TOKEN, 0)
    ids += [pad_id] * (max_len - len(ids))
    return torch.LongTensor(ids), length


# ---------------------------------------------------------------------------
# Answer vocabulary
# ---------------------------------------------------------------------------

def build_answer_vocab(
    annotations: List[dict],
    top_k: int = ANSWER_TOP_K,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build answer vocabulary from train+val annotations.

    "unanswerable", "yes", "no" are forcibly kept in the vocabulary even
    if they fall outside top-k.

    Returns:
        answer_to_idx: mapping answer string -> class index
        idx_to_answer: mapping class index -> answer string
    """
    counter: Counter = Counter()
    for ann in annotations:
        for a in ann["answers"]:
            counter[a["answer"]] += 1

    # forced words
    forced = {"unanswerable", "yes", "no"}
    # ensure they are counted so they make top-k naturally if possible
    # but we will also force-add them later

    # get top-k candidates, excluding forced ones to keep slots
    candidates = [(w, c) for w, c in counter.most_common() if w not in forced]
    top_candidates = candidates[:top_k]

    answer_to_idx: Dict[str, int] = {}
    # forced words first
    for w in sorted(forced):
        if w not in answer_to_idx:
            answer_to_idx[w] = len(answer_to_idx)
    # then top-k
    for w, _ in top_candidates:
        if w not in answer_to_idx:
            answer_to_idx[w] = len(answer_to_idx)

    idx_to_answer: Dict[int, str] = {v: k for k, v in answer_to_idx.items()}
    return answer_to_idx, idx_to_answer


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def select_target_answer(answers: List[dict]) -> str:
    """
    Hard labels: pick the most frequent answer from 10 ground-truth answers.

    Ties are broken by preferring answers with higher answer_confidence
    (yes > maybe > no). If still tied, the alphabetically first answer wins.
    """
    # Count frequency first
    freq: Dict[str, int] = {}
    for a in answers:
        freq[a["answer"]] = freq.get(a["answer"], 0) + 1

    max_count = max(freq.values())
    candidates = [ans for ans, cnt in freq.items() if cnt == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break: prefer answers with more "yes" confidence
    def confidence_score(answer: str) -> int:
        score = 0
        for a in answers:
            if a["answer"] == answer:
                if a["answer_confidence"] == "yes":
                    score += 3
                elif a["answer_confidence"] == "maybe":
                    score += 1
                # "no" gives 0
        return score

    candidates.sort(key=lambda x: (confidence_score(x), x), reverse=True)
    return candidates[0]

def build_soft_target(answer: List[dict], answer_vocab_size) -> str:
    """
    Soft labels: 
    """
    pass


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VizWizVQADataset(Dataset):
    """
    PyTorch Dataset for VizWiz-VQA.

    Each item returns:
        image_tensor:       (3, 224, 224) float tensor, normalized
        question_indices:   (max_len,) long tensor, padded
        question_length:    int, original token count
        target_idx:         int, class index of the most-frequent answer
        answerable:         int, 0/1 flag
        raw_answers:        list of dict, the original 10 answers
        image_id:           str, image filename stem
    """

    def __init__(
        self,
        ann_path: str,
        img_root: str,
        answer_to_idx: Dict[str, int],
        word_to_idx: Dict[str, int],
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            ann_path:       Path to JSON annotation file.
            img_root:       Root directory containing images.
            answer_to_idx:  Answer string -> class index mapping.
            word_to_idx:    Word -> index mapping for question tokenizer.
            transform:      Optional torchvision transform (e.g. ToTensor, Normalize).
            max_samples:    If set, limit to first N records (for debugging).
        """
        super().__init__()
        with open(ann_path, "r", encoding="utf-8") as f:
            self.annotations: List[dict] = json.load(f)

        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]

        self.img_root = img_root
        self.answer_to_idx = answer_to_idx
        self.word_to_idx = word_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple:
        ann = self.annotations[index]

        # --- image ---
        img_path = os.path.join(self.img_root, ann["image"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # --- question ---
        q_indices, q_length = encode_question(ann["question"], self.word_to_idx)

        # --- target ---
        target_str = select_target_answer(ann["answers"])
        target_idx = self.answer_to_idx.get(target_str, 0)  # default to first class
        answerable = ann.get("answerable", 0)

        # --- image id ---
        image_id = os.path.splitext(ann["image"])[0]

        # Serialize raw answers as JSON string to prevent DataLoader
        # default_collate from corrupting the nested list-of-dicts structure
        raw_answers_str = json.dumps(ann["answers"])

        return (
            image,
            q_indices,
            q_length,
            target_idx,
            answerable,
            raw_answers_str,
            image_id,
        )
