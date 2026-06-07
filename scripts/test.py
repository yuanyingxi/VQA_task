"""
Usage:
    python -m scripts.test --ckpt checkpoints/best.pt [--split val] [--batch_size 64]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.configs.vgg_lstm_concat_config import (
    VAL_ANN, TEST_ANN,
    VAL_IMG_ROOT, TEST_IMG_ROOT,
    CKPT_DIR, OUTPUT_DIR,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    MAX_QUSETION_VOCAB,
)
import src.configs.vgg_lstm_concat_config as cfg
from src.utils import get_logger, AverageMeter
from src.dataset import VizWizVQADataset, build_question_vocab
from src.models.vgg_lstm_concat import VGG_LSTM_Concat
from src.engine import predict
from src.metrics import vqa_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VizWiz-VQA Test / Eval")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (subdirectory under checkpoints/outputs/logs; "
                             "default: src.config.MODEL_NAME, env: VQA_MODEL_NAME)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Which split to evaluate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory with vocab files (default: checkpoints/<model_name>)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Model name override ───────────────────────────────────────────
    if args.model_name is not None:
        cfg.MODEL_NAME = args.model_name

    # Recompute paths (in case MODEL_NAME changed via CLI or env var)
    cfg.CKPT_DIR   = str(Path(cfg.PROJECT_ROOT) / "checkpoints" / cfg.MODEL_NAME)
    cfg.OUTPUT_DIR = str(Path(cfg.PROJECT_ROOT) / "outputs" / cfg.MODEL_NAME)

    # Resolve argparse defaults (were captured at module import time)
    args.save_dir = args.save_dir or cfg.CKPT_DIR

    logger = get_logger("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Load checkpoint ───────────────────────────────────────────────
    logger.info("Loading checkpoint from %s", args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location="cpu")

    # ── Load vocabs ───────────────────────────────────────────────────
    save_dir = args.save_dir
    with open(os.path.join(save_dir, "answer_vocab.json"), "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    answer_to_idx = vocab_data["answer_to_idx"]
    idx_to_answer = {int(k): v for k, v in vocab_data["idx_to_answer"].items()}
    num_answers = len(answer_to_idx)
    logger.info("Answer vocab size: %d", num_answers)

    with open(os.path.join(save_dir, "question_vocab.json"), "r", encoding="utf-8") as f:
        word_to_idx = json.load(f)
    vocab_size = len(word_to_idx)
    logger.info("Question vocab size: %d", vocab_size)

    # ── Dataset & DataLoader ──────────────────────────────────────────
    if args.split == "val":
        ann_path = VAL_ANN
        img_root = VAL_IMG_ROOT
    else:
        ann_path = TEST_ANN
        img_root = TEST_IMG_ROOT

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VizWizVQADataset(
        ann_path=ann_path,
        img_root=img_root,
        answer_to_idx=answer_to_idx,
        word_to_idx=word_to_idx,
        transform=transform,
        max_samples=None,
    )
    logger.info("Loaded %s split with %d samples", args.split, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = VGG_LSTM_Concat(
        vocab_size=vocab_size,
        num_answers=num_answers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded from epoch %d", checkpoint.get("epoch", "?"))

    # ── Predict ───────────────────────────────────────────────────────
    predictions = predict(model, loader, device, idx_to_answer)

    # Save predictions
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(cfg.OUTPUT_DIR, f"predictions_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Predictions saved to %s (%d entries)", out_path, len(predictions))

    # ── Evaluate (if answers available) ───────────────────────────────
    # Load annotations to get ground truth answers
    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # Build a mapping: image_id -> ground_truth_answers
    gt_map = {}
    for ann in annotations:
        img_id = os.path.splitext(ann["image"])[0]
        gt_map[img_id] = ann["answers"]

    vqa_meter = AverageMeter()
    for pred in predictions:
        img_id = pred["image"]
        gt_answers = gt_map.get(img_id, [])
        if gt_answers:
            score = vqa_accuracy(pred["answer"], gt_answers)
            vqa_meter.update(score, 1)

    logger.info(
        "%s VQA Accuracy over %d samples: %.4f",
        args.split, vqa_meter.count, vqa_meter.avg,
    )


if __name__ == "__main__":
    main()
