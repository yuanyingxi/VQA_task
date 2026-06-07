"""
Usage:
    python -m scripts.train [--epochs N] [--batch_size N] [--lr F] ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.configs.vgg_lstm_concat_config import (
    TRAIN_ANN, VAL_ANN, TEST_ANN,
    TRAIN_IMG_ROOT, VAL_IMG_ROOT, TEST_IMG_ROOT,
    CKPT_DIR, OUTPUT_DIR, LOG_DIR,
    SEED, BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS,
    STEP_SIZE, GAMMA, NUM_WORKERS, PIN_MEMORY,
    ANSWER_TOP_K, MAX_QUSETION_VOCAB, MAX_QUESTION,
    PATIENCE
)
import src.configs.vgg_lstm_concat_config as cfg
from src.utils import set_seed, get_logger, now_str
from src.dataset import (
    build_answer_vocab,
    build_question_vocab,
    VizWizVQADataset,
)
from src.models.vgg_lstm_concat import VGG_LSTM_Concat
from src.engine import train_one_epoch, validate, predict


def parse_args() -> argparse.Namespace:
    """
    adjust hyperparameters dynamically through terminal commands
    """
    parser = argparse.ArgumentParser(description="VizWiz-VQA Training")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name (subdirectory under checkpoints/outputs/logs; "
                             "default: {MODEL_NAME}, env: VQA_MODEL_NAME)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--top_k_answers", type=int, default=ANSWER_TOP_K, help="Top-K answer vocab")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Checkpoint save dir (default: checkpoints/<model_name>)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples for debug")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit val samples for debug")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (default: logs/<model_name>)")
    parser.add_argument("--early_stop", action="store_true",
                        help="Stop training when val VQA doesn't improve for --patience epochs")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early-stopping patience (epochs with no improvement)")
    return parser.parse_args()


def main() -> None:
    global CKPT_DIR, OUTPUT_DIR, LOG_DIR
    args = parse_args()

    # ── Model name override ───────────────────────────────────────────
    if args.model_name is not None:
        cfg.MODEL_NAME = args.model_name

    # Recompute paths (in case MODEL_NAME changed via CLI or env var)
    cfg.CKPT_DIR   = str(Path(cfg.PROJECT_ROOT) / "checkpoints" / cfg.MODEL_NAME)
    cfg.OUTPUT_DIR = str(Path(cfg.PROJECT_ROOT) / "outputs" / cfg.MODEL_NAME)
    cfg.LOG_DIR    = str(Path(cfg.PROJECT_ROOT) / "logs" / cfg.MODEL_NAME)

    # Sync module-level names so downstream code sees updated paths
    CKPT_DIR   = cfg.CKPT_DIR
    OUTPUT_DIR = cfg.OUTPUT_DIR
    LOG_DIR    = cfg.LOG_DIR

    # Resolve argparse defaults (were captured at module import time)
    args.save_dir = args.save_dir or cfg.CKPT_DIR
    args.log_dir  = args.log_dir or cfg.LOG_DIR

    ts = now_str()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Logger
    log_file = os.path.join(args.log_dir, f"train_{ts}.log")
    logger = get_logger("train", log_file=log_file)

    # Seed
    set_seed(args.seed)
    logger.info("Random seed set to %d", args.seed)
    logger.info("Arguments: %s", vars(args))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Load annotations ──────────────────────────────────────────────
    logger.info("Loading annotations...")
    with open(TRAIN_ANN, "r", encoding="utf-8") as f:
        train_ann = json.load(f)
    with open(VAL_ANN, "r", encoding="utf-8") as f:
        val_ann = json.load(f)

    if args.max_train_samples:
        train_ann = train_ann[: args.max_train_samples]
    if args.max_val_samples:
        val_ann = val_ann[: args.max_val_samples]

    logger.info("Train samples: %d", len(train_ann))
    logger.info("Val samples:   %d", len(val_ann))

    # ── Build vocabularies ────────────────────────────────────────────
    logger.info("Building answer vocabulary (top_k=%d)...", args.top_k_answers)
    answer_to_idx, idx_to_answer = build_answer_vocab(
        train_ann + val_ann, top_k=args.top_k_answers,
    )
    num_answers = len(answer_to_idx)
    logger.info("Answer vocab size: %d", num_answers)
    logger.info("Answer vocab sample: %s", list(answer_to_idx.keys())[:20])

    # Save answer vocab for later use by test.py
    vocab_dir = args.save_dir
    os.makedirs(vocab_dir, exist_ok=True)
    with open(os.path.join(vocab_dir, "answer_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"answer_to_idx": answer_to_idx, "idx_to_answer": {int(k): v for k, v in idx_to_answer.items()}}, f)
    logger.info("Answer vocab saved to %s", os.path.join(vocab_dir, "answer_vocab.json"))

    logger.info("Building question vocabulary...")
    word_to_idx, _, vocab_size = build_question_vocab(
        train_ann + val_ann, max_vocab=MAX_QUSETION_VOCAB,
    )
    logger.info("Question vocab size: %d", vocab_size)

    # Save question vocab
    with open(os.path.join(vocab_dir, "question_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(word_to_idx, f)
    logger.info("Question vocab saved")

    # ── Datasets & DataLoaders ────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VizWizVQADataset(
        ann_path=TRAIN_ANN,
        img_root=TRAIN_IMG_ROOT,
        answer_to_idx=answer_to_idx,
        word_to_idx=word_to_idx,
        transform=transform,
        max_samples=args.max_train_samples,
    )
    val_dataset = VizWizVQADataset(
        ann_path=VAL_ANN,
        img_root=VAL_IMG_ROOT,
        answer_to_idx=answer_to_idx,
        word_to_idx=word_to_idx,
        transform=transform,
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = VGG_LSTM_Concat(
        vocab_size=vocab_size,
        num_answers=num_answers,
    )
    model.to(device)
    logger.info("Model parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    logger.info("Trainable parameters: %.2fM",
                sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

    # ── Optimizer, loss, scheduler ────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # ── Training loop ─────────────────────────────────────────────────
    best_vqa_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0  # early stopping counter

    logger.info("=" * 60)
    logger.info("Starting training for %d epochs", args.epochs)
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
        )
        val_metrics = validate(
            model, val_loader, criterion, device, idx_to_answer, logger,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %2d | Train Loss %.4f | Train Acc %.4f | Val Loss %.4f | Val Acc %.4f | Val VQA %.4f | LR %.6f",
            epoch,
            train_metrics["loss"], train_metrics["acc"],
            val_metrics["loss"], val_metrics["acc"], val_metrics["vqa_acc"],
            current_lr,
        )

        # Save best model (model only, no optimizer state to save disk space)
        if val_metrics["vqa_acc"] > best_vqa_acc:
            best_vqa_acc = val_metrics["vqa_acc"]
            best_epoch = epoch
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_vqa_acc": best_vqa_acc,
                "args": vars(args),
            }, best_path)
            logger.info("New best model saved to %s (VQA Acc: %.4f)", best_path, best_vqa_acc)
            epochs_no_improve = 0
        else:
            # early stopping
            epochs_no_improve += 1
            if args.early_stop and epochs_no_improve >= args.patience:
                logger.info(
                    "Early stopping at epoch %d: no improvement for %d epochs "
                    "(best VQA %.4f at epoch %d)",
                    epoch, epochs_no_improve, best_vqa_acc, best_epoch
                )
                break

    logger.info("=" * 60)
    logger.info("Training finished. Best VQA Acc: %.4f at epoch %d", best_vqa_acc, best_epoch)


if __name__ == "__main__":
    main()
