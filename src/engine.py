import json
import logging
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import AverageMeter
from src.metrics import vqa_accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Args:
        model:     The VQA model.
        loader:    Training DataLoader.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: Optimizer.
        device:    torch.device.
        epoch:     Current epoch number (for logging).
        logger:    Logger instance.
        scheduler: Optional LR scheduler (stepped per batch or per epoch).

    Returns:
        dict with keys "loss" and "acc" (average over epoch).
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        images, questions, q_lengths, targets, _, _, _ = batch
        images = images.to(device)
        questions = questions.to(device)
        q_lengths = q_lengths.to(device)
        targets = targets.to(device)

        logits = model(images, questions, q_lengths)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean().item()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

        if batch_idx % 50 == 0:
            logger.info(
                "Train Epoch %d | Batch %3d/%d | Loss %.4f | Acc %.4f",
                epoch, batch_idx, len(loader), loss_meter.avg, acc_meter.avg,
            )

    elapsed = time.time() - start
    logger.info(
        "Train Epoch %d | Loss %.4f | Acc %.4f | Time %.1fs",
        epoch, loss_meter.avg, acc_meter.avg, elapsed,
    )

    if scheduler is not None:
        scheduler.step()

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    answer_vocab: Dict[int, str],
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Returns both classification accuracy and VQA accuracy (human agreement).

    Args:
        model:        The VQA model.
        loader:       Validation DataLoader.
        criterion:    Loss function.
        device:       torch.device.
        answer_vocab: idx_to_answer mapping (int -> str).
        logger:       Logger instance.

    Returns:
        dict with keys "loss", "acc" (top-1), "vqa_acc".
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    vqa_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            images, questions, q_lengths, targets, _, raw_answers_list, _ = batch
            images = images.to(device)
            questions = questions.to(device)
            q_lengths = q_lengths.to(device)
            targets = targets.to(device)

            logits = model(images, questions, q_lengths)
            loss = criterion(logits, targets)

            preds = logits.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

            # VQA accuracy
            for i in range(images.size(0)):
                pred_answer = answer_vocab.get(preds[i].item(), "unanswerable")
                gt_answers = json.loads(raw_answers_list[i])  # deserialize from JSON string
                vqa_acc = vqa_accuracy(pred_answer, gt_answers) # score = min(1, count_of_pred_in_gt / 3)
                vqa_meter.update(vqa_acc, 1)

    logger.info(
        "Val | Loss %.4f | Acc %.4f | VQA_Acc %.4f",
        loss_meter.avg, acc_meter.avg, vqa_meter.avg,
    )

    return {"loss": loss_meter.avg, "acc": acc_meter.avg, "vqa_acc": vqa_meter.avg}


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    answer_vocab: Dict[int, str],
) -> List[Dict]:
    """
    Generate predictions for all samples in loader, used in recording details.

    Args:
        model:        The VQA model.
        loader:       DataLoader (test or val).
        device:       torch.device.
        answer_vocab: idx_to_answer mapping.

    Returns:
        List of dicts, each with:
          - "image": str (image id)
          - "answer": str (predicted answer)
          - "answerable_conf": float (softmax probability of "unanswerable")
    """
    model.eval()
    predictions: List[Dict] = []
    # find unanswerable index if present
    unans_idx: Optional[int] = None
    for idx, ans in answer_vocab.items():
        if ans == "unanswerable":
            unans_idx = idx
            break

    with torch.no_grad():
        for batch in loader:
            images, questions, q_lengths, _, _, _, image_ids = batch
            images = images.to(device)
            questions = questions.to(device)
            q_lengths = q_lengths.to(device)

            logits = model(images, questions, q_lengths)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            for i in range(images.size(0)):
                pred_answer = answer_vocab.get(preds[i].item(), "unanswerable")
                ans_conf = probs[i, unans_idx].item() if unans_idx is not None else 0.0
                predictions.append({
                    "image": image_ids[i],
                    "answer": pred_answer,
                    "answerable_conf": ans_conf,
                })

    return predictions
