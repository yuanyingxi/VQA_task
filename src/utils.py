"""
Utility functions: seed, logger, average meter, timestamp.
"""
import os
import sys
import random
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across torch, numpy, and python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a logger that writes to console and optionally to a file.

    Args:
        name: Logger name.
        log_file: Optional path to a log file.
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Also configure the "src" logger so submodules (src.*) inherit handlers
    src_logger = logging.getLogger("src")
    src_logger.setLevel(level)
    src_logger.handlers.clear()
    src_logger.addHandler(ch)
    if log_file is not None:
        fh_src = logging.FileHandler(log_file, encoding="utf-8")
        fh_src.setLevel(level)
        fh_src.setFormatter(fmt)
        src_logger.addHandler(fh_src)
    src_logger.propagate = False

    return logger


class AverageMeter:
    """
    Keeps track of running average and current value.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def now_str() -> str:
    """
    Return current timestamp string e.g. '20260606_143021'.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
