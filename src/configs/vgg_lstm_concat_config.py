import os
from pathlib import Path

# TODO: Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "VizWiz-VQA" / "raw"

ANNOT_DIR = DATA_ROOT / "Annotations"
TRAIN_ANN  = str(ANNOT_DIR / "train.json")
VAL_ANN    = str(ANNOT_DIR / "val.json")
TEST_ANN   = str(ANNOT_DIR / "test.json")

TRAIN_IMG_ROOT = str(DATA_ROOT / "train" / "train")
VAL_IMG_ROOT   = str(DATA_ROOT / "val"   / "val")
TEST_IMG_ROOT  = str(DATA_ROOT / "test"  / "test")

# TODO: Model name. Default model is vgg_lstm_concat which is baseline.
MODEL_NAME = os.environ.get("VQA_MODEL_NAME", "vgg_lstm_concat")

CKPT_DIR   = str(PROJECT_ROOT / "checkpoints" / MODEL_NAME)
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / MODEL_NAME)
LOG_DIR    = str(PROJECT_ROOT / "logs" / MODEL_NAME)
BEST_CKPT_PATH = str(Path(CKPT_DIR) / "best.pt")

# TODO: Data / Vocabulary 
MAX_QUSETION_VOCAB = 10000    # max question word vocabulary size
MAX_QUESTION       = 30       # max question length
ANSWER_TOP_K       = 1000     # keep top-K most frequent answers
PAD_TOKEN          = "<pad>"  # fill in the blank
UNK_TOKEN          = "<unk>"  # rare words will be replaced by <unk>

# TODO: Model 
IMG_ENCODER   = "vgg16"          # backbone name
IMG_FEAT_DIM  = 4096             # VGG16 fc7 output dim
EMBED_DIM     = 300              # word embedding dimension
LSTM_HIDDEN   = 1024             # LSTM hidden size
FUSION_DIM    = 5120             # 4096 + 1024, in concatenating
MLP_HIDDEN    = 1024             # fusion MLP hidden
DROPOUT       = 0.5

# TODO: Training
SEED          = 42
BATCH_SIZE    = 64
LR            = 1e-4
WEIGHT_DECAY  = 0.0              # L2 regularization coefficient
EPOCHS        = 30
STEP_SIZE     = 10               # stepwise learning rate decay 
GAMMA         = 0.5              # every STEP_SIZE epochs, the learning rate should be multiplied by GAMMA
NUM_WORKERS   = 2                # data loader workers

# TODO: VGG16 fine-tune control
# only freezes VGG conv layers; fc6/fc7 always trainable
FREEZE_CONV   = True             # freeze conv blocks, only train classifier

# TODO: Misc
DEVICE        = "cuda"            # will fallback to cpu if unavailable
PIN_MEMORY    = True

# TODO: Early stopping
PATIENCE      = 3                 # stop after N epochs without val VQA improvement