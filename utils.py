import os
import torch
DATA_DIR = "/content/data"
CHECKPOINTS_DIR = "/content/checkpoints"
LOG_DIR = "/content/logs"
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)