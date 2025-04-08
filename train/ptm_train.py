import logging
import numpy as np
import os
import glob
import sys
import math
import json
from dataclasses import dataclass, field
# from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
# from pathlib import Path
import datasets
import torch
import torch.nn as nn
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import LambdaLR
# from datasets import load_dataset, concatenate_datasets, Dataset
from datetime import datetime, timezone
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed
)
from transformers.utils.versions import require_version
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .configuration_llm import llmConfig
from .modeling_llm import llmForCausalLM
from 