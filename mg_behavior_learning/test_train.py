#============================================================
'''
Train the Mariogpt
'''

import torch
import argparse

import os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mg_behavior_learning import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer
from mg_behavior_learning.utils import setup_fim_embeddings, view_level, convert_level_to_png, join_list_of_list, characterize
#from test_train_prompter import evaluate_prompts
from transformers import AutoConfig, AutoModelWithLMHead

import random
import numpy as np
import torch

# ======== 固定隨機種子 ========
def set_seed(seed=42):
    random.seed(seed)               # Python
    np.random.seed(seed)            # NumPy
    torch.manual_seed(seed)         # PyTorch (CPU)
    torch.cuda.manual_seed(seed)    # PyTorch (單卡 GPU)
    torch.cuda.manual_seed_all(seed)# PyTorch (多卡 GPU)
    # 保證 cudnn 決定性（會稍微慢一點）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#set_seed(42)  # 你可以改成自己要的數字
random_seed = random.randint(0, 2**32 - 1)  # Generate a random seed
set_seed(random_seed)
print(f"Random seed used: {random_seed}")

# =================================

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MarioGPT')
parser.add_argument('--MODEL_PATH', type=str, default=None, help='Path to model or "random"')
parser.add_argument('--save_prefix', type=int, default=0, help='prefix of path name')
parser.add_argument('--save_iteration', type=int, default=10, help='Save model every N iterations')
parser.add_argument('--eval_iteration', type=int, default=10, help='Evaluate model every N iterations')
parser.add_argument('--save_iteration_sample', action='store_true', help='Save samples at each save iteration')
parser.add_argument('--total_train_steps', type=int, default=21, help='Total training steps')
parser.add_argument('--sleep', type=int, default=5, help='Sleep time before testing')

parser.add_argument('--train_mode_phase_full', action='store_true')
parser.add_argument('--no_train_mode_phase_full', dest='train_mode_phase_full', action='store_false')
parser.set_defaults(train_mode_phase_full=True)

# collapse_mode:
# 0: no_collapse
# 1: collapse_to_allowed
# 2: collapse_to_allowed_and_dash

# masked_loss_mode:
# 0: no_masked_lm_loss
# 1: masked_lm_loss_allowed
# 2: masked_lm_loss_allowed_and_dash

parser.add_argument('--train_mode_expert', type=int, default=0, help='Use prompt augmentation during training')
parser.add_argument('--expert_collapse_mode', type=int, default=0, help='0:no collapse, 1:collapse, 2:collapse_to_allowed_and_dash')
parser.add_argument('--expert_masked_loss_mode', type=int, default=0, help='0: no_masked_lm_loss, 1: masked_lm_loss_allowed, 2: masked_lm_loss_allowed_and_dash')
parser.add_argument('--behavior_labels_path', type=str, default=None)
parser.add_argument('--lambda_beh', type=float, default=0.1)

args = parser.parse_args()

BASE = "distilgpt2"
tokenizer_path = "shyamsn97/Mario-GPT2-700-context-length"

# Set MODEL_PATH based on argument
if args.MODEL_PATH is not None:
    MODEL_PATH = args.MODEL_PATH
else:
    MODEL_PATH = "random"

default_beh = "./Mario-GPT2-700-context-length/behavior_labels.pt"

#config
config = TrainingConfig(
    save_iteration=args.save_iteration,
    eval_iteration=args.eval_iteration,
    #save_prefix=args.save_prefix,
    total_steps=args.total_train_steps,
    batch_size=4,
    #train_mode_phase_full=args.train_mode_phase_full,
    #train_mode_expert = args.train_mode_expert,
    #collapse_mode = args.expert_collapse_mode,
    #masked_loss_mode = args.expert_masked_loss_mode,

    #behavior learning
    lambda_beh=args.lambda_beh,
    num_behaviors=10,
    behavior_labels_path=args.behavior_labels_path or default_beh,
)

# Create directory and copy saved model
train_checkpoint_path = config.output_dir
backup_checkpoint_path = config.output_dir # + '_' + str(config.save_prefix)

print(f'***** Training MarioGPT with checkpoint path: {train_checkpoint_path} *****')
print(f'***** Backup checkpoint path: {backup_checkpoint_path} *****')

# Check if backup_checkpoint_path exists, if so, quit program
if os.path.exists(backup_checkpoint_path):
    print(f"***** Backup path '{backup_checkpoint_path}' already exists. Quitting program.")
    sys.exit(0)

#Make a sample before training
mario_lm = MarioLM(lm_path=MODEL_PATH, tokenizer_path=tokenizer_path)

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
mario_lm = mario_lm.to(device)


"""### Load Dataset (Optional)"""
dataset = MarioDataset(
    mario_lm.tokenizer,
    behavior_labels_path=config.behavior_labels_path,
)
setup_fim_embeddings(mario_lm, dataset)

"""### Setup training"""
trainer = MarioGPTTrainer(mario_lm, dataset, config=config)

# 測試專家配置
trainer.train()

"""### Backup trained models"""
import shutil
os.makedirs(backup_checkpoint_path, exist_ok=True)
for i in range(args.save_iteration, args.total_train_steps + 1, args.save_iteration):
    train_iteration_path = os.path.join(train_checkpoint_path, f"iteration_{i}")
    backup_iteration_path = os.path.join(backup_checkpoint_path, f"iteration_{i}")
    if os.path.exists(train_iteration_path):
        shutil.copytree(train_iteration_path, backup_iteration_path)
        print(f"----- Model {train_iteration_path}/ copied to {backup_iteration_path}/")