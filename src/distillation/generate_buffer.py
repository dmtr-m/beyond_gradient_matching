"""
Generate expert trajectories for MTT (Match Training Trajectories).

Run once per task/learner combination before MTT training:
    python src/distillation/generate_buffer.py \
        --task_name sst2 \
        --learner_model bert-base-uncased \
        --buffer_path ./buffers \
        --num_experts 100 \
        --train_epochs 30

Each replay_buffer_N.pt file contains a list of trajectories.
Each trajectory is a list of param-snapshot dicts (one per epoch, epoch 0 = init).
With --classifier_only, only the classifier head params are stored (~6 KB/snapshot
vs ~440 MB for full BERT), making buffers practical on disk.
"""

import argparse
import logging
import os
import random
import sys

# Make src/ importable regardless of where the script is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import DataCollatorWithPadding, set_seed

from dataset_attrs import DATASET_ATTRS
from learner import LearnerConfig, LearnerModel
from utils import configure_optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_classifier_params(learner: LearnerModel) -> dict[str, torch.Tensor]:
    """Return a snapshot of only the classifier head parameters."""
    classifier_names = set(learner.classifier_param_names())
    return {
        name: param.detach().cpu().clone()
        for name, param in learner.named_parameters()
        if name in classifier_names
    }


def get_all_params(learner: LearnerModel) -> dict[str, torch.Tensor]:
    """Return a snapshot of all learner parameters."""
    return {
        name: param.detach().cpu().clone()
        for name, param in learner.named_parameters()
    }


def load_real_dataset(task_name: str, data_dir: str, learner: LearnerModel):
    dataset_attr = DATASET_ATTRS[task_name]
    datasets_path = os.path.join(data_dir, task_name, "datasets")

    if os.path.exists(datasets_path):
        datasets = load_from_disk(datasets_path)
    else:
        datasets = load_dataset(*dataset_attr["load_args"])
        if "validation" not in datasets:
            datasets["validation"] = datasets.pop(dataset_attr["test_split_key"])
        os.makedirs(os.path.dirname(datasets_path), exist_ok=True)
        datasets.save_to_disk(datasets_path)

    datasets = datasets.rename_column(dataset_attr["label_key"], "labels")
    train_dataset = datasets["train"]

    sentence_keys = dataset_attr["sentence_keys"]

    def tokenize_fn(batch):
        sentences = [batch[k] for k in sentence_keys]
        encoded = learner.tokenizer(*sentences, max_length=512, truncation=True)
        encoded["labels"] = batch["labels"]
        return encoded

    train_dataset = train_dataset.map(tokenize_fn, batched=True, desc="Tokenizing")
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    remove_cols = [c for c in train_dataset.column_names if c not in keep_cols]
    train_dataset = train_dataset.remove_columns(remove_cols)

    return train_dataset


def train_one_expert(
    learner: LearnerModel,
    train_dataset,
    train_epochs: int,
    batch_size: int,
    lr: float,
    classifier_only: bool,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    """
    Train the learner from a fresh init for train_epochs epochs.
    Returns a list of param snapshots: [init, after_epoch_1, ..., after_epoch_N].
    """
    learner.init_weights()
    learner.to(device)
    learner.train()

    data_collator = DataCollatorWithPadding(
        learner.tokenizer, padding="longest", pad_to_multiple_of=8
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        drop_last=True,
    )

    num_train_steps = train_epochs * len(loader)
    optimizer, scheduler = configure_optimizer(
        learner,
        lr=lr,
        optimizer_type="adamw",
        scheduler_type="linear",
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_train_steps=num_train_steps,
    )

    snapshot_fn = get_classifier_params if classifier_only else get_all_params
    trajectory = [snapshot_fn(learner)]

    for epoch in trange(train_epochs, leave=False, desc="Expert epochs"):
        learner.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = learner(**batch)
            loss = outputs.loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        trajectory.append(snapshot_fn(learner))

    return trajectory


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build learner
    learner_config = LearnerConfig(
        model_name=args.learner_model,
        use_pretrained_model=True,
        disable_dropout=False,
        gradient_checkpointing=False,
    )
    learner = LearnerModel(learner_config, task_name=args.task_name)

    # Load & tokenise real training data
    logger.info(f"Loading dataset: {args.task_name}")
    train_dataset = load_real_dataset(args.task_name, args.data_dir, learner)
    logger.info(f"Training set size: {len(train_dataset)}")

    # Buffer output directory
    save_dir = os.path.join(args.buffer_path, args.task_name, args.learner_model)
    if args.classifier_only:
        save_dir = os.path.join(save_dir, "classifier_only")
    os.makedirs(save_dir, exist_ok=True)

    # Find next buffer file index
    n = 0
    while os.path.exists(os.path.join(save_dir, f"replay_buffer_{n}.pt")):
        n += 1

    trajectories = []
    for expert_idx in trange(args.num_experts, desc="Experts"):
        logger.info(f"Training expert {expert_idx + 1}/{args.num_experts}")
        trajectory = train_one_expert(
            learner=learner,
            train_dataset=train_dataset,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            classifier_only=args.classifier_only,
            device=device,
        )
        trajectories.append(trajectory)

        if len(trajectories) == args.save_interval:
            save_path = os.path.join(save_dir, f"replay_buffer_{n}.pt")
            logger.info(f"Saving {save_path}")
            torch.save(trajectories, save_path)
            trajectories = []
            n += 1

    if trajectories:
        save_path = os.path.join(save_dir, f"replay_buffer_{n}.pt")
        logger.info(f"Saving {save_path}")
        torch.save(trajectories, save_path)

    logger.info("Done. Buffer saved to: " + save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--learner_model", type=str, default="bert-base-uncased")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--buffer_path", type=str, default="./buffers")
    parser.add_argument("--num_experts", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--classifier_only", action="store_true", default=True)
    parser.add_argument("--no_classifier_only", dest="classifier_only", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
