from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from typing import Any, Dict

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data_handlers.codealltag_data_handler import CodealltagDataHandler
from lightning_module_for_seq_to_seq_lm import LightningModuleForSeq2SeqLM
from lightning_module_for_seq_to_seq_lm import Seq2SeqDataset
from utils.project_utils import ProjectUtils

import os
import random
import warnings

import numpy as np
import pytorch_lightning as pl
import torch


warnings.filterwarnings(
    "ignore", 
    message=r".*torch\.cuda\.amp\.GradScaler.*"
)
warnings.filterwarnings(
    "ignore", 
    message=r"No device id is provided via `init_process_group`.*"
)

def fine_tune():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_checkpoints_root_dir = os.environ.get("MODEL_CHECKPOINTS_ROOT_DIR", None)
    model_checkpoints_root_dir = Path(model_checkpoints_root_dir) if model_checkpoints_root_dir else Path.home() / "model_checkpoints"

    huggingface_cache_dir = os.environ.get("HF_HOME", None)

    data_dir = os.environ.get("DATA_DIR", None)
    data_dir = Path(data_dir) if data_dir else None

    data_fold_k_value = os.environ.get("DATA_FOLD_K_VALUE", None)
    data_fold_k_value = int(data_fold_k_value) if data_fold_k_value else 1

    use_multi_gpu = os.environ.get("USE_MULTI_GPU", None)
    use_multi_gpu = int(use_multi_gpu) if use_multi_gpu else 0
    use_multi_gpu = bool(use_multi_gpu) if use_multi_gpu else False

    log_to_wandb = os.environ.get("LOG_TO_WANDB", None)
    log_to_wandb = int(log_to_wandb) if log_to_wandb else 0
    log_to_wandb = bool(log_to_wandb) if log_to_wandb else False

    if log_to_wandb:
        wandb_entity = os.environ.get("WANDB_ENTITY", "sksdotsauravs-dev")

    transformer_model_name = os.environ.get("TRANSFORMER_MODEL_NAME", "google/mt5-base")

    learning_rate = os.environ.get("LEARNING_RATE", None)
    learning_rate = float(learning_rate) if learning_rate else 5e-5
    
    max_epochs = os.environ.get("MAX_EPOCHS", None)
    max_epochs = int(max_epochs) if max_epochs else 10
    
    mini_batch_size = os.environ.get("MINI_BATCH_SIZE", None)
    mini_batch_size = int(mini_batch_size) if mini_batch_size else 4
    
    project_root: Path = ProjectUtils.get_project_root()
    data_handler = CodealltagDataHandler(project_root, data_dir=data_dir)
    datasetdict = data_handler.get_train_dev_test_datasetdict(k=data_fold_k_value)
    
    label_order = ['MALE', 'FAMILY', 'ORG', 'CITY', 'DATE', 'URL', 'EMAIL', 
                   'FEMALE', 'UFID', 'PHONE', 'USER', 'STREET', 'STREETNO', 'ZIP']
    fold_stats = data_handler.get_fold_stats(datasetdict, label_order)
    
    sample_size = (
        len(datasetdict["train"]) + 
        len(datasetdict["dev"]) + 
        len(datasetdict["test"])
    )

    print(f"model_checkpoints_root_dir: {model_checkpoints_root_dir}")
    print(f"data_dir: {data_dir}")
    print(f"data_fold_k_value: {data_fold_k_value}")
    print(f"use_multi_gpu: {use_multi_gpu}")
    print(f"log_to_wandb: {log_to_wandb}")
    if log_to_wandb:
        print(f"wandb_entity: {wandb_entity}")
    print(f"transformer_model_name: {transformer_model_name}")
    print(f"learning_rate: {learning_rate}")
    print(f"max_epochs: {max_epochs}")
    print(f"mini_batch_size: {mini_batch_size}")
    print(f"sample_size: {sample_size}")

    model_dir_name = transformer_model_name.replace("/", "--").replace("_", "-")
    
    data_dir_path = model_checkpoints_root_dir / "codealltag" / "pg" / model_dir_name
    data_dir_path =  data_dir_path / f"sample-size-{sample_size}"
    data_dir_path.mkdir(parents=True, exist_ok=True)

    model_dir_path = data_dir_path / f"learning-rate-{learning_rate:.0e}".replace('e-0', 'e-')
    model_dir_path = model_dir_path / f"max-epochs-{max_epochs}"
    model_dir_path = model_dir_path / f"mini-batch-size-{mini_batch_size}"
    model_dir_path.mkdir(parents=True, exist_ok=True)

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    
    module_config: Dict[str, Any] = {
        "model_name_or_path": transformer_model_name,
        "huggingface_cache_dir": huggingface_cache_dir,
        "model_checkpoint_dir": model_dir_path
    }

    lightning_module = LightningModuleForSeq2SeqLM(module_config)

    train_dataset = Seq2SeqDataset(datasetdict["train"].to_pandas(), lightning_module.tokenizer)
    val_dataset = Seq2SeqDataset(datasetdict["dev"].to_pandas(), lightning_module.tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=mini_batch_size, num_workers=2)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:02d}-{step:05d}-{val_loss:.4f}", 
        monitor="val_loss", 
        mode="min", 
        save_top_k=1
    )

    wandb_logger = None
    if log_to_wandb:
        wandb_logger = WandbLogger(
            project=project_root.name,
            entity=wandb_entity,
            name = f"codealltag-pg__{model_dir_name}__fold-{data_fold_k_value}", 
            save_dir=str(model_dir_path / "wandb"),
            config={
                "transformer_model_name": transformer_model_name, 
                "data_fold": data_fold_k_value, 
                "learning_rate": learning_rate, 
                "max_epochs": max_epochs, 
                "mini_batch_size": mini_batch_size, 
                "sample_size": sample_size,
                "fold_stats": fold_stats,
                "model_checkpoint_dir": str(model_dir_path)
            },
            log_model=False
        )

    class OverrideEpochStepCallback(pl.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            self._log_step_as_current_epoch(trainer, pl_module)

        def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            self._log_step_as_current_epoch(trainer, pl_module)

        def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            pl_module.log("step", trainer.current_epoch)
    

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_epochs,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        precision=32,
        callbacks=[OverrideEpochStepCallback(), checkpoint_callback],
        logger=wandb_logger if log_to_wandb else False
    )

    trainer.fit(
        model=lightning_module, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    fine_tune()