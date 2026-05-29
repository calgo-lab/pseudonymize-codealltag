from typing import Any, Dict

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer
)
from transformers.optimization import (
    Adafactor, 
    AdafactorSchedule
)
from torch.utils.data import Dataset

import pytorch_lightning as pl


class LightningModuleForSeq2SeqLM(pl.LightningModule):

    def __init__(self, module_config: Dict[str, Any]):

        super(LightningModuleForSeq2SeqLM, self).__init__()

        self._module_config = module_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._module_config["model_name_or_path"],
            cache_dir=self._module_config["huggingface_cache_dir"]
        )

        # self.tokenizer.model_max_length = 2048
        tokenizer_config = self._module_config.get("tokenizer_config", dict())
        for key, value in tokenizer_config.items():
            if hasattr(self.tokenizer, key):
                setattr(self.tokenizer, key, value)
        
        model_config = AutoConfig.from_pretrained(self._module_config["model_name_or_path"])
        # model_config.n_positions = 2048
        # model_config.max_position_embeddings = 2048
        # model_config.relative_attention_num_buckets = 32

        for key, value in self._module_config.get("model_config", {}).items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        model_config.save_pretrained(self._module_config["model_checkpoint_dir"])
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self._module_config["model_name_or_path"],
            config=model_config,
            cache_dir=self._module_config["huggingface_cache_dir"]
        )
        self.save_hyperparameters()
    
    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def _step(self, batch):
        lm_labels = batch["decoder_input_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['decoder_attention_mask']
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adafactor(self.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)
        return [optimizer], [lr_scheduler]

    # def train_dataloader(self):
    #     return DataLoader(train_dataset, batch_size=self.config["train_batch_size"], shuffle=True, num_workers=2)

    # def val_dataloader(self):
    #     return DataLoader(val_dataset, batch_size=self.config["eval_batch_size"], num_workers=2)

class Seq2SeqDataset(Dataset):

    def __init__(self, input_df, tokenizer, max_length=512):

        self.input_df = input_df
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.tokenizer.max_length = max_length
        self.tokenizer.model_max_length = max_length

    def __len__(self):
        return len(self.input_df)

    def __getitem__(self, idx):
        input_text = self.input_df.iloc[idx]["input_text"]
        output_text = self.input_df.iloc[idx]["output_text"]

        input_encoded = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        output_encoded = self.tokenizer.batch_encode_plus(
            [output_text],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_encoded["input_ids"].squeeze(),
            "attention_mask": input_encoded["attention_mask"].squeeze(),
            "decoder_input_ids": output_encoded["input_ids"].squeeze(),
            "decoder_attention_mask": output_encoded["attention_mask"].squeeze()
        }