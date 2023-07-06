import os
import re
import math
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Any, List, Tuple
from torch.utils.data import Dataset
from nltk import edit_distance
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import VisionEncoderDecoderConfig
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig


class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
            self,
            data_set,
            # dataset_name_or_path: str,
            max_length: int,
            split: str = "train",
            ignore_id: int = -100,
            task_start_token: str = "<s>",
            prompt_end_token: str = None,
            sort_json_key: bool = True,
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        # self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        self.dataset = data_set
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        for sample in self.dataset:
            # ground_truth = json.loads(sample["ground_truth"])
            ground_truth = json.loads(sample["text"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                # assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                # gt_jsons = [ground_truth["gt_parse"]]
                gt_jsons = [ground_truth]
            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + processor.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )
        print("sequence:", self.gt_token_sequences[-1])
        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # inputs
        pixel_values = processor(sample["image"], random_padding=self.split == "train",
                                 return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        # targets
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        input_ids = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        return pixel_values, labels, target_sequence


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)

        outputs = self.model.generate(pixel_values,
                                      decoder_input_ids=decoder_input_ids,
                                      max_length=max_length,
                                      early_stopping=True,
                                      pad_token_id=self.processor.tokenizer.pad_token_id,
                                      eos_token_id=self.processor.tokenizer.eos_token_id,
                                      use_cache=True,
                                      num_beams=1,
                                      bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                      return_dict_in_generate=True, )
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = list()
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"\nPrediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")
        self.validation_step_outputs.append(scores)
        return scores

    # pytorch_lightning < 2.0.0
    # def validation_epoch_end(self, validation_step_outputs):
    #     # I set this to 1 manually
    #     # (previously set to len(self.config.dataset_name_or_paths))
    #     num_of_loaders = 1
    #     if num_of_loaders == 1:
    #         validation_step_outputs = [validation_step_outputs]
    #     assert len(validation_step_outputs) == num_of_loaders
    #     cnt = [0] * num_of_loaders
    #     total_metric = [0] * num_of_loaders
    #     val_metric = [0] * num_of_loaders
    #     for i, results in enumerate(validation_step_outputs):
    #         for scores in results:
    #             cnt[i] += len(scores)
    #             total_metric[i] += np.sum(scores)
    #         val_metric[i] = total_metric[i] / cnt[i]
    #         val_metric_name = f"val_metric_{i}th_dataset"
    #         self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
    #     self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)

    def on_validation_epoch_end(self):
        # I set this to 1 manually
        # (previously set to len(self.config.dataset_name_or_paths))
        num_of_loaders = 1
        if num_of_loaders == 1:
            self.validation_step_outputs = [self.validation_step_outputs]
        assert len(self.validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(self.validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # TODO add scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


class PushToHubCallback(Callback):
    best_val_metric = 100
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        # pl_module.model.push_to_hub("nielsr/donut-demo",
        #                             commit_message=f"Training in progress, epoch {trainer.current_epoch}")
        if trainer.callback_metrics['val_metric'] < self.best_val_metric:
            print(f"save current best model: epoch_{trainer.current_epoch}_ned_{trainer.callback_metrics['val_metric']}")
            model_save_path = f"./donut-save-hf/epoch_{trainer.current_epoch}_ned_{trainer.callback_metrics['val_metric']}"
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            pl_module.processor.save_pretrained(model_save_path,
                                                commit_message=f"Training in progress, epoch {trainer.current_epoch}")
            pl_module.model.save_pretrained(model_save_path,
                                            commit_message=f"Training in progress, epoch {trainer.current_epoch}")
            self.best_val_metric = trainer.callback_metrics['val_metric']

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        # pl_module.processor.push_to_hub("nielsr/donut-demo",
        #                             commit_message=f"Training done")
        model_save_path = f"./donut-save-hf/epoch_{trainer.current_epoch}_ned_{trainer.callback_metrics['val_metric']}"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        pl_module.processor.save_pretrained(model_save_path,
                                            commit_message=f"Training done")
        # pl_module.model.push_to_hub("nielsr/donut-demo",
        #                             commit_message=f"Training done")
        pl_module.model.save_pretrained(model_save_path,
                                        commit_message=f"Training done")


if __name__ == "__main__":
    # DP = r"C:\Users\21702\.cache\huggingface\datasets\naver-clova-ix___parquet\naver-clova-ix--cord-v2-c97f979311033a44\0.0.0\2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec"
    # train_dataset = Dataset.from_file(os.path.join(DP, "parquet-train.arrow"))
    # test_dataset = Dataset.from_file(os.path.join(DP, "parquet-test.arrow"))
    # validation_dataset = Dataset.from_file(os.path.join(DP, "parquet-validation.arrow"))
    # dataset = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})

    MP = "/mnt/j/model/pretrained-model/torch/donut-base"
    max_length = 768
    image_size = [960, 720]  # [1280, 960]
    added_tokens = []

    # step2: config and model from huggingface
    # update image_size of the encoder
    # during pre-training, a larger image size was used
    config = VisionEncoderDecoderConfig.from_pretrained(MP)
    config.encoder.image_size = image_size  # (height, width)
    # update max_length of the decoder (for generation)
    config.decoder.max_length = max_length
    # TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
    # https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602
    processor = DonutProcessor.from_pretrained(MP)
    model = VisionEncoderDecoderModel.from_pretrained(MP, config=config)
    # ========================================================================
    # we update some settings which differ from pretraining; namely the size of the images + no rotation required
    # source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
    # 修改为
    processor.image_processor.size = image_size[::-1]  # should be (width, height)
    processor.image_processor.do_align_long_axis = False

    # step1: get the data.arrow from generate_metadata.py and generate_arrowdata.py
    dataset_path = "/mnt/j/dataset/document-intelligence/ICDAR-2019-SROIE-master/data/img/data.arrow"
    # step3: load dataset
    from datasets import Dataset, DatasetDict

    dataset = Dataset.from_file(dataset_path)
    dataset = dataset.train_test_split(test_size=0.1)
    print(dataset)

    # load online dataset
    # train_dataset = DonutDataset("naver-clova-ix/cord-v2", max_length=max_length,
    #                              split="train", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
    #                              sort_json_key=False,  # cord dataset is preprocessed, so no need for this
    #                              )
    #
    # val_dataset = DonutDataset("naver-clova-ix/cord-v2", max_length=max_length,
    #                            split="validation", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
    #                            sort_json_key=False,  # cord dataset is preprocessed, so no need for this
    #                            )

    # load local dataset
    # 需要将processor前
    train_dataset = DonutDataset(dataset["train"], max_length=max_length,
                                 split="train", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
                                 sort_json_key=False,  # cord dataset is preprocessed, so no need for this
                                 )

    val_dataset = DonutDataset(dataset["test"], max_length=max_length,
                               split="validation", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
                               sort_json_key=False,  # cord dataset is preprocessed, so no need for this
                               )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]

    # sanity check
    print("Pad token ID:", processor.decode([model.config.pad_token_id]))
    print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))

    # feel free to increase the batch size if you have a lot of memory
    # I'm fine-tuning on Colab and given the large image size, batch size > 1 is not feasible
    # dataloader for pytorch_lightning
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(train_dataloader))
    pixel_values, labels, target_sequences = batch
    print("pixel_values:", pixel_values.shape)

    # for id in labels.squeeze().tolist()[:30]:
    #     if id != -100:
    #         print(processor.decode([id]))
    #     else:
    #         print(id)
    #
    # batch = next(iter(val_dataloader))
    # pixel_values, labels, target_sequences = batch
    # print(pixel_values.shape)

    # step4: Pl module config
    config = {"max_epochs": 25,
              "val_check_interval": 0.2,  # how many times we want to validate during an epoch
              "check_val_every_n_epoch": 1,
              "gradient_clip_val": 1.0,
              "num_training_samples_per_epoch": 800,
              "lr": 3e-5,
              "train_batch_sizes": [1],
              "val_batch_sizes": [1],
              # "seed":2022,
              "num_nodes": 1,
              "warmup_steps": 300,  # 800/8*30/10, 10%
              "result_path": "./result",
              "verbose": True,
              }

    model_module = DonutModelPLModule(config, processor, model)  # .load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=0-step=559.ckpt")

    print(target_sequences[0])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.get("max_epochs"),
        val_check_interval=config.get("val_check_interval"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision=16,  # we'll use mixed precision
        num_sanity_val_steps=0,
        # logger=wandb_logger,
        callbacks=[
            PushToHubCallback(),  # hf model save to local
            EarlyStopping(monitor="val_metric", patience=10, mode="min")
        ],
    )

    # step5: train
    trainer.fit(model_module)
    print("trained")
    # model_module.model.save_pretrained("./save")
