# import os
# from dataclasses import dataclass
#
# from datasets import load_dataset
# from datasets import Dataset, DatasetDict
# from seqeval.metrics import accuracy_score
# from sklearn.metrics import precision_recall_fscore_support
#
# # DP = r"C:\Users\21702\.cache\huggingface\datasets\naver-clova-ix___parquet\naver-clova-ix--cord-v2-c97f979311033a44\0.0.0\2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec"
# # train_dataset = Dataset.from_file(os.path.join(DP, "parquet-train.arrow"))
# # test_dataset = Dataset.from_file(os.path.join(DP, "parquet-test.arrow"))
# # validation_dataset = Dataset.from_file(os.path.join(DP, "parquet-validation.arrow"))
# # dataset = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})
#
# dataset_path = "/mnt/j/dataset/document-intelligence/ICDAR-2019-SROIE-master/data/img/data.arrow"
# MP = "/mnt/j/model/pretrained-model/torch/donut-base"
#
# dataset = Dataset.from_file(dataset_path)
# dataset = dataset.train_test_split(test_size=0.1)
#
# print(dataset)
#
# # ==============================================
# from transformers import VisionEncoderDecoderConfig
#
# max_length = 768
# image_size = [960, 720] #[1280, 960]
#
# # update image_size of the encoder
# # during pre-training, a larger image size was used
# config = VisionEncoderDecoderConfig.from_pretrained(MP)
# config.encoder.image_size = image_size  # (height, width)
# # update max_length of the decoder (for generation)
# config.decoder.max_length = max_length
# # TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
# # https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602
# # %% md
#
# # ================================================
# from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig
#
# # processor = DonutProcessor.from_pretrained("nielsr/donut-base")
# # model = VisionEncoderDecoderModel.from_pretrained("nielsr/donut-base", config=config)
#
# processor = DonutProcessor.from_pretrained(MP)
# processor.image_processor.size = image_size[::-1] # should be (width, height)
# processor.image_processor.do_align_long_axis = False
#
#
# model = VisionEncoderDecoderModel.from_pretrained(MP, config=config)
#
# import json
# import random
# from typing import Any, List, Tuple, Optional
#
# import torch
# from torch.utils.data import Dataset
#
# added_tokens = []
#
# #
# # class DonutDataset(object):
# #     """
# #     DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
# #     Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
# #     and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
# #     Args:
# #         dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
# #         max_length: the max number of tokens for the target sequences
# #         split: whether to load "train", "validation" or "test" split
# #         ignore_id: ignore_index for torch.nn.CrossEntropyLoss
# #         task_start_token: the special token to be fed to the decoder to conduct the target task
# #         prompt_end_token: the special token at the end of the sequences
# #         sort_json_key: whether or not to sort the JSON keys
# #     """
# #
# #     def __init__(
# #             self,
# #             data_set,
# #             max_length: int,
# #             split: str = "train",
# #             ignore_id: int = -100,
# #             task_start_token: str = "<s>",
# #             prompt_end_token: str = None,
# #             sort_json_key: bool = True,
# #     ):
# #         super().__init__()
# #
# #         self.max_length = max_length
# #         self.split = split
# #         self.ignore_id = ignore_id
# #         self.task_start_token = task_start_token
# #         self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
# #         self.sort_json_key = sort_json_key
# #
# #         #self.dataset = load_dataset(dataset_name_or_path, split=self.split)
# #         self.dataset = data_set
# #         self.dataset_length = len(self.dataset)
# #
# #         self.gt_token_sequences = []
# #         for sample in self.dataset:
# #             # ground_truth = json.loads(sample["ground_truth"])
# #             ground_truth = json.loads(sample["text"])
# #             if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
# #                 assert isinstance(ground_truth["gt_parses"], list)
# #                 gt_jsons = ground_truth["gt_parses"]
# #             else:
# #                 #assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
# #                 #gt_jsons = [ground_truth["gt_parse"]]
# #                 gt_jsons = [ground_truth]
# #             self.gt_token_sequences.append(
# #                 [
# #                     self.json2token(
# #                         gt_json,
# #                         update_special_tokens_for_json_key=self.split == "train",
# #                         sort_json_key=self.sort_json_key,
# #                     )
# #                     + processor.tokenizer.eos_token
# #                     for gt_json in gt_jsons  # load json from list of json
# #                 ]
# #             )
# #         print("sequence:", self.gt_token_sequences[-1])
# #         self.add_tokens([self.task_start_token, self.prompt_end_token])
# #         self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
# #
# #     def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
# #         """
# #         Convert an ordered JSON object into a token sequence
# #         """
# #         if type(obj) == dict:
# #             if len(obj) == 1 and "text_sequence" in obj:
# #                 return obj["text_sequence"]
# #             else:
# #                 output = ""
# #                 if sort_json_key:
# #                     keys = sorted(obj.keys(), reverse=True)
# #                 else:
# #                     keys = obj.keys()
# #                 for k in keys:
# #                     if update_special_tokens_for_json_key:
# #                         self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
# #                     output += (
# #                             fr"<s_{k}>"
# #                             + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
# #                             + fr"</s_{k}>"
# #                     )
# #                 return output
# #         elif type(obj) == list:
# #             return r"<sep/>".join(
# #                 [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
# #             )
# #         else:
# #             obj = str(obj)
# #             if f"<{obj}/>" in added_tokens:
# #                 obj = f"<{obj}/>"  # for categorical special tokens
# #             return obj
# #
# #     def add_tokens(self, list_of_tokens: List[str]):
# #         """
# #         Add special tokens to tokenizer and resize the token embeddings of the decoder
# #         """
# #         newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
# #         if newly_added_num > 0:
# #             model.decoder.resize_token_embeddings(len(processor.tokenizer))
# #             added_tokens.extend(list_of_tokens)
# #
# #     def __len__(self) -> int:
# #         return self.dataset_length
# #
# #     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         """
# #         Load image from image_path of given dataset_path and convert into input_tensor and labels
# #         Convert gt data into input_ids (tokenized string)
# #         Returns:
# #             input_tensor : preprocessed image
# #             input_ids : tokenized gt_data
# #             labels : masked labels (model doesn't need to predict prompt and pad token)
# #         """
# #         sample = self.dataset[idx]
# #
# #         # inputs
# #         pixel_values = processor(sample["image"], random_padding=self.split == "train",
# #                                  return_tensors="pt").pixel_values
# #         pixel_values = pixel_values.squeeze()
# #
# #         # targets
# #         target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
# #         input_ids = processor.tokenizer(
# #             target_sequence,
# #             add_special_tokens=False,
# #             max_length=self.max_length,
# #             padding="max_length",
# #             truncation=True,
# #             return_tensors="pt",
# #         )["input_ids"].squeeze(0)
# #
# #         labels = input_ids.clone()
# #         labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
# #         # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
# #         return pixel_values, labels, target_sequence
#
#
# class DonutDataset(Dataset):
#     """
#     DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
#     Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
#     and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
#     Args:
#         dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
#         max_length: the max number of tokens for the target sequences
#         split: whether to load "train", "validation" or "test" split
#         ignore_id: ignore_index for torch.nn.CrossEntropyLoss
#         task_start_token: the special token to be fed to the decoder to conduct the target task
#         prompt_end_token: the special token at the end of the sequences
#         sort_json_key: whether or not to sort the JSON keys
#     """
#
#     def __init__(
#             self,
#             data_set,
#             #dataset_name_or_path: str,
#             max_length: int,
#             split: str = "train",
#             ignore_id: int = -100,
#             task_start_token: str = "<s>",
#             prompt_end_token: str = None,
#             sort_json_key: bool = True,
#     ):
#         super().__init__()
#
#         self.max_length = max_length
#         self.split = split
#         self.ignore_id = ignore_id
#         self.task_start_token = task_start_token
#         self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
#         self.sort_json_key = sort_json_key
#
#         #self.dataset = load_dataset(dataset_name_or_path, split=self.split)
#         self.dataset = data_set
#         self.dataset_length = len(self.dataset)
#
#         self.gt_token_sequences = []
#         for sample in self.dataset:
#             # ground_truth = json.loads(sample["ground_truth"])
#             ground_truth = json.loads(sample["text"])
#             if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
#                 assert isinstance(ground_truth["gt_parses"], list)
#                 gt_jsons = ground_truth["gt_parses"]
#             else:
#                 #assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
#                 #gt_jsons = [ground_truth["gt_parse"]]
#                 gt_jsons = [ground_truth]
#             self.gt_token_sequences.append(
#                 [
#                     self.json2token(
#                         gt_json,
#                         update_special_tokens_for_json_key=self.split == "train",
#                         sort_json_key=self.sort_json_key,
#                     )
#                     + processor.tokenizer.eos_token
#                     for gt_json in gt_jsons  # load json from list of json
#                 ]
#             )
#         print("sequence:", self.gt_token_sequences[-1])
#         self.add_tokens([self.task_start_token, self.prompt_end_token])
#         self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
#
#     def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
#         """
#         Convert an ordered JSON object into a token sequence
#         """
#         if type(obj) == dict:
#             if len(obj) == 1 and "text_sequence" in obj:
#                 return obj["text_sequence"]
#             else:
#                 output = ""
#                 if sort_json_key:
#                     keys = sorted(obj.keys(), reverse=True)
#                 else:
#                     keys = obj.keys()
#                 for k in keys:
#                     if update_special_tokens_for_json_key:
#                         self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
#                     output += (
#                             fr"<s_{k}>"
#                             + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
#                             + fr"</s_{k}>"
#                     )
#                 return output
#         elif type(obj) == list:
#             return r"<sep/>".join(
#                 [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
#             )
#         else:
#             obj = str(obj)
#             if f"<{obj}/>" in added_tokens:
#                 obj = f"<{obj}/>"  # for categorical special tokens
#             return obj
#
#     def add_tokens(self, list_of_tokens: List[str]):
#         """
#         Add special tokens to tokenizer and resize the token embeddings of the decoder
#         """
#         newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
#         if newly_added_num > 0:
#             model.decoder.resize_token_embeddings(len(processor.tokenizer))
#             added_tokens.extend(list_of_tokens)
#
#     def __len__(self) -> int:
#         return self.dataset_length
#
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Load image from image_path of given dataset_path and convert into input_tensor and labels
#         Convert gt data into input_ids (tokenized string)
#         Returns:
#             input_tensor : preprocessed image
#             input_ids : tokenized gt_data
#             labels : masked labels (model doesn't need to predict prompt and pad token)
#         """
#         sample = self.dataset[idx]
#
#         # inputs
#         pixel_values = processor(sample["image"], random_padding=self.split == "train",
#                                  return_tensors="pt").pixel_values
#         pixel_values = pixel_values.squeeze()
#
#         # targets
#         target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
#         input_ids = processor.tokenizer(
#             target_sequence,
#             add_special_tokens=False,
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )["input_ids"].squeeze(0)
#
#         labels = input_ids.clone()
#         labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
#         # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
#         #return pixel_values, labels, target_sequence
#
#         #return pixel_values, labels, target_sequence
#         return {"pixel_values": pixel_values, "labels": labels, "target_sequence": target_sequence}
#
#
# # ========================================================================
# # we update some settings which differ from pretraining; namely the size of the images + no rotation required
# # source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
#
# # processor.feature_extractor.size = image_size[::-1] # should be (width, height)
# # processor.feature_extractor.do_align_long_axis = False
#
# # 修改为
# processor.image_processor.size = image_size[::-1]  # should be (width, height)
# processor.image_processor.do_align_long_axis = False
#
# # train_dataset = DonutDataset("naver-clova-ix/cord-v2", max_length=max_length,
# #                              split="train", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
# #                              sort_json_key=False,  # cord dataset is preprocessed, so no need for this
# #                              )
# #
# # val_dataset = DonutDataset("naver-clova-ix/cord-v2", max_length=max_length,
# #                            split="validation", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
# #                            sort_json_key=False,  # cord dataset is preprocessed, so no need for this
# #                            )
#
# train_dataset = DonutDataset(dataset["train"], max_length=max_length,
#                              split="train", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
#                              sort_json_key=False,  # cord dataset is preprocessed, so no need for this
#                              )
#
# val_dataset = DonutDataset(dataset["test"], max_length=max_length,
#                            split="validation", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
#                            sort_json_key=False,  # cord dataset is preprocessed, so no need for this
#                            )
#
# from torch.utils.data import DataLoader
#
# model.config.pad_token_id = processor.tokenizer.pad_token_id
# model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]
#
# # sanity check
# print("Pad token ID:", processor.decode([model.config.pad_token_id]))
# print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))
#
# print("added_tokens:", added_tokens)
#
# @dataclass
# class DataCollatorForDonut:
#     processor: DonutProcessor
#     max_length: Optional[int] = None
#
#     def __call__(self, features):
#         # prepare image input
#         features = features
#         pixel_values = torch.stack([f["pixel_values"] for f in features])
#         labels = torch.stack([f["labels"] for f in features])
#         batch = dict()
#         batch["pixel_values"] = pixel_values
#         batch["labels"] = labels
#         return batch
#
#
# data_collator = DataCollatorForDonut(
#     processor,
#     max_length=768,
# )
#
#
#
# from huggingface_hub import HfFolder
# from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
#
# # hyperparameters used for multiple args
# hf_repository_id = "donut-base-sroie"
#
# # Arguments for training
# training_args = Seq2SeqTrainingArguments(
#     output_dir=hf_repository_id,
#     num_train_epochs=20,
#     learning_rate=2e-5,
#     per_device_train_batch_size=1,
#     weight_decay=0.01,
#     fp16=True,
#     logging_steps=100,
#     save_total_limit=2,
#     evaluation_strategy="steps",
#     eval_steps=100,
#     do_eval=True,
#     save_strategy="steps",
#     save_steps=500,
#     predict_with_generate=True,
#     # push to hub parameters
#     report_to="tensorboard",
#     push_to_hub=False,
#     hub_strategy="every_save",
#     hub_model_id=hf_repository_id,
#     hub_token=HfFolder.get_token(),
# )
#
# def compute_metrics(p):
#     pred_relations, gt_relations = p
#     # score = re_score(pred_relations, gt_relations, mode="boundaries")
#     # print(pred_relations)
#     # print(gt_relations)
#     # print(processor.decode(pred_relations[0][0]))
#     # print(processor.decode(gt_relations[0][0]))
#     for gts, preds in zip(gt_relations,pred_relations):
#         gt_tokens, pred_tokens = [], []
#         for gt in gts:
#             if gt == -100:
#                 continue
#             gt_tokens.append(processor.decode(gt))
#         for pred in preds:
#             pred_tokens.append(processor.decode(pred))
#         print("\n")
#         print("gt:", "".join(gt_tokens))
#         print("pre:", "".join(pred_tokens))
#
#     return {
#         'accuracy': 0.3,
#         'f1': 0.3,
#         'precision': 0.3,
#         'recall': 0.3
#     }
#
#
#
#
# # Create Trainer
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     data_collator=data_collator,
#     #compute_metrics=compute_metrics,
#
# )
#
# trainer.train()