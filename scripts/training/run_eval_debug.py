import argparse
import copy
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from PIL import Image
from pixel import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Modality,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PIXELTrainerForContrastive,
    PIXELTrainingArguments,
    PoolingMode,
    PyGameTextRenderer,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    log_sequence_classification_predictions,
    resize_model_embeddings,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import datasets
from torch.utils.data import ConcatDataset

check_min_version("4.17.0")

require_version("datasets>=1.8.0", "To fix: pip install ./datasets")


datasets_keys = {
    "snli": ("rungalileo/snli", "premise", "hypothesis"),
    "mnli": ("SetFit/mnli", "text1", "text2"),
    "stsb": ('SetFit/stsb', "text1", "text2"),
}

DATASET_NAME = 'snli,mnli'
POOLING_MODE = "mean"
FALLBACK_FONTS_DIR = "data/fallback_fonts"
SEQ_LEN = 64
BSZ = 4

def get_sentence_keys(example):
    for name, val in datasets_keys.items():
        this_dataset_name, sentence1_key, sentence2_key = val
        if sentence1_key in example and sentence2_key in example:
            return sentence1_key, sentence2_key
        else:
            continue


class MULTIDATASETS(ConcatDataset):
    def __init__(self, dataset_list, split):
        # construct each datasets

        multi_datasets = []
        for name in dataset_list:
            dataset_name, sentence1_key, sentence2_key = datasets_keys[name]
            dataset = load_dataset(dataset_name, split=split)
            dataset.features["pixel_values"] = datasets.Image()
            multi_datasets.append(dataset)

        super().__init__(multi_datasets)


def main():
    dataset_name = DATASET_NAME.split(',')
    sentence1_key, sentence2_key = "premise", "hypothesis"

    print('Loading dataset')

    train_dataset = MULTIDATASETS(dataset_name, split="train")

    try:
        label_list = train_dataset.features["label"].names
    except AttributeError:
        label_list = None

    # Labels
    num_labels = 0  # len(label_list) no need
    if label_list:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    else:  # for mnli
        label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    model_name = "Team-PIXEL/pixel-base"

    print(f'Building models for {model_name}')
    config_kwargs = {
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task='snli',
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        **config_kwargs,
    )

    print(f'model type: {config.model_type}')

    model = PIXELForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        pooling_mode=PoolingMode.from_string(POOLING_MODE),
        add_layer_norm=True,
        **config_kwargs,
    )

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    modality = Modality.IMAGE
    renderer_cls = PangoCairoTextRenderer
    processor = renderer_cls.from_pretrained(
        model_name,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
        fallback_fonts_dir=FALLBACK_FONTS_DIR,
        rgb=False,
    )

    processor.max_seq_length = SEQ_LEN
    resize_model_embeddings(model, processor.max_seq_length)

    transforms = get_transforms(
        do_resize=True,
        size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
    )
    format_fn = glue_strip_spaces

    def image_preprocess_fn(examples):
        if sentence1_key not in examples:  # direct search from datasets_keys
            sentence1_key, sentence2_key = get_sentence_keys(examples)

        encodings = [processor(text=format_fn(a)) for a in examples[sentence1_key]]
        examples["pixel_values1"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
        examples["attention_mask1"] = [
            get_attention_mask(e.num_text_patches, seq_length=SEQ_LEN) for e in encodings
        ]

        encodings = [processor(text=format_fn(a)) for a in examples[sentence2_key]]
        examples["pixel_values2"] = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
        examples["attention_mask2"] = [
            get_attention_mask(e.num_text_patches, seq_length=SEQ_LEN) for e in encodings
        ]

        if "label" in examples:
            examples["label"] = [l if l != -1 else -100 for l in examples["label"]]

        return examples

    preprocess_fn = image_preprocess_fn

    train_dataset.set_transform(preprocess_fn)

    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    def image_collate_fn(examples):

        # two sentences for contrastive learning

        pixel_values1 = torch.stack([example["pixel_values1"] for example in examples])
        attention_mask1 = torch.stack([example["attention_mask1"] for example in examples])

        pixel_values2 = torch.stack([example["pixel_values2"] for example in examples])
        attention_mask2 = torch.stack([example["attention_mask2"] for example in examples])

        if "label" in examples[0]:
            labels = torch.LongTensor([example["label"] for example in examples])
        else:
            labels = None

        return {
            'pixel_values': labels,  # for ignore warning obly
            'sentence1': {"pixel_values": pixel_values1, "attention_mask": attention_mask1},
            'sentence2': {"pixel_values": pixel_values2, "attention_mask": attention_mask2},
            'labels': labels
        }

    seed_ = random.randint(0, train_dataset.__len__() - BSZ)
    inputs = [train_dataset[seed_ + idx] for idx in range(0, BSZ)]

    inputs = image_collate_fn(inputs)

    if "labels" in inputs:
        labels = inputs.pop("labels")

    # mask = [model.config.id2label[int(label)].capitalize() == 'Entailment' for label in labels]

    sentence1 = inputs.pop("sentence1")
    sentence2 = inputs.pop("sentence2")

    outputs_a = model(**sentence1)
    outputs_b = model(**sentence2)

    print(outputs_a['logits'].shape, outputs_b['logits'].shape)



if __name__ == "__main__":
    main()