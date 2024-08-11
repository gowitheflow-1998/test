import argparse
import copy
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import transformers
from datasets import load_dataset
from pixel import (
    PangoCairoTextRenderer,
    CLIPTrainerForContrastiveWithEvalGradCache,
    PIXELTrainingArguments,
    PyGameTextRenderer,
    glue_strip_spaces,
)

from transformers import CLIPProcessor, AutoModel
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import datasets

datasets_keys = {
    "stsb": ('SetFit/stsb', "text1", "text2"),
    "mteb": ('mteb/stsbenchmark-sts', "sentence1", "sentence2"),
    "allnli": ("gowitheflow/allnli-sup", "sentence1", "sentence2"),
    "ir": ("gowitheflowlab/ir", "sentence1", "sentence2"),
    "allnli-ir": ("gowitheflowlab/allnli-ir", "sentence1", "sentence2"),
    "allnli-reasoning": ("gowitheflowlab/allnli-reasoning", "sentence1", "sentence2"),
    "allnli-reasoning-6": ("gowitheflowlab/allnli-reasoning-6", "sentence1", "sentence2"),
    "allnli-reasoning-7": ("gowitheflowlab/allnli-reasoning-6-tr", "sentence1", "sentence2"),
    "math":("gowitheflowlab/math-train", "sentence1","sentence2"),
    "code":("gowitheflowlab/code-train", "sentence1","sentence2"),
    "allnlineg": ("gowitheflow/allnli-withnegs", "sentence1", "sentence2&sentence3"),
    "unsup-simcse": ("gowitheflow/wiki1M-character-level-all", "sentence1", "sentence1"),
    "unsup-c": ("gowitheflow/wiki1M-character-level-all", "sentence1", "sentence2"),
    "unsup-wr": ("gowitheflow/wiki1M-word-random-shuffle", "sentence1", "sentence2"),
    "unsup-wc": ("gowitheflow/wiki1M-word-condition-shuffle", "sentence1", "sentence2"),
    "unsup-wa": ("gowitheflow/wiki1M-word-character-all-multiple", "sentence1", "sentence2"),
    "para":("sentence-transformers/parallel-sentences", "Membership of Parliament: see Minutes", "Състав на Парламента: вж. протоколи"),
    "wikispan":("gowitheflow/wiki-span", "sentence1", "sentence2"),
    "msmarco":("bclavie/msmarco-10m-triplets", "query", "positive"),
    "compression":("sentence-transformers/sentence-compression", "text","simplified"),
    "en-de": ("gowitheflowlab/nli-sts-en-de", "sentence1", "sentence2"),
    "en-es": ("gowitheflowlab/nli-sts-en-es", "sentence1", "sentence2"),
    "en-fr": ("gowitheflowlab/nli-sts-en-fr", "sentence1", "sentence2"),
    "en-it": ("gowitheflowlab/nli-sts-en-it", "sentence1", "sentence2"),
    "en-nl": ("gowitheflowlab/nli-sts-en-nl", "sentence1", "sentence2"),
    "en-pl": ("gowitheflowlab/nli-sts-en-pl", "sentence1", "sentence2"),
    "en-pt": ("gowitheflowlab/nli-sts-en-pt", "sentence1", "sentence2"),
    "en-ru": ("gowitheflowlab/nli-sts-en-ru", "sentence1", "sentence2"),
    "en-zh": ("gowitheflowlab/nli-sts-en-zh", "sentence1", "sentence2"),
    "xnli-pooled":("gowitheflowlab/multi-pooled", "sentence1","sentence2"),
    "xnli-random":("gowitheflowlab/allnli-en-xnli-multi-random","sentence1","sentence2"),
    "parallel-pt-nl-pl":("gowitheflowlab/parallel-pt-nl-pl","sentence1","sentence2"),
    "parallel-9":("gowitheflowlab/parallel-9","sentence1","sentence2"),
    "parallel-all":("gowitheflowlab/parallel-all","sentence1","sentence2"),
    "parallel-small":("gowitheflowlab/parallel-small","English","Other Language"),
    "parallel-medium":("gowitheflowlab/parallel-medium","English","Other Language"),
    "parallel-small-nli":("gowitheflowlab/parallel-small-w-nli","English","Other Language"),
    "parallel-medium-nli":("gowitheflowlab/parallel-medium-w-nli","English","Other Language"),
    "supervised-multilingual":("gowitheflow/supervised-multilingual","sentence1","sentence2")
}


logger = logging.getLogger(__name__)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def condition(example):
    return example['label'].capitalize() == 'Entailment'

def get_column_names(dataset_name: str) -> Tuple[str, str, Optional[str]]:
    dataset_info = datasets_keys[dataset_name]
    sentence1_key, sentence2_key = dataset_info[1], dataset_info[2]
    if '&' in sentence2_key:
        sentence2_key, sentence3_key = sentence2_key.split('&')
    else:
        sentence3_key = None
    return sentence1_key, sentence2_key, sentence3_key

def filter_dataset(train_dataset):
    if "label" in train_dataset.column_names:
        logger.info("Select positive samples only.")
        train_dataset = train_dataset.filter(condition)
    return train_dataset

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the NLI dataset to use (via the datasets library)."}
    )
    dataset_name_val: Optional[str] = field(
        default="mteb", metadata={"help": "The name of the NLI dataset to use (via the datasets library)."}
    )
    dataset_config_name: str = field(
        default=None, metadata={"help": "Subset of the NLI dataset, e.g language ISO code"}
    )
    max_seq_length: Optional[int] = field(
        default=196,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained processor name or path if not the same as model_name"}
    )
    rendering_backend: Optional[str] = field(
        default="pangocairo", metadata={
            "help": "Rendering backend to use. Options are 'pygame' or 'pangocairo'. For most applications it is "
                    "recommended to use the default 'pangocairo'."}
    )
    fallback_fonts_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing fallback font files used by the text renderer for PIXEL. "
                          "PyGame does not support fallback fonts so this argument is ignored when using the "
                          "PyGame backend."},
    )
    render_rgb: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to render images in RGB. RGB rendering can be useful when working with emoji "
            "but it makes rendering a bit slower, so it is recommended to turn on RGB rendering only "
            "when there is need for it. PyGame does not support fallback fonts so this argument is ignored "
            "when using the PyGame backend."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if not self.rendering_backend.lower() in ["pygame", "pangocairo"]:
            raise ValueError("Invalid rendering backend. Supported backends are 'pygame' and 'pangocairo'.")
        else:
            self.rendering_backend = self.rendering_backend.lower()


def get_processor(model_args: argparse.Namespace):

    renderer_cls = PyGameTextRenderer if model_args.rendering_backend == "pygame" else PangoCairoTextRenderer
    processor = renderer_cls.from_pretrained(
        model_args.processor_name if model_args.processor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token if model_args.use_auth_token else None,
        fallback_fonts_dir=model_args.fallback_fonts_dir,
        rgb=model_args.render_rgb,
        )
    return processor

def get_collator():
    def image_collate_fn(examples):
        pixel_values1 = torch.stack([example["pixel_values1"] for example in examples])
        pixel_values2 = torch.stack([example["pixel_values2"] for example in examples])
        
        if 'pixel_values3' in examples[0]:
            pixel_values3 = torch.stack([example["pixel_values3"] for example in examples])
            pixel_values2 = torch.cat([pixel_values2, pixel_values3], dim=0)

        labels = torch.arange(pixel_values1.size(0), device=pixel_values1.device)

        return {
            'pixel_values1': pixel_values1,
            'pixel_values2': pixel_values2,
            'labels': labels
        }
 
    return image_collate_fn

def transform_to_square(rendered_text, 
                        num_patches_per_side = 14,
                        patch_size = 16):

    final_image = np.zeros((num_patches_per_side * patch_size, num_patches_per_side * patch_size))
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            patch_index = i * num_patches_per_side + j
            patch = rendered_text[:, patch_index * patch_size:(patch_index + 1) * patch_size]
            final_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch
    final_image = final_image.astype(np.uint8)
    final_image_rgb = np.stack((final_image,) * 3, axis=-1)
    return final_image_rgb

def get_preprocess_fn(
    processor: PangoCairoTextRenderer,
    clip_processor: CLIPProcessor,
    sentence_keys: Tuple[str, Optional[str]],
):
    sentence1_key, sentence2_key = sentence_keys
    if "&" in sentence2_key:
        sentence2_key, sentence3_key = sentence2_key.split('&')
    else:
        sentence3_key = None

    format_fn = glue_strip_spaces

    def image_preprocess_fn(examples):

        if not sentence2_key:
            raise ValueError(f"two sentences needed, but got one.")

        encodings = [processor(text=format_fn(a)).pixel_values for a in examples[sentence1_key]]
        squared_images1 = [transform_to_square(e) for e in encodings]
        pixel_values1 = clip_processor(images = squared_images1, return_tensors="pt")
        
        encodings = [processor(text=format_fn(a)).pixel_values for a in examples[sentence2_key]]
        squared_images2 = [transform_to_square(e) for e in encodings]
        pixel_values2 = clip_processor(images = squared_images2, return_tensors="pt")
        
        if sentence3_key is not None:
            encodings = [processor(text=format_fn(a)).pixel_values for a in examples[sentence3_key]]
            squared_images3 = [transform_to_square(e) for e in encodings]
            pixel_values3 = clip_processor(images = squared_images3, return_tensors="pt")
        else:
            pixel_values3 = None
        processed_examples = {
            "pixel_values1": pixel_values1["pixel_values"],
            "pixel_values2": pixel_values2["pixel_values"],
        }
        if pixel_values3 is not None:
            processed_examples["pixel_values3"] = pixel_values3

        if "label" in examples:
            processed_examples["label"] = [l if l != -1 else -100 for l in examples["label"]]
        if "score" in examples:
            processed_examples["label"] = [l if l != -1 else -100 for l in examples["score"]]
        return processed_examples

    return image_preprocess_fn


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, PIXELTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_dataset_name = datasets_keys[data_args.dataset_name][0]
    val_dataset_name = datasets_keys[data_args.dataset_name_val][0]
    sentence1_key, sentence2_key, sentence3_key = get_column_names(data_args.dataset_name)
    val_sentence1_key, val_sentence2_key, val_sentence3_key = get_column_names(data_args.dataset_name_val)

    if training_args.do_train:

        train_dataset = load_dataset(
            train_dataset_name,
            data_args.dataset_config_name,
            split="train",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if training_args.do_eval:

        eval_dataset = load_dataset(
            val_dataset_name,
            data_args.dataset_config_name,
            split="test",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model = AutoModel.from_pretrained(model_args.model_name_or_path).to(training_args.device)

    train_dataset = filter_dataset(train_dataset)
    processor = get_processor(model_args)

    if processor.max_seq_length != data_args.max_seq_length:
        processor.max_seq_length = data_args.max_seq_length

        # resize_model_embeddings(model, processor.max_seq_length)
    clip_processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)
    
    preprocess_fn = get_preprocess_fn(processor,clip_processor,(sentence1_key, sentence2_key))
    preprocess_fn_eval = get_preprocess_fn(processor, clip_processor,(val_sentence1_key, val_sentence2_key))

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset.features["pixel_values"] = datasets.Image()
        train_dataset.set_transform(preprocess_fn)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset.features["pixel_values"] = datasets.Image()
        eval_examples = copy.deepcopy(eval_dataset)
        eval_dataset.set_transform(preprocess_fn_eval)

    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the eval set: {eval_dataset[index]}.")
            
        
    trainer = CLIPTrainerForContrastiveWithEvalGradCache(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor,
        data_collator=get_collator(),
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        kwargs["language"] = data_args.dataset_config_name
        kwargs["dataset_tags"] = data_args.dataset_name
        kwargs["dataset_args"] = data_args.dataset_name
        kwargs["dataset"] = f"{data_args.dataset_name.upper()}"

    trainer.create_model_card(**kwargs)

if __name__ == "__main__":
    main()