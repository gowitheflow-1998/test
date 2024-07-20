from typing import List, Optional
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, is_torch_tpu_available
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import IterableDatasetShard, find_batch_size, nested_concat, nested_numpify
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    PredictionOutput,
    denumpify_detensorize,
    has_length,
)
from transformers.utils import logging

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr

from .utils.optimization import get_cosine_schedule_to_min_lr_with_warmup
from .utils.training import debug_log_inputs


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class PIXELTrainer(Trainer):
    """
    Same as a regular Trainer but with the option to visualize inputs before they are fed into the model
    for debugging purposes
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Uncomment this to visualize inputs
        # debug_log_inputs(inputs)

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

class CLIPTrainerForContrastiveWithEval(Trainer):
    """
    Trainer class for contrastive learning with evaluation, specialized for CLIP model with image inputs.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        labels = inputs.pop("labels")

        pixel_values1 = inputs.pop("pixel_values1").to(self.args.device)
        pixel_values2 = inputs.pop("pixel_values2").to(self.args.device)

        outputs_a = model.get_image_features(pixel_values=pixel_values1)
        outputs_b = model.get_image_features(pixel_values=pixel_values2)
        embeddings_a = F.normalize(outputs_a, p=2, dim=1)
        embeddings_b = F.normalize(outputs_b, p=2, dim=1)

        scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) / 0.05

        # Labels are the indices of the correct matches
        labels = torch.arange(scores.size(0), device=scores.device)

        loss_fct = CrossEntropyLoss()
        loss = (loss_fct(scores, labels) + loss_fct(scores.transpose(0, 1), labels)) / 2

        outputs = (outputs_a, outputs_b)
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, ignore_keys=None, metric_key_prefix: str = "eval"):
        logger.info("*** Training Evaluate ***")

        total_output_a = []
        total_output_b = []

        args = self.args
        model = self.model.to(args.device)

        model.eval()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader)):
                pixel_values1 = inputs.pop("pixel_values1").to(args.device)
                pixel_values2 = inputs.pop("pixel_values2").to(args.device)

                outputs_a = model.get_image_features(pixel_values=pixel_values1)
                outputs_b = model.get_image_features(pixel_values=pixel_values2)

                total_output_a.append(outputs_a.detach().cpu())
                total_output_b.append(outputs_b.detach().cpu())

        embeddings1 = torch.cat(total_output_a, dim=0)
        embeddings2 = torch.cat(total_output_b, dim=0)
        labels = [n['label'] for n in self.eval_dataset]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        metrics = {
            'eval_loss': 0,  # Placeholder to avoid errors
            'pearson_cosine': eval_pearson_cosine,
            'spearman_cosine': eval_spearman_cosine,
            'pearson_manhattan': eval_pearson_manhattan,
            'spearman_manhattan': eval_spearman_manhattan,
            'pearson_euclidean': eval_pearson_euclidean,
            'spearman_euclidean': eval_spearman_euclidean,
            'pearson_dot': eval_pearson_dot,
            'spearman_dot': eval_spearman_dot,
        }

        # Prefix all keys with metric_key_prefix + '_'
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

        logger.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logger.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logger.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        logger.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    
class PIXELTrainerForContrastiveWithEval(Trainer):
    """
    Same as a regular Trainer but with the option to visualize inputs before they are fed into the model
    for debugging purposes
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")

        # mask = [model.config.id2label[int(label)].capitalize() == 'Entailment' for label in labels]

        sentence1 = inputs.pop("sentence1")
        sentence2 = inputs.pop("sentence2")

        # if 'sentence3' in inputs:
        #     sentence3 = inputs.pop("sentence3")
        #     sentence2 = torch.cat([sentence2, sentence3], dim=0)

        outputs_a = model(**sentence1)
        outputs_b = model(**sentence2)

        embeddings_a = outputs_a['logits']
        embeddings_b = outputs_b['logits']

        # after pool
        scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) / 0.05

        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=embeddings_a.device)  # Example a[i] should match with b[i]

        loss = (model.loss(scores, labels) + model.loss(scores.transpose(0, 1), labels)) / 2
        # loss = model.loss(scores, labels)

        outputs = (outputs_a, outputs_b)

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, ignore_keys=None, metric_key_prefix: str = "eval"):

        logger.info("*** Training Evaluate ***")

        total_output_a = []
        total_output_b = []

        args = self.args
        model = self.model.to(args.device)

        model.eval()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        batch_size = eval_dataloader.batch_size
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader)):
            # for step in tqdm(range(0, len(self.eval_dataset), bs)):
                # inputs = [self.eval_dataset[step + idx] for idx in range(0, min(bs, len(self.eval_dataset) - step))]
                sentence1 = inputs.pop("sentence1")
                sentence2 = inputs.pop("sentence2")

                sentence1 = {k: v.to(args.device) for k, v in sentence1.items()}
                sentence2 = {k: v.to(args.device) for k, v in sentence2.items()}

                outputs_a = model(**sentence1).logits
                outputs_b = model(**sentence2).logits

                total_output_a.append(outputs_a.detach().cpu())
                total_output_b.append(outputs_b.detach().cpu())

        embeddings1 = torch.cat(total_output_a, dim=0)
        embeddings2 = torch.cat(total_output_b, dim=0)
        labels = [n['label'] for n in self.eval_dataset]

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        metrics = {}
        metrics['eval_loss'] = 0  # for ignore error obly
        metrics['pearson_cosine'] = eval_pearson_cosine
        metrics['spearman_cosine'] = eval_spearman_cosine
        metrics['pearson_manhattan'] = eval_pearson_manhattan
        metrics['spearman_manhattan'] = eval_spearman_manhattan
        metrics['pearson_euclidean'] = eval_pearson_euclidean
        metrics['spearman_euclidean'] = eval_spearman_euclidean
        metrics['pearson_dot'] = eval_pearson_dot
        metrics['spearman_dot'] = eval_spearman_dot

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # self.log(metrics)
        logger.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logger.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logger.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        logger.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


class PIXELTrainerForContrastive(Trainer):
    """
    Same as a regular Trainer but with the option to visualize inputs before they are fed into the model
    for debugging purposes
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")

        mask = [model.config.id2label[int(label)].capitalize() == 'Entailment' for label in labels]

        sentence1 = inputs.pop("sentence1")
        sentence2 = inputs.pop("sentence2")

        outputs_a = model(**sentence1)
        outputs_b = model(**sentence2)

        embeddings_a = outputs_a['logits'][mask]
        embeddings_b = outputs_b['logits'][mask]

        # after pool
        scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1))/ 0.05

        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=embeddings_a.device)  # Example a[i] should match with b[i]

        loss = (model.loss(scores, labels) + model.loss(scores.transpose(0, 1), labels)) / 2

        outputs = (outputs_a, outputs_b)

        return (loss, outputs) if return_outputs else loss
    
class PIXELTrainerForPretraining(PIXELTrainer):
    """
    PIXELTrainer for pretraining
    """

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_schedule_to_min_lr_with_warmup(
                self.optimizer if optimizer is None else optimizer,
                self.args.get_warmup_steps(num_training_steps),
                num_training_steps,
                self.args.learning_rate,
            )
        return self.lr_scheduler


class PIXELTrainerForBiaffineParsing(PIXELTrainer):
    """
    PIXELTrainer for biaffine universal dependency parsing.
    """

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.

        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            arc_logits, rel_logits = logits
            arc_labels, rel_labels = labels

            mask = arc_labels.ne(model.config.pad_token_id)
            arc_preds = torch.argmax(arc_logits, dim=-1)[mask]

            arc_labels = arc_labels[mask]

            rel_preds, rel_labels = rel_logits[mask], rel_labels[mask]
            rel_preds = rel_preds[torch.arange(len(arc_labels)), arc_labels]
            rel_preds = torch.argmax(rel_preds, dim=-1)

            logits = (arc_preds, rel_preds)
            labels = (arc_labels, rel_labels)

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.

        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        """
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        """

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


class PIXELTrainerForQuestionAnswering(PIXELTrainer):
    """
    PixelTrainer for extractive question answering
    """

    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(eval_dataloader, description="Evaluation")
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(predict_dataloader, description="Prediction")
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
