from typing import List, Optional
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.utils import logging
from grad_cache import GradCache
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from grad_cache.functional import cached, cat_input_tensor
from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states
from contextlib import nullcontext
from functools import partial
from typing import Any, Iterable, Iterator, Optional

class CLIPTrainerForContrastiveWithEvalGradCache(Trainer):
    """
    Trainer class for contrastive learning with evaluation, specialized for CLIP model with image inputs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_loss = CachedMultipleNegativesRankingLoss(self.model)  # Initialize the cached loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        pixel_values1 = inputs.pop("pixel_values1").to(self.args.device)
        pixel_values2 = inputs.pop("pixel_values2").to(self.args.device)

        # Create sentence features for the CachedMultipleNegativesRankingLoss
        sentence_features = [
            {"pixel_values": pixel_values1},
            {"pixel_values": pixel_values2},
        ]

        loss = self.cached_loss(sentence_features, labels=None)

        outputs = (pixel_values1, pixel_values2)
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
            for step, inputs in enumerate(tqdm.tqdm(eval_dataloader)):
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
    
class RandContext:
    """
    Random-state context manager class.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict],
    loss_obj: 'CachedMultipleNegativesRankingLoss',
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                surrogate.backward()


class CachedMultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        scale: float = 20.0,
        similarity_fct: callable = None,
        mini_batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Boosted version of MultipleNegativesRankingLoss by GradCache.

        Constrastive learning (here our MNRL loss) with in-batch negatives is usually hard to work with large batch sizes due to (GPU) memory limitation.
        Even with batch-scaling methods like gradient-scaling, it cannot work either. This is because the in-batch negatives make the data points within
        the same batch non-independent and thus the batch cannot be broke down into mini-batches. GradCache is a smart way to solve this problem.
        It achieves the goal by dividing the computation into two stages of embedding and loss calculation, which both can be scaled by mini-batches.
        As a result, memory of constant size (e.g. that works with batch size = 32) can now process much larger batches (e.g. 65536).

        In detail:

            (1) It first does a quick embedding step without gradients/computation graphs to get all the embeddings;
            (2) Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings;
            (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain.

        Notes: All steps are done with mini-batches. In the original implementation of GradCache, (2) is not done in mini-batches and
        requires a lot memory when batch size large. One drawback is about the speed. GradCache will sacrifice around 20% computation time according to the paper.

        Args:
            model: Model with vision encoders
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between embeddings. Default is cosine similarity.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/pdf/2101.06983.pdf
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct or self.dot_product #self.cos_sim
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mini_batch_size = mini_batch_size
        self.cache: Optional[list[list[Tensor]]] = None
        self.random_states: Optional[list[list[RandContext]]] = None
        self.show_progress_bar = show_progress_bar

    @staticmethod
    def cos_sim(a: Tensor, b: Tensor) -> Tensor:
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))
    @staticmethod
    def dot_product(a: Tensor, b: Tensor) -> Tensor:
        return torch.mm(a, b.transpose(0, 1))
    
    def embed_minibatch(
        self,
        sentence_feature: dict,
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: Optional[RandContext] = None,
    ) -> tuple[Tensor, Optional[RandContext]]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        pixel_values_minibatch = sentence_feature['pixel_values'][begin:end]
        with random_state_context:
            with grad_context():
                random_state = RandContext(pixel_values_minibatch) if copy_random_state else None
                reps = self.model.get_image_features(pixel_values=pixel_values_minibatch)  # (mbsz, hdim)
                reps = torch.nn.functional.normalize(reps, p=2, dim=1)
        return reps, random_state


    def embed_minibatch_iter(
        self,
        sentence_feature: dict,
        with_grad: bool,
        copy_random_state: bool,
        random_states: Optional[list[RandContext]] = None,
    ) -> Iterator[tuple[Tensor, Optional[RandContext]]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        pixel_values: Tensor = sentence_feature['pixel_values']
        bsz, _ = pixel_values.shape[:2]
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat([torch.cat(r) for r in reps[1:]])  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=embeddings_a.device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            loss_mbatch.backward()
            losses.append(loss_mbatch.detach())
            # scores_ab: Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            # scores_ba: Tensor = self.similarity_fct(embeddings_b[b:e], embeddings_a) * self.scale
            # loss_ab: torch.Tensor = self.cross_entropy_loss(scores_ab, labels[b:e]) * len(scores_ab) / batch_size
            # loss_ba: torch.Tensor = self.cross_entropy_loss(scores_ba, labels[b:e]) * len(scores_ba) / batch_size
            # loss_mbatch = (loss_ab + loss_ba) / 2
            # loss_mbatch.backward()
            # losses.append(loss_mbatch.detach())
            
        loss = sum(losses).requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def calculate_loss(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss. No need to cache the gradients."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat([torch.cat(r) for r in reps[1:]])  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=embeddings_a.device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            losses.append(loss_mbatch)
            # scores_ab: Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            # scores_ba: Tensor = self.similarity_fct(embeddings_b[b:e], embeddings_a) * self.scale
            # loss_ab: torch.Tensor = self.cross_entropy_loss(scores_ab, labels[b:e]) * len(scores_ab) / batch_size
            # loss_ba: torch.Tensor = self.cross_entropy_loss(scores_ba, labels[b:e]) * len(scores_ba) / batch_size
            # loss_mbatch = (loss_ab + loss_ba) / 2
            # losses.append(loss_mbatch)
            
        loss = sum(losses)
        return loss

    def forward(self, sentence_features: Iterable[dict], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps)

        return loss

    def get_config_dict(self) -> dict:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
    
# class CLIPTrainerForContrastiveWithEvalGradCache(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.chunk_size = 64  # Assuming batch size is passed as an argument
#         self.grad_cache = GradCache(
#             models=[self.model, self.model],
#             chunk_sizes=self.chunk_size,  # Use the batch size as the chunk size
#             loss_fn=self.contrastive_loss,
#             get_rep_fn=lambda v: F.normalize(v, p=2, dim=1)  # Normalize within get_rep_fn
#         )
#         # self.scaler = torch.cuda.amp.GradScaler()  # Initialize GradScaler here
#         self.cache_x = []
#         self.cache_y = []
#         self.closures_x = []
#         self.closures_y = []
#         self.step = 0  # Initialize step

#     @cat_input_tensor
#     @autocast()
#     def contrastive_loss(self, x, y):
#         target = torch.arange(0, y.size(0), int(y.size(0) / x.size(0)), device=x.device)
#         scores = torch.matmul(x, y.transpose(0, 1))
#         return F.cross_entropy(scores, target=target)

#     @cached
#     @autocast()
#     def call_model(self, model, input):
#         output = model.get_image_features(pixel_values=input)
#         normalized_output = F.normalize(output, p=2, dim=1)  # Normalize within call_model
#         return normalized_output

#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")

#         pixel_values1 = inputs.pop("pixel_values1").to(self.args.device)
#         pixel_values2 = inputs.pop("pixel_values2").to(self.args.device)

#         # Initialize caches and closures
#         cache_x = []
#         cache_y = []
#         closures_x = []
#         closures_y = []

#         # Split the batch into smaller chunks
#         batch_size = pixel_values1.size(0)
#         chunk_size = self.chunk_size

#         for start in range(0, batch_size, chunk_size):
#             end = start + chunk_size

#             outputs_a, closure_a = self.call_model(model, pixel_values1[start:end])
#             outputs_b, closure_b = self.call_model(model, pixel_values2[start:end])

#             cache_x.append(outputs_a)
#             cache_y.append(outputs_b)
#             closures_x.append(closure_a)
#             closures_y.append(closure_b)
#         # Compute the loss after caching all chunks
#         loss = self.contrastive_loss(cache_x, cache_y)
#         # self.scaler.scale(loss).backward(retain_graph=True)
#         loss.backward(retain_graph=True)
#         # Apply the closures after computing the loss
#         for f, r in zip(closures_x, cache_x):
#             f(r)
#         for f, r in zip(closures_y, cache_y):
#             f(r)

#         # self.scaler.step(self.optimizer)
#         # self.scaler.update()
#         self.optimizer.step()
#         self.optimizer.zero_grad()
        
#         return loss
#         # outputs = (cache_x, cache_y)
#         # return (loss, outputs) if return_outputs else loss

#     def evaluate(self, ignore_keys=None, metric_key_prefix: str = "eval"):
#         logger.info("*** Training Evaluate ***")

#         total_output_a = []
#         total_output_b = []

#         args = self.args
#         model = self.model.to(args.device)

#         model.eval()
#         eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
#         with torch.no_grad():
#             for step, inputs in enumerate(tqdm(eval_dataloader)):
#                 pixel_values1 = inputs.pop("pixel_values1").to(args.device)
#                 pixel_values2 = inputs.pop("pixel_values2").to(args.device)

#                 outputs_a = model.get_image_features(pixel_values=pixel_values1)
#                 outputs_b = model.get_image_features(pixel_values=pixel_values2)

#                 total_output_a.append(outputs_a.detach().cpu())
#                 total_output_b.append(outputs_b.detach().cpu())

#         embeddings1 = torch.cat(total_output_a, dim=0)
#         embeddings2 = torch.cat(total_output_b, dim=0)
#         labels = [n['label'] for n in self.eval_dataset]

#         cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
#         manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
#         euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
#         dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

#         eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
#         eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

#         eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
#         eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

#         eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
#         eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

#         eval_pearson_dot, _ = pearsonr(labels, dot_products)
#         eval_spearman_dot, _ = spearmanr(labels, dot_products)

#         metrics = {
#             'eval_loss': 0,  # Placeholder to avoid errors
#             'pearson_cosine': eval_pearson_cosine,
#             'spearman_cosine': eval_spearman_cosine,
#             'pearson_manhattan': eval_pearson_manhattan,
#             'spearman_manhattan': eval_spearman_manhattan,
#             'pearson_euclidean': eval_pearson_euclidean,
#             'spearman_euclidean': eval_spearman_euclidean,
#             'pearson_dot': eval_pearson_dot,
#             'spearman_dot': eval_spearman_dot,
#         }

#         # Prefix all keys with metric_key_prefix + '_'
#         metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

#         logger.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
#             eval_pearson_cosine, eval_spearman_cosine))
#         logger.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
#             eval_pearson_manhattan, eval_spearman_manhattan))
#         logger.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
#             eval_pearson_euclidean, eval_spearman_euclidean))
#         logger.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
#             eval_pearson_dot, eval_spearman_dot))

#         self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
#         return metrics