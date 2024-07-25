from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoModel, CLIPProcessor
from pixel import (
    PangoCairoTextRenderer,
    glue_strip_spaces
)
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import List, Dict
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

class PixelForRetrieval:
    def __init__(self, model_path=None, **kwargs):
        self.SEQ_LEN = 196 
        self.FALLBACK_FONTS_DIR = "data/fallback_fonts" 
        self.sep = " "
        self.model =AutoModel.from_pretrained(
            model_path
        )
        self.processor = PangoCairoTextRenderer.from_pretrained(
            model_path,
            cache_dir=None,
            revision="main",
            use_auth_token=None,
            fallback_fonts_dir=self.FALLBACK_FONTS_DIR,
            rgb=False,
        )
        self.clip_processor = CLIPProcessor.from_pretrained(model_path)

        self.processor.max_seq_length = self.SEQ_LEN
        self.format_fn = glue_strip_spaces
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def transform_to_square(self,rendered_text, 
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

    def encode_text(self, texts: List[str], batch_size: int) -> np.ndarray:
        embeddings = []

        for text_batch in tqdm([texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]):
            batch_embeddings = []
            encodings = [self.processor(text=self.format_fn(a)).pixel_values for a in text_batch]
            squared_images = [self.transform_to_square(e) for e in encodings]
            pixel_values = self.clip_processor(images = squared_images, return_tensors="pt")["pixel_values"]            
            
            with torch.no_grad():
                batch_embeddings = self.model.get_image_features(pixel_values = pixel_values.to(self.device))
            embeddings.append(batch_embeddings.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        return self.encode_text(queries, batch_size)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode_text(sentences, batch_size)

# model_paths = ["model-ablation/clip-parallel-all-1024/checkpoint-4000"]
# model_paths = ["model-ablation/clip-parallel-all-1024-allnli-1024/checkpoint-310"]
# model_paths = ["model-ablation/compression/checkpoint-180"]
# dataset = "arguana"#"nq"
for model_path in model_paths:
    dataset = "arguana"
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join("datasets")
    # data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=f"datasets/{dataset}").load(split="test")
    # dataset = "math-pooled"
    # dataset = "winogrande"
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join("datasets")
    # data_path = util.download_and_unzip(url, out_dir)
    # corpus, queries, qrels = GenericDataLoader(data_folder=f"RAR-b/full/{dataset}").load(split="test")
    model = DRES(PixelForRetrieval(model_path=model_path),batch_size=128)
    retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    print(ndcg, recall)