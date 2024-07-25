import torch
from pixel import (
    PangoCairoTextRenderer,
    glue_strip_spaces,)
from tqdm import tqdm
from mteb import MTEB
from transformers import AutoModel, CLIPProcessor
import numpy as np
from typing import List

class PixelLinguist:
    def __init__(self, model_path=None, **kwargs):
        self.SEQ_LEN = 196 
        self.FALLBACK_FONTS_DIR = "data/fallback_fonts" 
        self.sep = " "
        self.model =AutoModel.from_pretrained(
            model_path
        )
        self.processor = PangoCairoTextRenderer.from_pretrained(
            "Team-PIXEL/pixel-base",
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

    def encode(self, texts: List[str], batch_size: int, **kwargs) -> np.ndarray:
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
    
batch_size = 32

model_name = "model-ablation/clip-parallel-all-1024-allnli-1024/checkpoint-310"
# model_name = "openai/clip-vit-base-patch16"#"model-ablation/clip-parallel-all-1024/checkpoint-4000"
model = PixelLinguist(model_name, keep_mlp=True)
# model_name = "sentence-transformers/nli-bert-base"
# model_name = "princeton-nlp/unsup-simcse-bert-base-uncased"
# model_name = "google-bert/bert-base-cased"
# model = SentenceTransformer(model_name)
tasks = ["STSBenchmark","STS17"]
# tasks = ["STSBenchmark","SICK-R","STS12","STS13","STS14","STS15","STS16"]
# tasks = ["STS17"]
evaluation = MTEB(tasks = tasks)
# evaluation = MTEB(tasks=tasks, task_langs=["en"])
# evaluation = MTEB(task_types=['Classification'], task_langs=["en"])
results = evaluation.run(model, output_folder=f"results/all-sts/{model_name}")

