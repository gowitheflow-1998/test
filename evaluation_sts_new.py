import torch
from PIL import Image
from pixel import (
    AutoConfig,
    PangoCairoTextRenderer,
    PIXELForSequenceClassification,
    PIXELForRepresentation,
    PoolingMode,
    get_attention_mask,
    get_transforms,
    glue_strip_spaces,
    resize_model_embeddings,
)
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr

class PixelLinguist:
    def __init__(self, model_name, num_labels=0, device=None, keep_mlp = False):
        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        if keep_mlp == True:
            self.model = PIXELForSequenceClassification.from_pretrained(
                model_name,
                config=self.config,
                pooling_mode=PoolingMode.from_string("mean"),
                add_layer_norm=True
            ).to(self.device)
        else:
            self.model = PIXELForRepresentation.from_pretrained(
                model_name,
                config=self.config,
                pooling_mode=PoolingMode.from_string("mean"),
                add_layer_norm=True
            ).to(self.device)
        self.processor = PangoCairoTextRenderer.from_pretrained(model_name, rgb=False)
        self.processor.max_seq_length = 64
        resize_model_embeddings(self.model, self.processor.max_seq_length)
        self.transforms = get_transforms(do_resize=True, size=(self.processor.pixels_per_patch, self.processor.pixels_per_patch * self.processor.max_seq_length))

    def preprocess(self, texts):
        encodings = [self.processor(text=glue_strip_spaces(a)) for a in texts]
        pixel_values = torch.stack([self.transforms(Image.fromarray(e.pixel_values)) for e in encodings])
        attention_mask = torch.stack([get_attention_mask(e.num_text_patches, seq_length=self.processor.max_seq_length) for e in encodings])
        return {'pixel_values': pixel_values, 'attention_mask': attention_mask}

    def encode(self, texts, batch_size=16, **kwargs):
        all_outputs = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = self.preprocess(batch_texts)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs).logits.detach().cpu()
            all_outputs.append(outputs)
        return torch.cat(all_outputs, dim=0)
    
dataset = load_dataset(
        'stsb_multi_mt',
        "en",
        split="test",
        cache_dir=None,
        use_auth_token=None,
    )
sentences1 = dataset["sentence1"]
sentences2 = dataset["sentence2"]
labels = dataset["similarity_score"]

batch_size = 16

model_name = "Pixel-Linguist/Pixel-Linguist-v0"
model = PixelLinguist(model_name, keep_mlp=True)
embeddings1 = model.encode(sentences1, batch_size=batch_size)
embeddings2 = model.encode(sentences2, batch_size=batch_size)
cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)
print(eval_spearman_cosine)