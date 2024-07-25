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
from mteb import MTEB
from sentence_transformers import SentenceTransformer

class PixelLinguist:
    def __init__(self, model_name, batch_size = 16, max_seq_length = 64, 
                 device=None, keep_mlp = False):
        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.config = AutoConfig.from_pretrained(model_name, num_labels=0)
        self.batch_size = batch_size
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
        self.processor.max_seq_length = max_seq_length
        resize_model_embeddings(self.model, self.processor.max_seq_length)
        self.transforms = get_transforms(do_resize=True, size=(self.processor.pixels_per_patch, self.processor.pixels_per_patch * self.processor.max_seq_length))

    def preprocess(self, texts):
        encodings = [self.processor(text=glue_strip_spaces(a)) for a in texts]
        pixel_values = torch.stack([self.transforms(Image.fromarray(e.pixel_values)) for e in encodings])
        attention_mask = torch.stack([get_attention_mask(e.num_text_patches, seq_length=self.processor.max_seq_length) for e in encodings])
        return {'pixel_values': pixel_values, 'attention_mask': attention_mask}

    def encode(self, texts, **kwargs):
        all_outputs = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i+self.batch_size]
            inputs = self.preprocess(batch_texts)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs).logits.detach().cpu()
            all_outputs.append(outputs)
        return torch.cat(all_outputs, dim=0)
    
batch_size = 32

model_name = "AnonymousPage/checkpoint-all"
# model_name = "./model-ablation/unsup-simcse-last-allnli"
# model_name = "pixel-linguist-v0"
# model_name = "Team-PIXEL/pixel-base"
# model_name = "model/0-allnli-normed"
# model_name = "./model/0-unsup-c-allnli-normed" #"model/0-unsup-wc-allnli-normed"
# model_name = "model/0-unsup-wc-allnli-normed"
# model_name = "model/0-unsup-ensemble-allnli-normed"
# model_name = "model/0-unsup-ensemble-wikispan-allnli-normed"
# model_name = "model/pixel-linguist-v0-final-3epochs"
# model_name = "model/pixel-linguist-v0-final-3epochs"
# model_name = "Pixel-Linguist/Pixel-Linguist-v0"
model = PixelLinguist(model_name, keep_mlp=True)
# model_name = "sentence-transformers/nli-bert-base"
# model_name = "princeton-nlp/unsup-simcse-bert-base-uncased"
# model_name = "google-bert/bert-base-cased"
# model = SentenceTransformer(model_name)
tasks = ["STSBenchmark"]
# tasks = ["STSBenchmark","SICK-R","STS12","STS13","STS14","STS15","STS16"]
# tasks = ["STS17"]
evaluation = MTEB(tasks = tasks, task_langs=["en"])
# evaluation = MTEB(tasks=tasks, task_langs=["en"])
# evaluation = MTEB(task_types=['Classification'], task_langs=["en"])
results = evaluation.run(model, output_folder=f"results/all-sts/{model_name}")

