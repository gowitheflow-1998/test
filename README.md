# Visual Sentence Representation Learning

## Installation

```
conda create -n pixel python=3.9 -y && conda activate pixel
```

### package install
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge pycairo pygobject manimpango scikit-learn
pip install -r requirements.txt
pip install ./datasets
pip install -e .
```

## Fallback fonts downloading
```
python scripts/data/download_fallback_fonts.py ‘data/fallback_fonts’
```

## Finetune Visual Sentence
```
bash run_bash/run_contrastive_training_eval.sh
```



