# export MODEL="2-parallel-all-**zxh4546/allnli_wikispan_unsup_ensemble_last**-64-128-3e-5-32000/checkpoint-32000"
# export MODEL="Team-PIXEL/pixel-base"
export MODEL="model-ablation/simcse-unsup/checkpoint-7800"
# export MODEL="gowitheflowlab/wikispan-unsup_ensemble_last"
# export MODEL="gowitheflowlab/wikispan-unsup_ensemble_last"
# export DATASETNAME='en-de'
export DATASETNAME='allnli'
export POOLING_MODE="mean"
export SEQ_LEN=64
export FALLBACK_FONTS_DIR="data/fallback_fonts"
export BSZ=128
export GRAD_ACCUM=1
export LR=3e-5
export SEED=42
export NUM_STEPS=2600
# export NUM_STEPS=9400 == parallel-pt-nl-pl
# export NUM_STEPS=26000 == parallel-9
# export NUM_STEPS=32000 == parallel-all
# export NUM_STEPS=29600
# export NUM_STEPS=15400
# export NUM_STEPS=3600
export WARM_STEPS=200
# export WARM_STEPS=200
export RUN_NAME="./model-ablation/unsup-simcse-last-allnli"
# export RUN_NAME="./model-ablation/full-en-de"
python scripts/training/run_contrastive_training_eval.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --dataset_name=${DATASETNAME} \
  --do_train \
  --do_eval \
  --metric_for_best_model="spearman_cosine" \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --per_device_train_batch_size=${BSZ} \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=${WARM_STEPS} \
  --run_name=${RUN_NAME} \
  --output_dir=${RUN_NAME} \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=200 \
  --evaluation_strategy=steps \
  --eval_steps=200 \
  --save_strategy=steps \
  --save_steps=200 \
  --save_total_limit=5 \
  --load_best_model_at_end=True \
  --seed=${SEED} \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR}


# export MODEL="gowitheflowlab/wikispan-unsup_ensemble_last"
