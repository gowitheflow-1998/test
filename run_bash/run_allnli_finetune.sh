export DATASETNAME='allnli'
export POOLING_MODE="mean"
export FALLBACK_FONTS_DIR="data/fallback_fonts"
export GRAD_ACCUM=1
export LR=3e-5
export SEED=42
export NUM_STEPS=2600
export WARM_STEPS=200

export SEQ_LEN=64
export BSZ=128
export MODEL="gowitheflow/unsup-ensemble-s64-bs128-lr6"
# export MODEL="allnli/contrastive-allnli-mean-gowitheflow/unsup-ensemble-s64-bs128-lr6-1-3e-5-2600-42"

export RUN_NAME="debug-contrastive-${DATASETNAME}-${POOLING_MODE}-${MODEL}-${GRAD_ACCUM}-${BSZ}-${LR}-${NUM_STEPS}-${SEED}"
python scripts/training/run_contrastive_training_eval.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --dataset_name=${DATASETNAME} \
  --do_train \
  --do_eval \
  --metric_for_best_model="eval_pearson_cosine" \
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
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=200 \
  --save_strategy=steps \
  --save_steps=200 \
  --save_total_limit=5 \
  --load_best_model_at_end=True \
  --seed=${SEED} \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR}

