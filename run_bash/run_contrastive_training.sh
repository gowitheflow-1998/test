export MODEL="Team-PIXEL/pixel-base"
export DATASETNAME='snli'
export POOLING_MODE="mean"
export SEQ_LEN=64
export FALLBACK_FONTS_DIR="data/fallback_fonts"
export BSZ=128
export GRAD_ACCUM=1
export LR=3e-5
export SEED=42
export NUM_STEPS=5000
export WARM_STEPS=500

export RUN_NAME="contrastive-${DATASETNAME}-$(basename ${MODEL})-${POOLING_MODE}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
python scripts/training/run_contrastive_nli.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --dataset_name=${DATASETNAME} \
  --do_train \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=${WARM_STEPS} \
  --run_name=${RUN_NAME} \
  --output_dir=${RUN_NAME} \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=2000 \
  --save_strategy=steps \
  --save_steps=2000 \
  --save_total_limit=50 \
  --load_best_model_at_end=True \
  --seed=${SEED} \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR}


# --do_eval
# --do_predict \
# --metric_for_best_model="eval_accuracy" \