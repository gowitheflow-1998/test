# export MODEL="openai/clip-vit-base-patch16"
export MODEL="model-ablation/clip-parallel-all-1024-allnli-1024/checkpoint-310"
# export MODEL="./clip-parallel-all-last"  #"./clip-vit-base-patch16"
# export MODEL="./clip-parallel-all-last"
export RENDERER_PATH="model-ablation/clip-parallel-all-1024-allnli-1024/checkpoint-310" #"Team-PIXEL/pixel-base"
export DATASETNAME="parallel-all" #'compression' #"parallel-all" #'allnli' #"unsup-c"
export POOLING_MODE="mean"
export SEQ_LEN=196
export FALLBACK_FONTS_DIR="data/fallback_fonts"
export BSZ=1024
export GRAD_ACCUM=1
export LR=3e-5
export SEED=42
export NUM_STEPS=4000 #1200 #64000
export WARM_STEPS=400 #120 #3000
export RUN_NAME="./model-ablation/clip-parallel-all-1024-allnli-1024-parallel-all-1024"
python scripts/training/run_contrastive_training_eval_clip_gradcache.py \
  --model_name_or_path=${MODEL} \
  --processor_name=${RENDERER_PATH} \
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
  --save_total_limit=3 \
  --load_best_model_at_end=True \
  --seed=${SEED} \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR}


# export MODEL="gowitheflowlab/wikispan-unsup_ensemble_last"
