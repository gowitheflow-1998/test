# export MODEL="0-allnli-ps/checkpoint-49400"
# export MODEL="gowitheflowlab/wikispan-unsup_ensemble_last"
# export MODEL="0-unsup-ensemble-wikispan-normed/checkpoint-55000"
# export MODEL="Team-PIXEL/pixel-base"
# export MODEL="zxh4546/allnli_wikispan_unsup_ensemble_last"
export MODEL="model/parallel-medium-iter-3/checkpoint-205000"    #"zxh4546/allnli_wikispan_unsup_ensemble_last"
# export MODEL="00-p9-allnli-wikispan-vanilla/checkpoint-26000" #"0-vanilla-wikispan-allnli-normed"
# export DATASETNAME="parallel-medium-nli"  #'parallel-9'
# export DATASETNAME='wikispan'
export DATASETNAME="allnli"
export POOLING_MODE="mean"
export SEQ_LEN=64
export FALLBACK_FONTS_DIR="data/fallback_fonts"
export BSZ=128
export GRAD_ACCUM=1
export LR=3e-5
export SEED=1
# export NUM_STEPS=2600
# export NUM_STEPS=9400 == parallel-pt-nl-pl
# export NUM_STEPS=26000 == parallel-9
# export NUM_STEPS=32000 == parallel-all
# export NUM_STEPS=49500 == parallel-small
# export NUM_STEPS=52000 == parallel-small-w-nli
# export NUM_STEPS=205000 #== parallel-medium-w-nli
export NUM_STEPS=7400
# export WARM_STEPS=20000
export WARM_STEPS=740

export RUN_NAME="model/pixel-linguist-v0-final-3epochs"
# export RUN_NAME="0-unsup-wc-allnli-normed"
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
  --save_total_limit=2 \
  --load_best_model_at_end=True \
  --seed=${SEED} \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR}


# export MODEL="gowitheflowlab/wikispan-unsup_ensemble_last"