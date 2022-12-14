export WANDB_WATCH=all

#CUDA_VISIBLE_DEVICES=1
python run_mmsd.py \
--do_train \
--output_dir ../mmsd/ \
--seed=4542 \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--max_steps=2000 \
--logging_steps=50 \
--learning_rate=5e-4 \
--eval_accumulation_steps=32 \
--gradient_accumulation_steps=1 \
--overwrite_output_dir \
--fp16 \
--evaluation_strategy="steps" \
--max_train_samples=8 \
--run_name=mmsd_e_8 \
--pool=True \
--tune="e" \
--prompt_len=0 \
--prompt_mask=0 \
--no_cuda