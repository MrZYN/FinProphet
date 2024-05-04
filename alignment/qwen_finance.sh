# base_model="/data/models/qwen/Qwen1.5-7B"
base_model="saves/qwen/pt"
lora_checkpoint="saves/qwen/f_sft_lora"
output_dir="saves/qwen/f_sft"

deepspeed --num_gpus 2 \
    src/train_bash.py \
    --deepspeed fulltune_zero0.json \
    --stage sft \
    --do_train True \
    --model_name_or_path $base_model \
    --finetuning_type lora \
    --template qwen \
    --dataset_dir data \
    --dataset finance_train \
    --cutoff_len 1000000 \
    --learning_rate 3e-4 \
    --num_train_epochs 5.0 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 4.0 \
    --adam_beta2 0.95 \
    --logging_steps 3 \
    --save_steps 10000 \
    --warmup_ratio 0.01 \
    --weight_decay 0.1 \
    --optim adamw_torch \
    --gradient_checkpointing False \
    --output_dir $lora_checkpoint \
    --overwrite_output_dir \
    --bf16 True \
    --lora_rank 64 \
    --lora_dropout 0.2 \
    --lora_target all \
    --plot_loss True

if [ $? -ne 0 ]; then
    exit 1
fi

CUDA_VISIBLE_DEVICES=1 python src/export_model.py \
    --model_name_or_path $base_model \
    --adapter_name_or_path $lora_checkpoint \
    --template qwen \
    --finetuning_type lora \
    --export_dir $output_dir \
    --export_size 4 \
    --export_legacy_format False