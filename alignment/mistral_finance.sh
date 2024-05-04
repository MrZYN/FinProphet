base_model="saves/Mistral/pt"
# base_model="/data/models/AI-ModelScope/Mistral-7B-v0.1"
lora_checkpoint="saves/Mistral/f_sft_lora"
output_dir="saves/Mistral/f_sft"

deepspeed --include localhost:0,1 \
    --master_port 29503 \
    src/train_bash.py \
    --deepspeed fulltune_zero0.json \
    --stage sft \
    --do_train True \
    --model_name_or_path $base_model \
    --finetuning_type lora \
    --template mistral \
    --dataset_dir data \
    --dataset finance_train \
    --cutoff_len 8192 \
    --learning_rate 3e-4 \
    --num_train_epochs 5.0 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_samples 20000 \
    --max_grad_norm 4.0 \
    --logging_steps 3 \
    --save_steps 100000 \
    --warmup_ratio 0.01 \
    --weight_decay 0.1 \
    --gradient_checkpointing False \
    --optim adamw_torch \
    --report_to none \
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

CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path $base_model \
    --adapter_name_or_path $lora_checkpoint \
    --template mistral \
    --finetuning_type lora \
    --export_dir $output_dir \
    --export_size 4 \
    --export_legacy_format False