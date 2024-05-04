base_model="/data/models/AI-ModelScope/Mistral-7B-Instruct-v0.2"
lora_checkpoint="saves/Mistral/pt_temp"
output_dir="saves/Mistral/pt"

deepspeed --include localhost:2,3 --master_port 29588 \
    src/train_bash.py \
    --deepspeed fulltune_zero0.json \
    --stage pt \
    --do_train True \
    --model_name_or_path $base_model \
    --quantization_bit 8 \
    --finetuning_type lora \
    --template default \
    --dataset_dir data \
    --dataset all \
    --cutoff_len 8192 \
    --learning_rate 3e-4 \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.5 \
    --adam_beta2 0.95 \
    --logging_steps 1 \
    --save_steps 10000 \
    --warmup_ratio 0.01 \
    --weight_decay 0.1 \
    --optim adamw_torch \
    --gradient_checkpointing False \
    --output_dir $lora_checkpoint \
    --overwrite_output_dir \
    --streaming \
    --max_steps 10000000 \
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
    --template qwen \
    --finetuning_type lora \
    --export_dir $output_dir \
    --export_size 4 \
    --export_legacy_format False