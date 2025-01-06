# run train_gpt2.py
# Usage: sh run_train_gpt2.sh
torchrun --standalone --nproc_per_node=8 train_gpt2.py # 8 GPUs
