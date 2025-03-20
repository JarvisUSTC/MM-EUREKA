set -x

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export RAY_TEMP_DIR=/tmp/ray_shared_memory

OUTPUT_DIR='outputs/train_mm_eureka_7b_single_node_qwenvl2_5/'

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export WORKING_DIR=$PWD

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if [ "$NODE_RANK" -eq 0 ]; then
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 4
else
    sleep 30
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 4 --block
fi

sleep 30


if [ "$PET_NODE_RANK" -eq 0 ]; then
  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
    --working-dir $WORKING_DIR \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --ref_num_gpus_per_node 1 \
    --ref_num_nodes 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_enable_sleep \
    --pretrain /workspace/Jiawei/Models/Qwen2.5-VL-7B-Instruct  \
    --remote_rm_url examples/scripts/reward_func_qwen.py \
    --save_path ${OUTPUT_DIR} \
    --micro_train_batch_size 1 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 1 \
    --rollout_batch_size 32 \
    --temperature 1.0 \
    --n_samples_per_prompt 4 \
    --max_epochs 1 \
    --num_episodes 3 \
    --prompt_max_len 3000 \
    --max_samples 100000 \
    --generate_max_len 3000 \
    --advantage_estimator rloo \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 3e-7 \
    --init_kl_coef 0.0 \
    --prompt_data /workspace/Jiawei/Datasets/MM-Eureka-Dataset/dataset_qwen.jsonl \
    --disable_fast_tokenizer \
    --input_key message \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 50 \
    --ckpt_path "${OUTPUT_DIR}/ckpt" \
    --max_ckpt_num 1000000 \
    --save_hf_ckpt \
    --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
    --train_vlm \
    --load_checkpoint | tee ${OUTPUT_DIR}/training.log
fi
#     --colocate_all_models \
# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward