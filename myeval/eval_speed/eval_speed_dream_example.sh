MODEL_PATH='path/to/Dream-v0-7B-Instruct/'
MODEL_TYPE='dream_chat'
DATA_PATH='path/to/data'
DATA_TYPE='data_type'
OUTPUT_DIR='path/to/output_dir'
python dream_sparse_dllm.py --model_path ${MODEL_PATH} --model_type ${MODEL_TYPE} --data_path ${DATA_PATH} --data_type ${DATA_TYPE} --output_dir ${OUTPUT_DIR} --kernel_size 3 --keep_ratio 0.5 --block_length 32 --apply_chat_template True

MODEL_PATH='path/to/Dream-v0-7B-Base/'
MODEL_TYPE='dream_base'
python dream_sparse_dllm.py --model_path ${MODEL_PATH} --model_type ${MODEL_TYPE} --data_path ${DATA_PATH} --data_type ${DATA_TYPE} --output_dir ${OUTPUT_DIR} --kernel_size 3 --keep_ratio 0.5 --block_length 32

