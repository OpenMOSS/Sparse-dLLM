MODEL_PATH='path/to/LLaDA-8B-Instruct/'
MODEL_TYPE='llada_chat'
DATA_PATH='path/to/data'
DATA_TYPE='data_type'
OUTPUT_DIR='path/to/output_dir'
python llada_sparse_dllm.py --model_path ${MODEL_PATH} --model_type ${MODEL_TYPE} --data_path ${DATA_PATH} --data_type ${DATA_TYPE} --output_dir ${OUTPUT_DIR} --kernel_size 3 --keep_ratio 0.5

MODEL_PATH='path/to/LLaDA-1.5/'
MODEL_TYPE='llada_1_5'
python llada_sparse_dllm.py --model_path ${MODEL_PATH} --model_type ${MODEL_TYPE} --data_path ${DATA_PATH} --data_type ${DATA_TYPE} --output_dir ${OUTPUT_DIR} --kernel_size 3 --keep_ratio 0.5

