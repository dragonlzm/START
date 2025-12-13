**Download the data:**
1. Download the annotation ([reasoning](https://huggingface.co/datasets/zhuomingliu/START_eval/resolve/main/charxiv_val_reasoning.json?download=true), [descriptive](https://huggingface.co/datasets/zhuomingliu/START_eval/resolve/main/charxiv_val_descriptive.json?download=true)) and the [images](https://huggingface.co/datasets/zhuomingliu/START_eval/resolve/main/charxiv_images.tar?download=true) under `eval/charxiv`.

2. Clone the CharXiv repo under `eval/charxiv`.
```bash
git clone https://github.com/princeton-nlp/CharXiv.git
```

**Run inference:**
```bash
# For Qwen2.5-VL-7B descriptive evaluation:
python eval/charxiv/inference_with_vllm.py \
    --annotation_path /storage/home/zliu2346/start_testing/START/eval/charxiv/charxiv_val_descriptive.json \
    --image_root /storage/home/zliu2346/start_testing/START/eval/charxiv/images \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --output_file predictions/predictions_descriptive.json

# For Qwen2.5-VL-7B reasoning evaluation:
python eval/charxiv/inference_with_vllm.py \
    --annotation_path /storage/home/zliu2346/start_testing/START/eval/charxiv/charxiv_val_reasoning.json \
    --image_root /storage/home/zliu2346/start_testing/START/eval/charxiv/images \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --output_file predictions/predictions_reasoning.json

# For START-RL-7B descriptive evaluation:
CUDA_VISIBLE_DEVICES=1,2 python eval/charxiv/inference_with_vllm.py \
    --annotation_path /storage/home/zliu2346/start_testing/START/eval/charxiv/charxiv_val_descriptive.json \
    --image_root /storage/home/zliu2346/start_testing/START/eval/charxiv/images \
    --model_name zhuomingliu/START \
    --output_file predictions/predictions_descriptive_start.json \
    --system_prompt_template eval/chart_format_v2.jinja \
    --tensor_parallel_size=2

# For START-RL-7B reasoning evaluation:
python eval/charxiv/inference_with_vllm.py \
    --annotation_path /storage/home/zliu2346/start_testing/START/eval/charxiv/charxiv_val_reasoning.json \
    --image_root /storage/home/zliu2346/start_testing/START/eval/charxiv/images \
    --model_name zhuomingliu/START \
    --output_file predictions/predictions_reasoning_start.json \
    --system_prompt_template eval/chart_format_v2.jinja
```

**Convert to CharXiv format:**
```bash
# For Qwen2.5-VL-7B descriptive
python eval/charxiv/convert_descriptive_train_validation_file_to_charxiv_format.py \
    --train_eval_file predictions/predictions_descriptive.json \
    --save_file eval/charxiv/CharXiv/results/gen-Qwen2.5-VL-7B-Instruct-descriptive_val.json \
    --annotation_file eval/charxiv/CharXiv/data/descriptive_val.json

# For Qwen2.5-VL-7B reasoning
python eval/charxiv/convert_reasoning_train_validation_file_to_charxiv_format.py \
    --train_eval_file predictions/predictions_reasoning.json \
    --save_file eval/charxiv/CharXiv/results/gen-Qwen2.5-VL-7B-Instruct-reasoning_val.json \
    --annotation_file eval/charxiv/CharXiv/data/reasoning_val.json

# The best way to eval the START on CharXiv is to pretent the model is Qwen model (So here we pretend it as Qwen2.5-VL-32B-Instruct)
# For START-RL-7B descriptive
python eval/charxiv/convert_descriptive_train_validation_file_to_charxiv_format.py \
    --train_eval_file predictions/predictions_descriptive_start.json \
    --save_file eval/charxiv/CharXiv/results/gen-Qwen2.5-VL-32B-Instruct-descriptive_val.json \
    --annotation_file eval/charxiv/CharXiv/data/descriptive_val.json \
    --answer_format box

# For START-RL-7B reasoning
python eval/charxiv/convert_reasoning_train_validation_file_to_charxiv_format.py \
    --train_eval_file predictions/predictions_reasoning_start.json \
    --save_file eval/charxiv/CharXiv/results/gen-Qwen2.5-VL-32B-Instruct-reasoning_val.json \
    --annotation_file eval/charxiv/CharXiv/data/reasoning_val.json \
    --answer_format box
```

**run the eval of the with the API:**

```bash
# For Qwen2.5-VL-7B
cd eval/charxiv/CharXiv/
python src/evaluate.py \
    --model_name Qwen2.5-VL-7B-Instruct \
    --split val \
    --mode descriptive \
    --api_key <You openai apikey>


python src/evaluate.py \
    --model_name Qwen2.5-VL-7B-Instruct \
    --split val \
    --mode reasoning \
    --api_key <You openai apikey>

# "Overall Score (Descriptive)": 73.0
# "Overall Score (Reasoning)": 43.2
python src/get_stats.py \
    --model_name Qwen2.5-VL-7B-Instruct \
    --split val


# For START-RL-7B
cd eval/charxiv/CharXiv/
python src/evaluate.py \
    --model_name Qwen2.5-VL-32B-Instruct \
    --split val \
    --mode descriptive \
    --api_key <You openai apikey>

python src/evaluate.py \
    --model_name Qwen2.5-VL-32B-Instruct \
    --split val \
    --mode reasoning \
    --api_key <You openai apikey>

### Get statistics for the model performance ###
# "Overall Score (Descriptive)": 78.5
# "Overall Score (Reasoning)": 47.6
python src/get_stats.py \
    --model_name Qwen2.5-VL-32B-Instruct \
    --split val
```
