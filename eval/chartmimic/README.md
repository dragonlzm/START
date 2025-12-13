**Download the data:**
1. download the image from ChartMimic
```bash
cd eval/chartmimic/ChartMimic # cd to the root directory of this repository
mkdir dataset
wget https://huggingface.co/datasets/ChartMimic/ChartMimic/resolve/main/dataset-iclr.tar.gz
tar -xzvf dataset-iclr.tar.gz -C dataset
```

Install ChartMimic environment:
```bash
conda env create -f environment.yaml
```

2. Run the inference
```bash
# Use the base environment to run the inference
# Run inference with Qwen2.5-VL-7B-Instruct
python eval/chartmimic/inference_with_vllm.py \
  --image_root eval/chartmimic/ChartMimic/dataset/direct_600 \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --output_file predictions/qwen2_5_vl_7b_chartmimic_predictions.json

# Run inference with START-7B
CUDA_VISIBLE_DEVICES=1 python eval/chartmimic/inference_with_vllm.py \
  --image_root eval/chartmimic/ChartMimic/dataset/direct_600 \
  --model_name zhuomingliu/START \
  --output_file predictions/start_7b_chartmimic_predictions.json \
  --system_prompt_template eval/chart_format_v2.jinja
```

3. Run the evaluation script
```bash
conda activate chartmimic
pip install mathruler
pip install ipdb
pip install pylatexenc
# pip install torchvision
# you may need to install `Poppler` to load the pdf correctly

export OPENAI_API_KEY="Your OPENAI API key"
# Run the evaluation script for Qwen2.5-VL-7B-Instruct
#                        example_count ExecRate  TextScore LayoutScore  TypeScore ColorScore    Average GPT4VScore    Overall
# model_agent
# qwen2_5_vl_DirectAgent           356     73.0  36.176221   53.488889  37.259196  32.220566  39.786218  54.266667  47.026442
bash eval/chartmimic/eval_qwen2_5_vl.sh

# Run the evaluation script for START
#                        example_count   ExecRate  TextScore LayoutScore TypeScore ColorScore    Average GPT4VScore    Overall
# model_agent
# qwen2_5_vl_DirectAgent           551  76.666667  76.085735   88.888889  64.62226  57.251743  71.712157  68.086667  69.899412
bash eval/chartmimic/eval_start.sh
```
