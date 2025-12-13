**Download the data:**
Download the [annotation](https://huggingface.co/datasets/zhuomingliu/START_eval/blob/main/chartqapro.json) and the [image](https://huggingface.co/datasets/zhuomingliu/START_eval/blob/main/chartqapro_images.tar) of ChartQAPro, put them under eval/chartqapro.


**Run inference:**
```bash
# Qwen2.5-VL-7B
python eval/chartqapro/inference_with_vllm.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --image_root eval/chartqapro/images \
  --batch_size 8 \
  --output_file ./predictions/chartqapro_predictions.json

# START-RL-7B
python eval/chartqapro/inference_with_vllm.py \
  --model_name zhuomingliu/START \
  --image_root eval/chartqapro/images \
  --batch_size 8 \
  --output_file ./predictions/chartqapro_predictions_start.json \
  --answer_format box \
  --system_prompt_template eval/chart_format_v2.jinja
```

**Compute statistics:**
```bash
# Qwen2.5-VL-7B, Overall: 41.97%
python eval/chartqapro/evaluate_predictions.py --predictions-file ./predictions/chartqapro_predictions.json

# START-RL-7B, Overall: 47.19%
python eval/chartqapro/evaluate_predictions.py --predictions-file ./predictions/chartqapro_predictions_start.json
```
