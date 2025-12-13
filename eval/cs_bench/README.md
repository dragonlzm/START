**Download the data:**
Download the data from [here](https://huggingface.co/datasets/zhuomingliu/CS-Bench).

**Run inference:**
```bash
# Qwen2.5-VL-7B
python eval/cs_bench/inference_with_vllm.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --batch_size 8 \
  --output_file ./predictions/cs_bench_predictions.json

# START-RL-7B
python eval/cs_bench/inference_with_vllm.py \
  --model_name zhuomingliu/START \
  --batch_size 8 \
  --output_file ./predictions/cs_bench_predictions_start.json
```

**Compute statistics:**
```bash
# Qwen2.5-VL-7B
# acc: 0.509, recall@0.5: 0.169, recall@0.3: 0.231
python eval/cs_bench/compute_the_stat_from_eval_file.py \
    --pred_res_path ./predictions/cs_bench_predictions.json \
    --save_res_path ./predictions/cs_bench_statistics.json

# START-RL-7B
# acc: 0.655, recall@0.5: 0.329, recall@0.3: 0.486
python eval/cs_bench/compute_the_stat_from_eval_file.py \
    --pred_res_path ./predictions/cs_bench_predictions_start.json \
    --save_res_path ./predictions/cs_bench_statistics_start.json
```
