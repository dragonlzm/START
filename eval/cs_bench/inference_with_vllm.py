"""CS Bench inference script using vLLM for vision-language models"""

import os
import json
import argparse
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm


def load_annotations(annotation_path: str) -> List[Dict[str, Any]]:
    """Load CS Bench annotations from JSON file"""
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def run_inference(
    annotations: List[Dict[str, Any]],
    image_root: str,
    model_name: str,
    output_file: str,
    batch_size: int = 8,
    max_samples: int = None,
    tensor_parallel_size: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
):
    """Run vLLM inference on CS Bench dataset

    Args:
        annotations: List of annotation dictionaries from CS Bench
        image_root: Root directory containing images
        model_name: Path to the vLLM model
        output_file: Path to save output predictions
        batch_size: Batch size for inference
        max_samples: Maximum number of samples to process (None for all)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens to generate
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    # Set environment variables for vLLM multiprocessing compatibility
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Limit samples if specified
    if max_samples is not None:
        annotations = annotations[:max_samples]

    print(f"Loading AutoProcessor for {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Initialize vLLM model
    vllm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": 8192,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.95,
    }

    # Add image resolution limits for Qwen models to prevent OOM
    if "Qwen" in model_name:
        print(f"Setting image resolution limits for Qwen model")
        vllm_kwargs["mm_processor_kwargs"] = {
            "max_pixels": 4194304,
            "min_pixels": 262144,
        }

    print(f"Initializing vLLM with config: {vllm_kwargs}")
    llm = LLM(**vllm_kwargs)

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Process in batches
    results = []
    total_batches = (len(annotations) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(0, len(annotations), batch_size), total=total_batches, desc="Processing batches"):
        batch_annotations = annotations[batch_idx:batch_idx + batch_size]

        # Prepare batch data
        prompts = []
        images_list = []
        valid_indices = []

        for idx, annotation in enumerate(batch_annotations):
            # Get image path
            img_id = annotation['img_id']
            image_path = os.path.join(image_root, img_id)

            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            # Create prompt with the question
            question = annotation['question']
            prompt = f"Answer the following question about the image:\n\n{question}\n\nProvide your answer:"

            images_list.append([img])
            prompts.append(prompt)
            valid_indices.append(idx)

        if not prompts:
            continue

        # Format prompts using chat template
        formatted_prompts = []
        for prompt in prompts:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]

            try:
                formatted_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted_prompts.append(formatted_prompt)
            except Exception as e:
                print(f"Error applying chat template: {e}")
                formatted_prompts.append(prompt)

        # Run batched inference
        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
            for prompt, image in zip(formatted_prompts, images_list)
        ]

        outputs = llm.generate(inputs, sampling_params)

        # Process outputs
        for valid_idx, output in zip(valid_indices, outputs):
            annotation = batch_annotations[valid_idx]
            generated_text = output.outputs[0].text.strip()

            result = {
                "q_id": annotation['q_id'],
                "question": annotation['question'],
                "ground_truth_answer": annotation['answer'],
                "ground_truth_location": annotation['location'],
                "predicted_answer": generated_text,
                "img_id": annotation['img_id'],
                "image_size": annotation['image_size'],
                "question_type": annotation['question_type'],
                "finish_reason": output.outputs[0].finish_reason,
            }
            results.append(result)

    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {output_file}")
    return results


# sample usage:
# python /storage/home/zliu2346/start_testing/START/eval/cs_bench/inference_with_vllm.py \
#   --model_name Qwen/Qwen2.5-VL-7B-Instruct \
#   --batch_size 8 \
#   --output_file ./predictions/cs_bench_predictions.json

# python /storage/home/zliu2346/start_testing/START/eval/cs_bench/inference_with_vllm.py \
#   --model_name zhuomingliu/START \
#   --batch_size 8 \
#   --output_file ./predictions/cs_bench_predictions_start.json



def main():
    parser = argparse.ArgumentParser(description="Run vLLM inference on CS Bench dataset")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="/storage/home/zliu2346/start_testing/data/cs_bench/cs_bench_annotations.json",
        help="Path to CS Bench annotations JSON file"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/storage/home/zliu2346/start_testing/data/cs_bench/images",
        help="Root directory containing images"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the vLLM model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="cs_bench_predictions.json",
        help="Output file path for predictions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )

    args = parser.parse_args()

    # Load annotations
    print(f"Loading annotations from {args.annotation_path}")
    annotations = load_annotations(args.annotation_path)
    print(f"Loaded {len(annotations)} annotations")

    # Run inference
    run_inference(
        annotations=annotations,
        image_root=args.image_root,
        model_name=args.model_name,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
