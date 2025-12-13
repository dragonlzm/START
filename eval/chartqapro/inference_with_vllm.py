"""
Run vLLM inference on ChartQAPro dataset and save predictions in the format
compatible with the ChartQAPro evaluation script.

Input: eval/chartqapro/chartqapro.json
Output: predictions in the format of eval/chartqapro/sample_predictions.json
"""

import os
import json
import argparse
import re
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm
from jinja2 import Template


def load_system_prompt_template(template_path: str) -> Template:
    """Load a Jinja2 template for the system prompt.

    Args:
        template_path: Path to the Jinja2 template file

    Returns:
        Jinja2 Template object
    """
    with open(template_path, 'r') as f:
        template_content = f.read()
    return Template(template_content)


def apply_system_prompt(question: str, template: Template) -> str:
    """Apply the system prompt template to a question.

    Args:
        question: The original question text
        template: Jinja2 template with 'content' placeholder

    Returns:
        Question with system prompt appended
    """
    return template.render(content=question)


def extract_answer(text: str) -> str:
    """
    Extract content between <answer> and </answer> tags.
    If pattern is found, return the answer content.
    Otherwise, return the full string.

    Args:
        text (str): Input string to search for answer pattern

    Returns:
        str: Either the extracted answer or the full input string
    """
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return text


def extract_boxed_content(text: str) -> str:
    """
    Extract content from \\boxed{} format.
    If pattern is found, return the content inside.
    Otherwise, return "None".

    Args:
        text (str): Input string to search for boxed pattern

    Returns:
        str: Either the extracted content or "None"
    """
    pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(pattern, text)

    if match:
        return match.group(1).strip()
    else:
        return "None"


def load_annotations(annotation_path: str) -> List[Dict[str, Any]]:
    """Load ChartQAPro annotations from JSON file"""
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
    answer_format: str = "answer",
    system_prompt_template: str = None,
):
    """Run vLLM inference on ChartQAPro dataset

    Args:
        annotations: List of annotation dictionaries from ChartQAPro
        image_root: Root directory containing images
        model_name: Path to the vLLM model
        output_file: Path to save output predictions
        batch_size: Batch size for inference
        max_samples: Maximum number of samples to process (None for all)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens to generate
        answer_format: Format to extract answer from model output ("answer" or "box")
        system_prompt_template: Path to Jinja2 template file for system prompt (optional)
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if max_samples is not None:
        annotations = annotations[:max_samples]

    # Load system prompt template if provided
    prompt_template = None
    if system_prompt_template:
        print(f"Loading system prompt template from {system_prompt_template}")
        prompt_template = load_system_prompt_template(system_prompt_template)

    print(f"Loading AutoProcessor for {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    vllm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": 8192,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.95,
    }

    if "Qwen" in model_name or "START" in model_name:
        print(f"Setting image resolution limits for Qwen model")
        vllm_kwargs["mm_processor_kwargs"] = {
            "max_pixels": 4194304,
            "min_pixels": 262144,
        }

    print(f"Initializing vLLM with config: {vllm_kwargs}")
    llm = LLM(**vllm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    results = []
    total_batches = (len(annotations) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(0, len(annotations), batch_size), total=total_batches, desc="Processing batches"):
        batch_annotations = annotations[batch_idx:batch_idx + batch_size]

        prompts = []
        images_list = []
        valid_indices = []

        for idx, annotation in enumerate(batch_annotations):
            img_id = annotation['img_id']
            image_type = annotation.get('image_type', '.png')
            image_path = os.path.join(image_root, f"{img_id}{image_type}")

            if not os.path.exists(image_path):
                image_path = os.path.join(image_root, str(img_id))
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue

            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            question = annotation['question']

            # Apply system prompt template if provided
            if prompt_template:
                question = apply_system_prompt(question, prompt_template)

            images_list.append([img])
            prompts.append(question)
            valid_indices.append(idx)

        if not prompts:
            continue

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

        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
            for prompt, image in zip(formatted_prompts, images_list)
        ]

        outputs = llm.generate(inputs, sampling_params)

        for valid_idx, output in zip(valid_indices, outputs):
            annotation = batch_annotations[valid_idx]
            generated_text = output.outputs[0].text.strip()

            if answer_format == 'answer':
                extracted_answer = extract_answer(generated_text)
            elif answer_format == 'box':
                extracted_answer = extract_boxed_content(generated_text)
                if extracted_answer == "None":
                    extracted_answer = generated_text
            else:
                extracted_answer = generated_text

            result = {
                "Answer": [annotation['answer']],
                "Question Type": annotation['Question Type'],
                "Year": annotation['Year'],
                "prediction": extracted_answer,
            }
            results.append(result)

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run vLLM inference on ChartQAPro dataset")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "chartqapro.json"),
        help="Path to ChartQAPro annotations JSON file"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
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
        default="chartqapro_predictions.json",
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
    parser.add_argument(
        "--answer_format",
        type=str,
        default="answer",
        choices=["answer", "box"],
        help="Format to extract answer from model output ('answer' for <answer> tags, 'box' for \\boxed{})"
    )
    parser.add_argument(
        "--system_prompt_template",
        type=str,
        default=None,
        help="Path to Jinja2 template file for system prompt (e.g., chart_format_v2.jinja). The template should use '{{ content }}' as placeholder for the question."
    )

    args = parser.parse_args()

    print(f"Loading annotations from {args.annotation_path}")
    annotations = load_annotations(args.annotation_path)
    print(f"Loaded {len(annotations)} annotations")

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
        answer_format=args.answer_format,
        system_prompt_template=args.system_prompt_template,
    )


if __name__ == "__main__":
    main()
