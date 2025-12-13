import os
import json
import argparse
import ipdb
import torch
import multiprocessing as mp
from multiprocessing import Process, Manager
from generate_lib.utils import get_generate_fn, get_client_fn, generate_response_remote_wrapper

ADDITIONAL_PROMPT = {
    'focus_rl': 'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.',
    'normal_rl': 'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.'
}

SFT_WIHT_CODE='''### Role
You are a visual-reasoning assistant.  
Your first task is to **reverse-engineer** Python code that could have produced the given chart image.  
Your second task is to **answer the user's question about the chart**.  
Think step-by-step but reveal only the required outputs.

### Inputs
1. **Chart image:** (see attached image)  
2. **User question:**  
   {question}

### Instructions
1. **Infer Chart-Generating Code**  
   • Re-create the plot as faithfully as possible: axes, titles, labels, legend, colours, line/marker styles, etc.  
   • Use reasonable dummy data if exact values can't be read.  
   • Wrap the code in a fenced block with the language tag `python`.  
   • Do **not** execute the code—just write it.  

2. **Provide the Final Answer**  
   • After the code block, answer the user's question in plain language.  
   • Base your answer solely on evidence visible in the image (or derivable from the reconstructed code).  
   • Enclose the entire answer in LaTeX `\boxed{{}}` so it renders highlighted.

3. **Output Format (strict)**  

```python
# --- Inferred chart-generating code ---
<your code here>
```

**Answer:** \\boxed{{<your answer here>}}
'''


SFT_WIHT_COT = """You FIRST think about the reasoning process as an internal monologue and then provide the final answer. 
The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}.
{question}
"""



def run_inference_on_gpu(gpu_id, queries_split, model_path, model_api, results_dict):
    """Run inference on a specific GPU with a split of the dataset"""
    # Set CUDA_VISIBLE_DEVICES to use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"GPU {gpu_id}: Processing {len(queries_split)} queries")
    
    try:
        generate_fn = get_generate_fn(model_path)
        if model_api is not None:
            client, model = get_client_fn(model_path)(model_path, model_api)
            generate_response_remote_wrapper(generate_fn, queries_split, model, model_api, client)
        else:
            generate_fn(queries_split, model_path)
        
        # Store results in shared dictionary
        for k, v in queries_split.items():
            results_dict[k] = v
            
        print(f"GPU {gpu_id}: Completed processing")
        
    except Exception as e:
        print(f"GPU {gpu_id}: Error occurred: {e}")
        raise e

def split_dict(input_dict, n_splits):
    """Split a dictionary into n roughly equal parts"""
    items = list(input_dict.items())
    avg_size = len(items) // n_splits
    remainder = len(items) % n_splits
    
    splits = []
    start = 0
    
    for i in range(n_splits):
        # Add one extra item to the first 'remainder' splits
        size = avg_size + (1 if i < remainder else 0)
        end = start + size
        split_dict = dict(items[start:end])
        splits.append(split_dict)
        start = end
    
    return splits

def main():
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--data_dir', type=str, required=False, default="./data", \
                        help="Directory containing the input json files")
    parser.add_argument('--image_dir', type=str, required=False, default="./images", \
                        help="Directory containing the images")
    parser.add_argument('--output_dir', type=str, required=False, default="./results", \
                        help="Directory to save the output json files")
    parser.add_argument('--split', type=str, required=False, choices=['val', 'test'], default='val',
                        help="Split of the data")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['descriptive', 'reasoning'],
                        help="Mode of the evaluation")
    parser.add_argument('--additional_prompt', type=str, default=None, help="Type of additional prompt")
    parser.add_argument('--num_gpus', type=int, default=None, help="Number of GPUs to use (default: auto-detect)")

    # custom arguments
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_api', type=str, required=False, default=None)
    args = parser.parse_args()

    input_file = os.path.join(args.data_dir, f"{args.mode}_{args.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 
            f'gen-{args.model_name}-{args.mode}_{args.split}.json')

    if args.mode == 'descriptive':
        from descriptive_utils import build_descriptive_quries
        queries = build_descriptive_quries(data, args.image_dir)
    elif args.mode == 'reasoning':
        from reasoning_utils import build_reasoning_queries
        queries = build_reasoning_queries(data, args.image_dir)
    else: 
        raise ValueError("Mode not supported")
    
    # add additional prompt
    if args.additional_prompt is not None:
        if args.additional_prompt == 'sft_with_code':
            curr_template = SFT_WIHT_CODE
            for key in queries:
                # print('current prompt:', curr_template.format(question=queries[key]['question']))
                queries[key]['question'] = curr_template.format(question=queries[key]['question'])
        elif args.additional_prompt == 'sft_with_cot':
            curr_template = SFT_WIHT_COT
            for key in queries:
                # print('current prompt:', curr_template.format(question=queries[key]['question']))
                queries[key]['question'] = curr_template.format(question=queries[key]['question'])
        else:
            curr_additional_prompt = ADDITIONAL_PROMPT[args.additional_prompt]
            for key in queries:
                queries[key]['question'] = queries[key]['question'] + ' ' +  curr_additional_prompt

    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", args.mode)
    print("Output file:", output_file)

    # Determine number of GPUs to use
    if args.num_gpus is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = 1
            print("Warning: CUDA not available, using CPU")
    else:
        num_gpus = args.num_gpus
    
    print(f"Using {num_gpus} GPU(s)")

    if num_gpus <= 1:
        # Single GPU or CPU processing (original behavior)
        generate_fn = get_generate_fn(args.model_path)
        if args.model_api is not None:
            client, model = get_client_fn(args.model_path)(args.model_path, args.model_api)
            generate_response_remote_wrapper(generate_fn, queries, model, args.model_api, client)
        else:
            generate_fn(queries, args.model_path)
    else:
        # Multi-GPU processing
        print(f"Splitting dataset into {num_gpus} parts...")
        query_splits = split_dict(queries, num_gpus)
        
        # Create a manager for sharing results between processes
        manager = Manager()
        results_dict = manager.dict()
        
        # Create and start processes for each GPU
        processes = []
        for gpu_id in range(num_gpus):
            if gpu_id < len(query_splits) and len(query_splits[gpu_id]) > 0:
                p = Process(target=run_inference_on_gpu, 
                           args=(gpu_id, query_splits[gpu_id], args.model_path, args.model_api, results_dict))
                p.start()
                processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine results back into queries
        queries.update(dict(results_dict))
        
        print("All GPU processes completed")

    # Clean up queries for output
    for k in queries:
        queries[k].pop("figure_path", None)
        queries[k].pop("question", None)

    try:
        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(queries, f, indent=4)
        print(f"Results saved.")
    except Exception as e:
        print(e)
        print(f"Error in saving {output_file}")

if __name__ == '__main__':
    # Set the start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
