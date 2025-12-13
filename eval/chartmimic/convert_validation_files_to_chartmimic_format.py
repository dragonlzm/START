import json
import ipdb
import re
import argparse
from mathruler.grader import extract_boxed_content, grade_answer
import ipdb
import os

def extract_answer(text):
    """
    Extract content between <answer> and </answer> tags.
    If pattern is found, return the answer content.
    Otherwise, return the full string.
    
    Args:
        text (str): Input string to search for answer pattern
        
    Returns:
        str: Either the extracted answer or the full input string
    """
    # Search for content between <answer> and </answer> tags
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return text


# def extract_python_code(prediction: str) :
#     """
#     Extract the first ```python ... ``` fenced code block from a model prediction.
#     Parameters
#     ----------
#     prediction : str
#         The raw text produced by the model (e.g. one entry from your SFT dataset).
#     Returns
#     -------
#     Optional[str]
#         The code inside the fenced block, stripped of outer whitespace, or
#         None if no Python block is present.
#     """
#     # Regex: look for ```python  ...  ```  (non-greedy, DOTALL = match newlines)
#     m = re.search(r"```python\s*(.*?)```", prediction, flags=re.DOTALL | re.IGNORECASE)
#     return m.group(1).strip() if m else None


def convert_and_eval_the_file(args):
    # 1. load the annotation 
    # 2. assert the order
    # 3. extract the content and save

    train_eval_file_content = json.load(open(args.train_eval_file))
    annotation_file_content = json.load(open(args.annotation_file))
    # ipdb.set_trace()
    # assert the order is correct
    for pred, gt in zip(train_eval_file_content, annotation_file_content): 
        try:
            assert pred['label'] == gt['q_id']
        except:
            ipdb.set_trace()

    # prepare the root 
    save_root = '/'.join(args.save_file.split('/')[:-1])
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # ipdb.set_trace()
    # calculate the acc
    for ele in train_eval_file_content:
        origin_answer = ele['output']
        # gt = ele['label']
        # ipdb.set_trace()
        # try potential extraction from the answer
        if args.answer_format == 'python': 
            # ipdb.set_trace()
            # extracted_answer = extract_python_code(origin_answer) # will return the origin response if extraction fail
            extracted_answer = origin_answer
        elif args.answer_format == 'box_python': 
            # ipdb.set_trace()
            extracted_answer = extract_boxed_content(origin_answer)
            if extracted_answer == "None":
                extracted_answer = origin_answer            
            # extracted_answer = extract_python_code(extracted_answer)
        else:
            raise NotImplementedError
        # save the result
        curr_file_name = ele['label'] + '.pdf'
        curr_file_path = os.path.join(args.image_root, curr_file_name)
        with open(args.save_file, "a", encoding="utf-8") as f:
            json_str = json.dumps(
                {"file": curr_file_path, "response": extracted_answer}
            )
            f.write(json_str + "\n")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_eval_file", type=str, default="./")
    parser.add_argument("--save_file", type=str, default="siglip_features_spool.pkl")
    parser.add_argument("--image_root", type=str, default="siglip_features_spool.pkl")
    parser.add_argument("--annotation_file", type=str, default="siglip_features_spool.pkl")
    parser.add_argument("--answer_format", type=str, default="box")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    print(args)
    convert_and_eval_the_file(args)

# python qwen_exp/evaluation_code/chartmimic/convert_train_validation_files_to_chartmimic_format.py \
# --train_eval_file /home/ec2-user/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/chartmimic_res/val_samples_step_0.json \
# --save_file /home/ec2-user/ChartMimic/results/direct/chart2code_qwen2_5_vl_DirectAgent_results.json \
# --annotation_file /home/ec2-user/generated_annotation/chartmimic.json \
# --image_root /home/ec2-user/ChartMimic/dataset/direct_600 \
# --answer_format python



# need to convert it into
# 1. data["response"] is pure code
# 2. the answer_format is python, boxed python or something else
# 3. need to be save into a json line files
# 4. need image root to fill in the "file" key in the final prediction

# expected format:
# {"file":"\/home\/ec2-user\/ChartMimic\/dataset\/direct_600\/3d_1.pdf",
# "response":"{\"id\":\"chatcmpl-C73GC7s0kFKhBVVHavoJ3DVZaaemN\",\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"message\":{\"content\":\"Sure! Below is the Python code using `matplotlib` to reproduce the 3D plot shown in the image:\\n\\n```python\\nimport matplotlib.pyplot as plt\\nfrom mpl_toolkits.mplot3d import Axes3D\\n\\n# Data for the plot\\ntargets = {\\n    'x': [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6],\\n    'y': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8],\\n    'z': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]\\n}\\n\\npredictions = {\\n    'x': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],\\n    'y': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],\\n    'z': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]\\n}\\n\\n# Create a new figure\\nfig = plt.figure(figsize=(7.0, 7.0))\\nax = fig.add_subplot(111, projection='3d')\\n\\n# Plot the targets\\nax.plot(targets['x'], targets['y'], targets['z'], 'o-', color='orange', label='Targets')\\n\\n# Plot the predictions\\nax.plot(predictions['x'], predictions['y'], predictions['z'], 'o-', color='blue', label='Predictions')\\n\\n# Set labels\\nax.set_xlabel('X')\\nax.set_ylabel('Y')\\nax.set_zlabel('Z')\\n\\n# Add a legend\\nax.legend()\\n\\n# Show the plot\\nplt.show()\\n```\\n\\nThis code will generate a 3D plot with two sets of data points connected by lines, similar to the image you provided. The `figsize` parameter is set to `(7.0, 7.0)` to match the original size.\",\"role\":\"assistant\",\"function_call\":null,\"tool_calls\":null,\"refusal\":null,\"annotations\":[]},\"logprobs\":null}],\"created\":1755796864,\"model\":\"gpt-4o-2024-05-13\",\"object\":\"chat.completion\",\"system_fingerprint\":\"fp_3b6eded18c\",\"usage\":{\"completion_tokens\":695,\"prompt_tokens\":875,\"total_tokens\":1570,\"prompt_tokens_details\":{\"cached_tokens\":0,\"audio_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":0,\"audio_tokens\":0,\"accepted_prediction_tokens\":0,\"rejected_prediction_tokens\":0}},\"service_tier\":\"default\"}"}

# with open(self.results_file + str(rank), "a", encoding="utf-8") as f:
#     json_str = json.dumps(
#         {"file": self.dataset[sub_index[i]]["file"], "response": response}
#     )
#     f.write(json_str + "\n")