## The target reply format:
#   {
#     "index": 0,
#     "input": "system\nYou are a helpful assistant.\n# Tools\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"image_zoom_in_tool\",\"description\":\"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.\",\n\"parameters\":{\"type\":\"object\",\"properties\":{\"coordinate\":{\"type\":\"array\",\"items\":{\"type\":\"int\"},\n\"description\":\"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.\"},'\n\"label\":{\"type\":\"string\",\"description\":\"The name or label of the object in the specified bounding box (optional).\"}},\n\"required\":[\"coordinate\"]}}}\n</tools>\n# How to call a tool\nReturn a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n**Example**:  \n<tool_call>  \n{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"coordinate\": [8, 40, 50, 150], \"label\": \"the legend on the subplot of the right\"}}  \n</tool_call>\nYour task involves multiple turns of reasoning, and each turn has EXACTLY one <think> step and one action. At each turn, think in the mind first, and then decide whether to call tool OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> (if any tools needed) OR <think>...</think> <answer>...</answer> (if no tools needed)\nuser\n\n At around what minute mark does Exp 3 first drop below 200 responses per 60 seconds?\n    * Your final answer must be grounded to a number that is exlicitly written and relevant to the question in the chart, even if it's an approximate value.\n    * You are allowed to extract numbers within some text when needed.\n    \nassistant\n",
#     "output": "<think>Observing the graph, I need to zoom in on the area when \"Exp 3\" drops below 200 responses per 60 seconds. This is about the 40th minute mark where the green line (Exp 3) starts to dip sharply. I can see this part of the data closely by zooming in or focusing on this portion.\n\n</think>\n<tool_call>\n{\"name\": \"image_zoom_in_tool\", \"arguments\": {\"coordinate\": [218, 420, 546, 488], \"label\": \"the green line of Exp 3\"}}\n</tool_call>\nuser\n<observation>\nHere is the crop of the image centered on the coordinate:\n</observation>\nassistant\n<think>Observing the zoomed-in view, it's clear that the green line representing \"Exp 3\" initially stays above 200 until around the 40th minute, and then there's a dramatic drop. The question asks for the first occurrence which is evident at around the 40-minute mark. Based on this, Exp 3 first drops below 200 responses per 60 seconds at around the 40th minute.\n\n</think> \n<answer>40</answer>",
#     "label": "40",
#     "score": 5.0
#   },

# target name:
# gen-Qwen2.5-VL-3B-Instruct-reasoning_val.json 
# target format:
# {
    # "0": {
    #     "figure_id": 0,
    #     "inst_category": 1,
    #     "raw_question": "Which model shows a greater decline in accuracy from Session 1 to Session 9 in the 5-way full-shot scenario?",
    #     "response": "Joint-CNN"
    # },
    # "2": {
    #     "figure_id": 2,
    #     "inst_category": 1,
    #     "raw_question": "What is the name of the line that is the furthest away from its fi value from the W_H axis?",
    #     "response": "\\text{\u03bb_L = 0.30}"
    # },



import json
import ipdb
import re

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



train_eval_file = '/home/ec2-user/CharXiv/results/val_samples_step_35.json'
train_eval_file_content = json.load(open(train_eval_file))

# bulid up a mapping from label to the qa instance for the predictions from the training pipeline
from_label_to_prediction = {}
for ele in train_eval_file_content:
    curr_label = ele['label']
    if curr_label not in from_label_to_prediction:
        from_label_to_prediction[curr_label] = []
    from_label_to_prediction[curr_label].append(ele)

# for each question in the annotation, follow it order to find the answer in the mapping and save it into the json file
annotation_file = "/home/ec2-user/CharXiv/data/reasoning_val.json"
annotation_file_content = json.load(open(annotation_file))

final_prediction_contents = []

for key in annotation_file_content:
    curr_content = annotation_file_content[key]
    curr_figure_id = curr_content["figure_id"]
    curr_inst_category = curr_content["inst_category"]
    curr_query = curr_content["query"]
    curr_gt = curr_content["answer"]

    # find the answer base on the label
    curr_model_answers = from_label_to_prediction[curr_gt]
    if len(curr_model_answers) == 1:
        model_answer = curr_model_answers[0]["output"]
    else:
        # find the answer base on the question
        for pre_ele in curr_model_answers:
            if curr_query in pre_ele["input"]:
                break
        model_answer = pre_ele["output"]
        

    # strip out the model response
    extracted_answer = extract_answer(model_answer)
    ipdb.set_trace()

    curr_prediction_res = {
        "figure_id": curr_figure_id,
        "inst_category":curr_inst_category,
        "raw_question":curr_query,
        "response":extracted_answer,
    }

