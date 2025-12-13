## The target reply format:
# [
#   {
#     "index": 0,
#     "input": "system\nYou are a helpful assistant.\nuser\nFor left-most subplot, how many discrete labels are there in the legend?\n    * Your final answer should account for only labels relevant to the plot in the legend, even if the legend is located outside the plot. \n    * If the plot does not have a legend or no legend is not considered relevant to this plot, answer \"Not Applicable\". You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.\nassistant\n",
#     "output": "<think>\nTo determine the number of discrete labels in the legend, I'll analyze the legend in the image:\n\n1. Identify the elements in the legend:\n   - TensorFlow (green bar with green dots)\n   - XLA (light blue bar with blue dots)\n   - MXNet (green checkmark)\n   - PyTorch (blue bar with blue dots)\n   - Kernel Exec. Time (yellow bar with yellow dots)\n\n2. Count the distinct labels:\n   - TensorFlow\n   - XLA\n   - MXNet\n   - PyTorch\n   - Kernel Exec. Time\n\nThere are 5 distinct labels in the legend.\n\n</think>\n\\boxed{5}",
#     "label": "5",
#     "score": 1.0
#   },

# target name:
# gen-Qwen2.5-VL-3B-Instruct-reasoning_val.json 
# target format:
# {
#     "0_0": {
#         "figure_id": 0,
#         "subq_idx": 0,
#         "qid": 7,
#         "response": "60"
#     },
#     "0_1": {
#         "figure_id": 0,
#         "subq_idx": 1,
#         "qid": 13,
#         "response": "Not Applicable"
#     },
#     "0_2": {
#         "figure_id": 0,
#         "subq_idx": 2,
#         "qid": 11,
#         "response": "No"
#     },

### NOTED: that if the GT are the same, even the mapping from the question in the annotation is not matched with the question in the validation, it will not change the acc

import json
import ipdb
import re
import argparse
from mathruler.grader import extract_boxed_content, grade_answer

DESCRIPTIVE_RESP_INST = {
    1: \
    """{}what is its title?
    * Your final answer should be the most relevant title of the plot that is explicitly written.
    * If the plot does not have an explicit title or contains only a letter, answer 'Not Applicable'.
    """,

    2: \
    """{}what is the label of the x-axis?
    * Your final answer should be the label of the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer the label of the x-axis at the bottom.
    * If the plot does not have an explicit x-axis label, answer 'Not Applicable'.
    """,

    3: \
    """{}what is the label of the y-axis?
    * Your final answer should be the label of the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, answer the label of the y-axis at the left.
    * If the plot does not have an explicit y-axis label, answer 'Not Applicable'.""",

    4: \
    """{}what is the leftmost labeled tick on the x-axis?
    * Your final answer should be the tick value on the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",

    5: \
    """{}what is the rightmost labeled tick on the x-axis?
    * Your final answer should be the tick value on the x-axis that is explicitly written, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",

    6: \
    """{}what is the spatially lowest labeled tick on the y-axis?
    * Your final answer should be the tick value on the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",

    7: \
    """{}what is the spatially highest labeled tick on the y-axis?
    * Your final answer should be the tick value on the y-axis that is explicitly written, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.""",

    8: \
    """{}what is difference between consecutive numerical tick values on the x-axis?
    * Your final answer should be the difference between consecutive numerical tick values of the x-axis, including the case when x-axis is shared across multiple subplots. When the x-axis is present on both the top and bottom of the plot, answer based on the axis at the bottom. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.
    * If the plot does not have an explicit x-axis tick value, or if the tick values are not numerical, or if the difference is not constant between all consecutive tick values, answer "Not Applicable".""",

    9: \
    """{}what is difference between consecutive numerical tick values on the y-axis?
    * Your final answer should be the difference between consecutive numerical tick values of the y-axis, including the case when y-axis is shared across multiple subplots. When the y-axis is present on both the left and right of the plot, answer based on the axis at the left. Ignore units or scales that are written separately from the tick, such as units and scales from the axis label or the corner of the plot.
    * If the plot does not have an explicit y-axis tick value, or if the tick values are not numerical, or if the difference is not constant between all consecutive tick values, answer "Not Applicable".""",

    10: \
    """{}how many lines are there?
    * Your final answer should be the number of lines in the plot. Ignore grid lines, tick marks, and any vertical or horizontal auxiliary lines.
    * If the plot does not contain any lines or is not considered a line plot, answer "Not Applicable".""",

    11: \
    """{}do any lines intersect?
    * Your final answer should be "Yes" if any lines intersect, and "No" otherwise. Ignore grid lines, tick marks, and any vertical or horizontal auxiliary lines.
    * If the plot does not contain any lines or is not considered a line plot, answer "Not Applicable".""",

    12: \
    """{}how many discrete labels are there in the legend?
    * Your final answer should account for only labels relevant to the plot in the legend, even if the legend is located outside the plot. 
    * If the plot does not have a legend or no legend is not considered relevant to this plot, answer "Not Applicable".""",

    13: \
    """{}what are the names of the labels in the legend?
    * You should write down the labels from top to bottom, then from left to right and separate the labels with commas. Your final answer should account for only labels relevant to the plot in the legend, even if the legend is located outside the plot.
    * If the plot does not have a legend or no legend is not considered relevant to this plot, answer "Not Applicable".""",

    14: \
    """{}what is the difference between the maximum and minimum values of the tick labels on the continuous legend (i.e., colorbar)?
    * You should remove the percentage sign (if any) in your answer.
    * If the plot does not have an explicit colorbar-based continuous legend or the legend is not considered relevant to this subplot, answer "Not Applicable".""",

    15: \
    """{}what is the maximum value of the tick labels on the continuous legend (i.e., colorbar)?
    * You should remove the percentage sign (if any) in your answer. 
    * If the plot does not have an explicit colorbar-based continuous legend or the legend is not considered relevant to this subplot, answer "Not Applicable".""",

    16: \
    """{}what is the general trend of data from left to right?
    * Your final answer should be within a few words, such as "increases", "increases then stabilizes".""",

    17: \
    """{}What is the total number of explicitly labeled ticks across all axes?
    * Your final answer should be the total number of explicitly labeled ticks across all axes, including the case when any axis is shared across multiple subplots.""",

    18: \
    """What is the layout of the subplots?
    * Your final answer should follow "n by m" format, where n is the number of rows and m is the number of columns.
    * If the plot does not contain subplots, answer "1 by 1".""",

    19: \
    """What is the number of subplots?
    * Your final answer should be the total number of subplots in the plot.
    * If the plot does not contain subplots, answer "1".""",
}


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


def descriptive_query_helper(qid, subplot_loc):
    if qid in [18, 19]:
        # skip subplot location when asking about the layout of the subplots
        return DESCRIPTIVE_RESP_INST[qid] 
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            # when there is only one subplot
            prefix = "For the current plot, "
        else:
            # when there are multiple subplots
            prefix = f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
    # when subplots do not form a grid
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"Invalid subplot_loc: {subplot_loc}")
    # return the question with the subplot location
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)


def convert_the_file(args):
    train_eval_file_content = json.load(open(args.train_eval_file))
    annotation_file_content = json.load(open(args.annotation_file))

    all_gt_label = []
    for key in annotation_file_content:
        all_gt_label += annotation_file_content[key]['answers']

    for pred, gt in zip(train_eval_file_content, all_gt_label): # assert the order is correct
        assert pred['label'] == gt

    all_predicts = [ele["output"] for ele in train_eval_file_content]
    count = 0
    final_prediction_contents = {}

    for image_key in annotation_file_content:
        curr_content = annotation_file_content[image_key]
        curr_figure_id = curr_content["figure_id"]
        curr_all_q_types = curr_content["qids"]
        curr_subplot_loc = curr_content["subplot_loc"]
        # curr_answer = curr_content["answers"]
        # curr_inst_category = curr_content["inst_category"]
        # curr_all_query = curr_content["query"]
        curr_all_gt = curr_content["answers"]

        # loop over all the questions
        assert len(curr_all_q_types) == len(curr_all_gt)
        for question_idx, (curr_q_type, curr_gt) in enumerate(zip(curr_all_q_types, curr_all_gt)):
            # prepare the query
            # curr_query = descriptive_query_helper(curr_q_type, curr_subplot_loc)
            # find the answer base on the label
            model_answer = all_predicts[count]
            # strip out the model response
            if args.answer_format == 'answer': 
                extracted_answer = extract_answer(model_answer)
            elif args.answer_format == 'box': 
                extracted_answer = extract_boxed_content(model_answer)
                if extracted_answer == "None":
                    extracted_answer = model_answer
            else:
                raise NotImplementedError
            # {
            #     "0_0": {
            #         "figure_id": 0,
            #         "subq_idx": 0,
            #         "qid": 7,
            #         "response": "60"
            #     },

            curr_prediction_res = {
                "figure_id": curr_figure_id,
                "subq_idx": question_idx,
                # "inst_category": curr_q_type,
                "qid": curr_q_type,
                # "raw_question":curr_query,
                "response":extracted_answer,
            }
            count += 1
            qa_key = image_key + '_' + str(question_idx)
            # final_prediction_contents.append(curr_prediction_res)
            final_prediction_contents[qa_key] = curr_prediction_res

    file = open(args.save_file, 'w')
    file.write(json.dumps(final_prediction_contents))
    file.close()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_eval_file", type=str, default="./")
    parser.add_argument("--save_file", type=str, default="siglip_features_spool.pkl")
    parser.add_argument("--annotation_file", type=str, default="siglip_features_spool.pkl")
    parser.add_argument("--answer_format", type=str, default="box")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    print(args)
    convert_the_file(args)

# python qwen_exp/evaluation_code/charxiv_inference/convert_descriptive_validation_file_to_charxiv_format.py \
# --train_eval_file /home/ec2-user/EasyR1/checkpoints/easy_r1/testing_eval_with_training/val_samples_step_0.json \
# --save_file /home/ec2-user/CharXiv/results/gen-Qwen2.5-VL-7B-Instruct-descriptive_val.json \
# --annotation_file /home/ec2-user/CharXiv/data/descriptive_val.json \
# --answer_format box



# prediction_file = '/home/ec2-user/EasyR1/checkpoints/easy_r1/testing_eval_with_training/val_samples_step_0.json'
# prediction_file = '/home/ec2-user/grounded-rl/data/checkpoints/rl/eval_with_train_pipeline/val_samples_step_0.json'
# prediction_content = json.load(open(prediction_file))
# gt_file = '/home/ec2-user/CharXiv/data/descriptive_val.json'
# gt_content = json.load(open(gt_file))

# all_gt_label = []
# for key in gt_content:
#     all_gt_label += gt_content[key]['answers']

# for pred, gt in zip(prediction_content, all_gt_label):
#     if pred['label'] != gt:
#         print(pred)