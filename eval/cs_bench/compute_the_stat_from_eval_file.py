# This script load the prediction result and calculate the final iou.

import json
import argparse

from mathruler.grader import extract_boxed_content, grade_answer
import re
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ipdb
from typing import Optional, Union, List, Dict, Any
import re
import json
from pathlib import Path


def extract_coordinates_from_text(text):
    """Extract coordinates from the JSON text in gt or response fields."""
    try:
        # ipdb.set_trace()
        # Remove the markdown code block markers
        text = text.strip()
        if text.startswith('``json') and text.endswith('``'):
            text = text[6:-2].strip()
        elif text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()

        # Parse the JSON
        data = json.loads(text)

        if isinstance(data, list) and len(data) > 0:
            item = data[0]

            # Extract coordinates based on the type
            if 'bbox_2d' in item:
                return item['bbox_2d'], 'bbox'
            elif 'point' in item:
                point = item['point']
                if isinstance(point[0], list):  # Line case: [[x1, y1], [x2, y2]]
                    return point, 'line'
                else:  # Dot case: [x, y]
                    return point, 'dot'

        return None, None
    except:
        return None, None

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def calculate_l2_distance(point1, point2):
    """Calculate L2 distance between two points."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def match_line_points(gt_points, pred_points):
    """Find the best matching between two sets of line points."""
    if len(gt_points) != len(pred_points):
        return None, None

    # Calculate distance matrix
    distances = cdist(gt_points, pred_points)

    # Find optimal assignment using Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distances)

    matched_gt = [gt_points[i] for i in row_indices]
    matched_pred = [pred_points[i] for i in col_indices]

    return matched_gt, matched_pred


def extract_json_block(text: str, target_save_path: Optional[str] = None):
    """
    Extract JSON block from a string and optionally save it to a file.
    Supports two formats:
    1. JSON wrapped in ```json ... ``` code blocks
    2. JSON directly in the string without code block wrapper

    Args:
        text (str): Input string containing JSON data
        target_save_path (Optional[str]): Path to save the extracted JSON file.
                                        If None, no file will be saved.

    Returns:
        Optional[Union[List, Dict]]: Parsed JSON data or None if extraction fails
    """

    def clean_json_string(json_str: str):
        """Clean and prepare JSON string for parsing"""
        # Remove leading/trailing whitespace
        json_str = json_str.strip()

        # Fix double braces issue - replace {{ with { and }} with }
        # But be careful not to replace valid escaped braces in strings
        json_str = re.sub(r'\{\{', r'{', json_str)
        json_str = re.sub(r'\}\}', r'}', json_str)

        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        return json_str

    def try_parse_json(json_str: str):
        """Attempt to parse JSON string with error handling"""
        try:
            cleaned_json = clean_json_string(json_str)
            return json.loads(cleaned_json)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error: {e}")
            return None

    def save_json_to_file(json_data: Union[List, Dict], file_path: str):
        """Save JSON data to file with error handling"""
        try:
            # Create directory if it doesn't exist
            path_obj = Path(file_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Save JSON with proper formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            print(f"JSON successfully saved to: {file_path}")
            return True

        except (IOError, OSError, PermissionError) as e:
            print(f"Error saving JSON to {file_path}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error while saving JSON: {e}")
            return False

    # Method 1: Try to extract from ```json``` code blocks
    def extract_from_code_block(text: str) -> Optional[Union[List, Dict]]:
        """Extract JSON from ```json``` code blocks"""
        # Pattern to match ```json ... ``` blocks
        pattern = r'```json\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            result = try_parse_json(match)
            if result is not None:
                return result

        # Also try without newlines after ```json
        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            result = try_parse_json(match)
            if result is not None:
                return result

        return None

    # Method 2: Try to extract JSON directly from string
    def extract_direct_json(text: str) -> Optional[Union[List, Dict]]:
        """Extract JSON directly from string"""
        text = text.strip()

        # Look for JSON array or object patterns
        json_patterns = [
            # Array pattern
            r'(\[[\s\S]*\])',
            # Object pattern
            r'(\{[\s\S]*\})',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                result = try_parse_json(match)
                if result is not None:
                    return result

        # If no brackets found, try parsing the entire string
        result = try_parse_json(text)
        if result is not None:
            return result

        return None

    # Method 3: Advanced extraction for edge cases
    def extract_advanced(text: str) -> Optional[Union[List, Dict]]:
        """Advanced extraction for complex cases"""
        # Try to find the start and end of JSON structure
        start_chars = ['{', '[']
        end_chars = ['}', ']']

        for start_char, end_char in zip(start_chars, end_chars):
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue

            # Find matching closing bracket/brace
            bracket_count = 0
            end_idx = -1

            for i in range(start_idx, len(text)):
                if text[i] == start_char:
                    bracket_count += 1
                elif text[i] == end_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i
                        break

            if end_idx != -1:
                potential_json = text[start_idx:end_idx + 1]
                result = try_parse_json(potential_json)
                if result is not None:
                    return result

        return None

    # Try extraction methods in order
    methods = [
        extract_from_code_block,
        extract_direct_json,
        extract_advanced
    ]

    extracted_json = None

    for method in methods:
        try:
            result = method(text)
            if result is not None:
                extracted_json = result
                break
        except Exception as e:
            print(f"Error in {method.__name__}: {e}")
            continue

    # If JSON was successfully extracted and target_save_path is provided, save it
    if extracted_json is not None and target_save_path is not None:
        save_success = save_json_to_file(extracted_json, target_save_path)
        if not save_success:
            print("Warning: JSON was extracted but could not be saved to file.")

    return extracted_json




def extract_and_calc_the_acc_iou(curr_pred):
    """
    Extract answer and calculate IoU from prediction result.

    Args:
        curr_pred: Dictionary with keys:
            - predicted_answer: str - Model's generated answer
            - ground_truth_answer: str - Ground truth answer
            - ground_truth_location: List[int] - Ground truth bbox [x1, y1, x2, y2]
            - question_type: str - Type of question (e.g., 'bbox')

    Returns:
        correct: bool - Whether answer is correct
        curr_iou: float or None - IoU score if bbox question, else None
    """
    correct = False
    curr_iou = None

    # Extract predicted answer
    predicted_text = curr_pred.get("predicted_answer", "")
    ground_truth = curr_pred.get("ground_truth_answer", "")
    gt_location = curr_pred.get("ground_truth_location")

    # The answer will appear in two potential location
    # 1. directly in the box
    # 2. inside the json answer

    # Extract from boxed content
    test_extract = extract_boxed_content(predicted_text)
    if test_extract == "None":  # No boxed content, use original answer
        test_extract = predicted_text

    if "```json" not in test_extract:  # Directly generated answer
        answer = test_extract
    else:  # JSON block, extract the json
        try:
            answer_json = extract_json_block(test_extract)
            answer = answer_json[0]['answer']
        except:  # Could not extract json, use original string
            try:
                answer = answer_json['answer']
            except:
                answer = test_extract

    answer = str(answer)
    if grade_answer(answer, ground_truth):
        correct = True

    # Extract the location directly from raw prediction for bbox questions
    if gt_location is not None:
        try:
            json_content = extract_json_block(predicted_text)
            if json_content is not None:
                try:
                    # Try list format first
                    predicted_bbox = json_content[0]["bbox_2d"]
                except:
                    try:
                        # Try dict format
                        predicted_bbox = json_content["bbox_2d"]
                    except:
                        predicted_bbox = None

                if predicted_bbox is not None and len(predicted_bbox) == 4:
                    curr_iou = calculate_iou(predicted_bbox, gt_location)
                else:
                    curr_iou = 0.0
            else:
                curr_iou = 0.0
        except:
            curr_iou = 0.0

    return correct, curr_iou


def analyze_predictions(predictions):
    """Analyze the prediction results.

    Args:
        predictions: List of prediction dictionaries with keys:
            - predicted_answer: str - Model's generated answer
            - ground_truth_answer: str - Ground truth answer
            - ground_truth_location: List[int] or None - Ground truth bbox
            - question_type: str - Type of question

    Returns:
        Dictionary with analysis results
    """
    results = {
        'bbox': {'total': 0, 'extracted': 0, 'ious': []},
        'question': {'total': 0, 'result': [], 'ious': []}
    }

    for curr_pred in predictions:
        ground_truth = curr_pred.get('ground_truth_answer', '')

        # Check if this is a pure grounding task (ground truth starts with ```json)
        if ground_truth.startswith('```json'):
            gt_coords, gt_type = extract_coordinates_from_text(ground_truth)

            predicted_text = curr_pred.get('predicted_answer', '')
            extracted_content = extract_boxed_content(predicted_text)
            if extracted_content == 'None':
                extracted_content = predicted_text

            pred_coords, pred_type = extract_coordinates_from_text(extracted_content)

            if gt_type == 'bbox':
                results['bbox']['total'] += 1
                # Check if extraction was successful
                if gt_coords is not None and pred_coords is not None and gt_type == pred_type:
                    try:
                        # Calculate IoU for bbox
                        iou = calculate_iou(gt_coords, pred_coords)
                        results['bbox']['ious'].append(iou)
                        results['bbox']['extracted'] += 1
                    except:
                        continue
        else:
            # This is a QA task
            correct, curr_iou = extract_and_calc_the_acc_iou(curr_pred)
            results['question']['total'] += 1
            results['question']['result'].append(correct)
            results['question']['ious'].append(curr_iou)

    return results

def print_results(results):
    """Print the analysis results."""
    print("="*60)
    print("PREDICTION ANALYSIS RESULTS")
    print("="*60)

    # print the acc
    print('acc:', sum(results['question']['result']) / len(results['question']['result']))

    # print the iou
    filtered_iou = [ele for ele in results['question']['ious'] if ele is not None] + results['bbox']['ious']
    iou_larger_05 = [ele for ele in filtered_iou if ele >= 0.5]
    iou_larger_03 = [ele for ele in filtered_iou if ele >= 0.3]
    print('average iou:', sum(filtered_iou) / len(filtered_iou) , 'recall@0.5:', len(iou_larger_05) / len(filtered_iou) , 'recall@0.3:', len(iou_larger_03) / len(filtered_iou))

def main(args):
    """Main function to compute statistics from evaluation file.

    Args:
        args: Command-line arguments with:
            - pred_res_path: Path to prediction results JSON from inference_with_vllm.py
            - save_res_path: Path to save computed statistics
    """
    pred_res_path = args.pred_res_path
    save_res_path = args.save_res_path

    # Load prediction results
    print(f"Loading predictions from {pred_res_path}")
    with open(pred_res_path, 'r') as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions")

    # Analyze the predictions
    results = analyze_predictions(predictions)

    # Print results
    print_results(results)

    # Save detailed results to file
    with open(save_res_path, 'w') as f:
        # Convert metrics to serializable format
        serializable_results = {}
        for q_type, metrics in results.items():
            serializable_results[q_type] = {
                'total': metrics['total']
            }

            # Add extracted field only if it exists (bbox type has it, question type doesn't)
            if 'extracted' in metrics:
                serializable_results[q_type]['extracted'] = metrics['extracted']
                serializable_results[q_type]['success_rate'] = metrics['extracted'] / metrics['total'] * 100 if metrics['total'] > 0 else 0

            if q_type == 'bbox' and metrics['ious']:
                serializable_results[q_type]['avg_iou'] = float(np.mean(metrics['ious']))
                serializable_results[q_type]['iou_std'] = float(np.std(metrics['ious']))
                serializable_results[q_type]['recall@0.5'] = len([iou for iou in metrics['ious'] if iou >= 0.5]) / len(metrics['ious']) * 100
                serializable_results[q_type]['recall@0.3'] = len([iou for iou in metrics['ious'] if iou >= 0.3]) / len(metrics['ious']) * 100

            if q_type == 'question':
                # Calculate accuracy
                if metrics['result']:
                    serializable_results[q_type]['accuracy'] = sum(metrics['result']) / len(metrics['result']) * 100

                # Calculate IoU metrics for questions with location grounding
                question_ious = [iou for iou in metrics['ious'] if iou is not None]
                if question_ious:
                    serializable_results[q_type]['avg_iou'] = float(np.mean(question_ious))
                    serializable_results[q_type]['iou_std'] = float(np.std(question_ious))
                    serializable_results[q_type]['recall@0.5'] = len([iou for iou in question_ious if iou >= 0.5]) / len(question_ious) * 100
                    serializable_results[q_type]['recall@0.3'] = len([iou for iou in question_ious if iou >= 0.3]) / len(question_ious) * 100

        json.dump(serializable_results, f, indent=2)

    print(f"\nDetailed results saved to {save_res_path}")


# sample usage:
# python eval/cs_bench/compute_the_stat_from_eval_file.py \
#     --pred_res_path /home/zliu2346/start_testing/START/predictions/cs_bench_predictions.json \
#     --save_res_path /home/zliu2346/start_testing/START/predictions/cs_bench_statistics.json

# python eval/cs_bench/compute_the_stat_from_eval_file.py \
#     --pred_res_path /home/zliu2346/start_testing/START/predictions/cs_bench_predictions_start.json \
#     --save_res_path /home/zliu2346/start_testing/START/predictions/cs_bench_statistics_start.json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics from CS Bench prediction results")
    parser.add_argument(
        '--pred_res_path',
        type=str,
        required=True,
        help="Path to prediction results JSON from inference_with_vllm.py"
    )
    parser.add_argument(
        '--save_res_path',
        type=str,
        required=True,
        help="Path to save computed statistics JSON"
    )
    args = parser.parse_args()
    main(args)
