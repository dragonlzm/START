import json
import os
import pandas as pd
import re
from pathlib import Path
import ipdb
from typing import Optional, Union, List, Dict, Any

models = ["gpt-4-vision-preview"]
agents = ["DirectAgent"]
project_path = '/home/ec2-user/ChartMimic'

def get_code_passed_files(modelagent):
    template_type = modelagent.split("_")[-1].split("Agent")[0].lower()
    print(template_type)
    file_dir = project_path + "/results/direct/chart2code_{}_results/{}_checker".format(modelagent, template_type)
    filter_files = os.listdir(file_dir)
    filter_files = [ item.split(".pdf")[0]+".py" for item in filter_files if ".pdf" in item]
    print(len(filter_files))
    return filter_files

table_type = "all" # "all". "prompting"

model_agents = [ "{}_{}".format(model, agent) for model in models for agent in agents ]

filter_type = "code_pass"
# level = "hard"
denominator = 600

if filter_type == "no_filter":
    filter_files_dict = None
elif filter_type == "code_pass":
    filter_files_dict = { model_agent: get_code_passed_files(model_agent) for model_agent in model_agents}
else:
    raise ValueError("filter_type not supported")

result_df = pd.DataFrame(columns=["model_agent", "example_count","ExecRate", "TextScore", "LayoutScore", "TypeScore", "ColorScore", "Average", "GPT4VScore", "Overall"])

# insert the model name
result_df["model_agent"] = [ model + "_" + agent for model in models for agent in agents]
# set the index to be the model name
result_df.set_index("model_agent", inplace=True)


files = []
for model in models:
    for agent in agents:
        filename =  project_path + "/results/direct/chart2code_" + model + "_" + agent +"_results_code4evaluation.json"
        if os.path.exists(filename):
            files.append(filename)
        else:
            raise FileNotFoundError("File not found: {}".format(filename))

for idx, file in enumerate(files):
    print("Processing file:", os.path.basename(file))
    
    data = pd.read_json(file, lines=True)
    data["orginial"] = data["orginial"].apply(lambda x: x.split("/")[-1])
    data["generated"] = data["generated"].apply(lambda x: x.split("/")[-1])

    # filter_files = get_code_passed_files(model_agents[idx])
    if filter_files_dict is not None:
        filter_files = filter_files_dict[model_agents[idx]]
        # print(data["orginial"])
        data = data[ data["orginial"].apply(lambda x: any([item == x for item in filter_files])) ]
        print("Length of filter files:", len(filter_files))
        

    print("Length of Data:", len(data))
    print("Denominator:", denominator)

    f1s = []

    result_df.loc[model_agents[idx], "example_count"] = len(data)
    result_df.loc[model_agents[idx], "ExecRate"] = len(filter_files) / denominator
    print("Execution Rate:", len(filter_files) / denominator)


    text_metrics = data["text_metrics"]
    avg_f1 = text_metrics.apply(lambda x: x["f1"]).sum()*100 / denominator
    print(avg_f1)
    result_df.loc[model_agents[idx], "TextScore"] = avg_f1
    f1s.append(avg_f1)

    layout_metrics = data['layout_metrics']
    avg_f1 = layout_metrics.apply(lambda x: x["f1"]).sum()*100 / denominator
    print(avg_f1)
    result_df.loc[model_agents[idx], "LayoutScore"] = avg_f1
    f1s.append(avg_f1)

    chart_type_metrics = data["chart_type_metrics"]
    avg_f1 = chart_type_metrics.apply(lambda x: x["f1"]).sum()*100 / denominator
    print(avg_f1)
    result_df.loc[model_agents[idx], "TypeScore"] = avg_f1
    f1s.append(avg_f1)

    color_metrics = data["color_metrics"]
    avg_f1 = color_metrics.apply(lambda x: x["f1"]).sum()*100 / denominator
    print(avg_f1)
    result_df.loc[model_agents[idx], "ColorScore"] = avg_f1
    f1s.append(avg_f1)

    print( sum(f1s)/len(f1s) )

    result_df.loc[model_agents[idx], "Average"] = sum(f1s)/len(f1s)

    print()


files = []
for model in models:
    for agent in agents:
        file =  project_path + "/results/direct/chart2code_" + model + "_" + agent +"_results_gpt4v.json"
        if os.path.exists(file):
            files.append(file)


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
        
        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}$$])', r'\1', json_str)
        
        return json_str
    
    def try_parse_json(json_str: str):
        """Attempt to parse JSON string with error handling"""
        try:
            cleaned_json = clean_json_string(json_str)
            return json.loads(cleaned_json)
        except (json.JSONDecodeError, ValueError) as e:
            # print(f"JSON parsing error: {e}")
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
            
            # print(f"JSON successfully saved to: {file_path}")
            return True
            
        except (IOError, OSError, PermissionError) as e:
            # print(f"Error saving JSON to {file_path}: {e}")
            return False
        except Exception as e:
            # print(f"Unexpected error while saving JSON: {e}")
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
            r'($$[\s\S]*$$)',
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
            # print(f"Error in {method.__name__}: {e}")
            continue
    
    # If JSON was successfully extracted and target_save_path is provided, save it
    # if extracted_json is not None and target_save_path is not None:
    #     save_success = save_json_to_file(extracted_json, target_save_path)
    #     if not save_success:
            # print("Warning: JSON was extracted but could not be saved to file.")
    
    return extracted_json

def extract_score_value(text):
    pattern = r'Score:\s*(\d+)/\d+'
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


for idx, file in enumerate(files):
    print(os.path.basename(file))
    
    data = pd.read_json(file, lines=True)
    # data["orginial"] = data["orginial"].apply(lambda x: x.split("/")[-1])
    # data["generated"] = data["generated"].apply(lambda x: x.split("/")[-1])

    # if filter_files_dict is not None:
    #     filter_files = filter_files_dict[model_agents[idx]]
    #     data = data[ data["orginial"].apply(lambda x: any([item == x for item in filter_files])) ]
    # print(len(data))
    all_res = []
    for ele in data['response']:
        # ipdb.set_trace()
        curr_json = extract_json_block(ele)
        curr_content = curr_json['choices'][0]['message']['content']
        score = extract_score_value(curr_content)
        all_res.append(score)
    # result_df.loc[model_agents[idx], "example_count"] = len(data)
    # gpt4v_score = data["gpt4v_score"]
    avg_gpt4v_score = sum(all_res) / denominator
    result_df.loc[model_agents[idx], "GPT4VScore"] = avg_gpt4v_score
    print(avg_gpt4v_score)
    print()

# calculate the overall score
result_df["Overall"] = result_df[["Average", "GPT4VScore"]].mean(axis=1)
result_df["ExecRate"] = result_df["ExecRate"] * 100
print(result_df)

result_df.to_csv('results.csv', index=True)