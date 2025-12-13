from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm
import torch
import ipdb
from mathruler.grader import extract_boxed_content, grade_answer

def generate_response(queries, model_path, ):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path)

    # min_pixels = 256 * 28 * 28
    # max_pixels = 1280 * 28 * 28
    # processor = AutoProcessor.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    # )

    for k in tqdm(queries):
        # query: 'Which model shows a greater decline in accuracy from Session 1 to Session 9 in the 5-way full-shot scenario?\n    
        # * Your final answer must be grounded to some text that is explicitly written and relevant to the question in the chart.\n    
        # * If you need to answer multiple terms, separate them with commas.\n    * Unless specified in the question (such as answering with a letter), 
        # you are required to answer the full names of subplots and/or labels by default.\n    '
        # queries[k]["figure_path"]: './images/0.jpg'
        # image.size: (1024, 496)
        # ipdb.set_trace() # check the query
        query = queries[k]["question"]
        image = queries[k]["figure_path"]
        image = Image.open(image).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        # text: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Which model shows a greater decline in accuracy from Session 1 to Session 9 in the 5-way full-shot scenario?\n    
        # * Your final answer must be grounded to some text that is explicitly written and relevant to the question in the chart.\n    
        # * If you need to answer multiple terms, separate them with commas.\n    
        # * Unless specified in the question (such as answering with a letter), you are required to answer the full names of subplots and/or labels by default.\n    
        # <|im_end|>\n<|im_start|>assistant\n'
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # video_inputs: None
        # image_inputs: len(image_inputs) == 1, image_inputs[0].size: (1036, 504) -> 522144 (max)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # ipdb.set_trace() # check the response, whether we should extract the content in the box
        box_content = extract_boxed_content(response)
        # ipdb.set_trace()
        
        if box_content == 'None' or box_content is None:
            print('\\box extract error, save the origin response')
            queries[k]["response"] = response
        else:
            queries[k]["response"] = box_content


