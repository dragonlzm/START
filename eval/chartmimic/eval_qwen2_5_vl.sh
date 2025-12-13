TASK_NAME=qwen2_5_vl_7b
CONVERSION_FORMAT=python
CHARTMIMIC_RES_ANNO_FILE=eval/chartmimic/chartmimic_annotations.json
CHARTMIMIC_RES_IMAGE_ROOT=eval/chartmimic/ChartMimic/dataset/direct_600
export PROJECT_PATH="/storage/home/zliu2346/start_testing/START/eval/chartmimic/ChartMimic"
template_type=direct;agent_type=DirectAgent
model=qwen2_5_vl


# convert the format
python eval/chartmimic/convert_validation_files_to_chartmimic_format.py \
--train_eval_file predictions/qwen2_5_vl_7b_chartmimic_predictions.json \
--save_file ${PROJECT_PATH}/results/${TASK_NAME}/chart2code_${model}_${agent_type}_results.json \
--annotation_file ${CHARTMIMIC_RES_ANNO_FILE} \
--image_root ${CHARTMIMIC_RES_IMAGE_ROOT} \
--answer_format ${CONVERSION_FORMAT}


# run the code
cd /storage/home/zliu2346/start_testing/START/eval/chartmimic/ChartMimic
python chart2code/utils/post_process/code_checker.py \
    --input_file ${PROJECT_PATH}/results/${TASK_NAME}/chart2code_${model}_${agent_type}_results.json \
    --template_type ${template_type}


# for any case save a image for the corresponding code, even though the code is now running correctly
cd /storage/home/zliu2346/start_testing/START/eval/chartmimic/ChartMimic
python chart2code/utils/post_process/code_interpreter.py \
    --input_file ${PROJECT_PATH}/results/${TASK_NAME}/chart2code_${model}_${agent_type}_results.json \
    --template_type ${template_type}


# low level eval
cd /storage/home/zliu2346/start_testing/START/eval/chartmimic/ChartMimic
evaluation_dir=${PROJECT_PATH}/results/${TASK_NAME}
python chart2code/main.py --cfg eval_configs/direct/code4evaluation.yaml --tasks code4evaluation --model "${model}" --evaluation_dir ${evaluation_dir}


# high level eval
cd /storage/home/zliu2346/start_testing/START/eval/chartmimic/ChartMimic
evaluation_dir=${PROJECT_PATH}/results/${TASK_NAME}
python chart2code/main.py --cfg_path eval_configs/evaluation_direct.yaml --tasks gpt4evaluation --evaluation_dir ${evaluation_dir}


# run the score aggregation
cd /storage/home/zliu2346/start_testing/START/eval/chartmimic/ChartMimic
python chart2code/tasks/get_the_score.py --model ${model} --task_name ${TASK_NAME} --project_path ${PROJECT_PATH}
