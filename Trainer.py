import os
import subprocess


def main(src_dataset_mode = 0,
    training_num = 10,
    per_step = 50,
    mode = 0,
    model_mode = "v1.4"):

    training_mode_list = ['db_prior', 'db']
    overall_step = training_num * per_step
    if src_dataset_mode == 0:
        src_dataset_path = "wikiart_vangogh"
        data_type = "style"
    elif src_dataset_mode == 1:
        src_dataset_path = "object_dog"
        data_type = "dog"
    if mode == 0:
        for current_style in os.listdir(f"./datasets/{src_dataset_path}/{training_num}"):
            if os.path.exists(f"{training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}"):
                print(f"skip exist output dir {training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}")
                continue
            if not os.path.exists(f"./datasets/class_dataset/{src_dataset_path}"):
                os.makedirs(f"./datasets/class_dataset/{src_dataset_path}")
            if not os.path.exists(f"{training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}"):
                os.makedirs(f"{training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}")
 
            training_script = f"""
            export MODEL_NAME="CompVis/stable-diffusion-v1-4"
            export INSTANCE_DIR="./datasets/{src_dataset_path}/{training_num}/{current_style}/membership"
            export CLASS_DIR="./datasets/class_dataset/{src_dataset_path}"
            export OUTPUT_DIR="{training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}"

            accelerate launch train_dreambooth.py \
            --pretrained_model_name_or_path=$MODEL_NAME  \
            --instance_data_dir=$INSTANCE_DIR \
            --class_data_dir=$CLASS_DIR \
            --output_dir=$OUTPUT_DIR \
            --with_prior_preservation --prior_loss_weight=1.0 \
            --instance_prompt="sks {data_type}" \
            --class_prompt="art {data_type}" \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 \
            --learning_rate=2e-6 \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps={overall_step} \
            --num_class_images={overall_step}\
            --train_text_encoder \
            --checkpointing_steps={overall_step} \
            --use_8bit_adam \
            --gradient_checkpointing 
            """
            subprocess.call(["sh", "-c", training_script])

            # raise ValueError("Break! Current Testing for only one dataset")
    elif mode == 1:
        for current_style in os.listdir(f"./datasets/{src_dataset_path}/{training_num}"):
            if os.path.exists(f"{training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}"):
                print(f"skip exist output dir {training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}")
                continue
            if not os.path.exists(f"./datasets/class_dataset/{src_dataset_path}/{training_num}/{current_style}"):
                os.makedirs(f"./datasets/class_dataset/{src_dataset_path}/{training_num}/{current_style}")
            if not os.path.exists(f"{training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}"):
                os.makedirs(f"{training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}")
 
            training_script = f"""
            export MODEL_NAME="CompVis/stable-diffusion-v1-4"
            
            export INSTANCE_DIR="./datasets/{src_dataset_path}/{training_num}/{current_style}/membership"
            export OUTPUT_DIR="{training_mode_list[mode]}/{src_dataset_path}/{training_num}_{per_step}_{model_mode}/{current_style}"

            accelerate launch train_dreambooth.py \
            --pretrained_model_name_or_path=$MODEL_NAME  \
            --instance_data_dir=$INSTANCE_DIR \
            --output_dir=$OUTPUT_DIR \
            --instance_prompt="A figure" \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 \
            --learning_rate=2e-6 \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps={overall_step} \
            --train_text_encoder \
            --checkpointing_steps={overall_step} \
            --use_8bit_adam 
            """
            subprocess.call(["sh", "-c", training_script])

            # raise ValueError("Break! Current Testing for only one dataset")
            
            
if __name__ == '__main__':
    main(src_dataset_mode = 0, training_num = 10)
    main(src_dataset_mode = 1, training_num = 2)