import sys
sys.path.append('/root/VideoLLaMA2')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from videollama2.mm_utils import expand2square

import os
import gc
import torch
import argparse
import base64
import numpy as np

from config import *
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from torch.utils.data import DataLoader
from eval.create_evaluator import Evaluator
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoProcessor, AutoModel, AutoTokenizer, LlavaForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
from utils.utils import *
from datasets import load_dataset
from pathlib import Path


def load_custom_dataset(parquet_path):
    dataset = load_dataset("parquet", data_files=parquet_path)
    return dataset["train"]

   
def test(args):
    # Accelerator for DDP, FSDP, DeepSpeed, etc [Should First Call]
    accel = Accelerator()

    # 일단 llava만 불러오기
    if args.model == "llava":
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).cuda()
        processor = AutoProcessor.from_pretrained(model_id)
    elif args.model == "internVL2":
        path = "OpenGVLab/InternVL2-8B"
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    elif args.model == "IXC2b5":
        ckpt_path = "internlm/internlm-xcomposer2d5-7b"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        model.tokenizer = tokenizer
        model = model.eval()
    elif args.model == "blip2":
        model_id = "Salesforce/blip2-opt-6.7b"  # BLIP-2 model ID
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        processor = Blip2Processor.from_pretrained(model_id)
    elif args.model == "videollama2":
        disable_torch_init()
        modal = 'image'
        model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
        model, processor, tokenizer = model_init(model_path)
        model = model.to('cuda')
        model = model.eval()
    elif args.model == 'instructblip':
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    model.eval()

    # Initialize dataset & evaluator
    # test_dataset = load_dataset("topyun/SPARK", split="train", cache_dir=args.dataset_dir)
    custom_dataset_path = "/content/drive/MyDrive/backup/spark2_new.parquet"
    test_dataset = load_custom_dataset(custom_dataset_path)
    evaluator = Evaluator(root=args.dataset_dir)

    # Update dataset & evaluator
    evaluator.reset()
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                collate_fn=lambda x: x)

    # Accel distributed
    test_dataloader = accel.prepare(test_dataloader)

    # progress bar
    prog_bar = tqdm(enumerate(test_dataloader), disable=not accel.is_local_main_process, total=len(test_dataloader))
    # eval start
    for batch_ind, inputs in prog_bar:

        # memory deallocation
        gc.collect()

        # removing cache
        torch.cuda.empty_cache()
        
        if args.model == "llava":
            all_predictions =[]
            for x in inputs:
                conversation = [
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": x['question_query']},
                        {"type": "image"},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

                raw_image = x['image'] 
                input = processor(prompt, raw_image, return_tensors='pt').to("cuda").to(torch.float16)

                output = model.generate(**input, max_new_tokens=64, do_sample=False)
                answer = processor.decode(output[0][2:], skip_special_tokens=True).split("ASSISTANT: ")[-1]
                all_predictions.append(answer)
        elif args.model == "internVL2":
            pixel_values = [load_image(x['image'], max_num=12).to(torch.bfloat16).cuda() for x in inputs]
            num_patches_list = [x.size(0) for x in pixel_values]
            pixel_values = torch.cat(pixel_values, dim = 0)
            questions = [x['question_query'] for x in inputs]
            
            generation_config = dict(max_new_tokens=1024, do_sample=False)
            
            responses = model.batch_chat(tokenizer, pixel_values,
                            num_patches_list=num_patches_list,
                            questions=questions,
                            generation_config=generation_config)
            all_predictions = responses
        elif args.model == "IXC2b5":
            all_predictions = []
            for x in inputs:
                query = '<ImageHere>'+x['question_query']
                image = [x['image']]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
                all_predictions.append(response)
        elif args.model == "blip2":
            all_predictions = []
            for x in inputs:
                question = x['question_query']
                raw_image = x['image']

                # Prepare inputs for BLIP-2
                inputs_ = processor(images=raw_image, text=question, return_tensors="pt").to("cuda").to(torch.float16)
                output = model.generate(**inputs_, max_new_tokens=1024)
                answer = processor.decode(output[0], skip_special_tokens=True)
                all_predictions.append(answer)
        elif args.model == "videollama2":
            all_predictions = []
            for x in inputs:
                question = x['question_query']
                raw_image = np.array(x['image'])

                output = mm_infer(processor[modal](raw_image), question, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
                all_predictions.append(output)
        elif args.model == 'instructblip':
            all_predictions = []
            for x in inputs:
                inputs_ = processor(images=x['image'], text=x['question_query'], return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs_,
                    do_sample=False,
                    num_beams=5,
                    max_length=256,
                    min_length=1,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )
                answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                all_predictions.append(answer) 

        evaluator.process(inputs, all_predictions)

        # garbage collection
        torch.cuda.empty_cache()
    
    print(f"[Device: {accel.device}] Finished!")
    accel.wait_for_everyone()
    # memory opt
    memory_optimization()

    # evaluate on dataset
    evaluator.evaluate(args.model, accel)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--model', default='llava', type=str, help='llava|internVL2|IXC2b5')
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()

    # test
    test(args)

