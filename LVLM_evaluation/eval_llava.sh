#!/bin/bash

python eval.py \
    --model_name LLaVA \
    --device 0 \
    --batch_size 1 \
    --dataset_name TextVQA \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-c4-4bit.pt",w_bits=4,a_bits=32 \
    --eval_vqa

python eval.py \
    --model_name LLaVA \
    --device 0 \
    --batch_size 1 \
    --dataset_name TextVQA \
    --quant_args quant_checkpoint="/home/chengzhang/Multimodal-Quantization/GPTQ-for-LLaMa/models/llava-1.5-7b-textv-4bit.pt",w_bits=4,a_bits=32 \
    --eval_vqa

# python eval.py \
#     --model_name LLaVA \
#     --device 0 \
#     --batch_size 1 \
#     --dataset_name TextVQA \
#     --quant_args w_bits=32,a_bits=32 \
#     --eval_vqa

# python eval.py \
#     --model_name LLaVA \
#     --device 0 \
#     --batch_size 1 \
#     --dataset_name TextVQA \
#     --quant_args w_bits=32,a_bits=4,act_quant_func="lut",act_token_split=1 \
#     --eval_vqa

# python eval.py \
#     --model_name LLaVA \
#     --device 0 \
#     --batch_size 1 \
#     --dataset_name TextVQA \
#     --quant_args w_bits=32,a_bits=4,act_quant_func="lut.hybrid",act_token_split=0 \
#     --eval_vqa

# python eval.py \
#     --model_name LLaVA \
#     --device 0 \
#     --batch_size 1 \
#     --dataset_name TextVQA \
#     --quant_args w_bits=32,a_bits=4,act_quant_func="lut.vision",act_token_split=0 \
#     --eval_vqa

# python eval.py \
#     --model_name LLaVA \
#     --device 0 \
#     --batch_size 1 \
#     --dataset_name TextVQA \
#     --quant_args w_bits=32,a_bits=4,act_quant_func="lut.text",act_token_split=0 \
#     --eval_vqa
