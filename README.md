# Finding Highlight Images In Your Albums: From Benchmark To MLLM

This repository provides the official implementation and LoRA weights for our paper: **"Finding Highlight Images In Your Albums: From Benchmark To MLLM"**.

## Overview
Finding Highlight Images In Your Albums: From Benchmark To MLLM

## Implementation
This project is built based on the **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** framework. We leverage its efficient fine-tuning pipeline to achieve highlight scoring.

## Model Weights (LoRA)
We provide the model training weights based on the PEC dataset, which are hosted on Hugging Face:
- **Repo:** [suncongcong/AHR-PEC](https://huggingface.co/suncongcong/AHR-PEC)

These LoRA weights are specifically trained for the Qwen/Qwen3-VL-8B-Instruct base model. You can load these weights to integrate affective intelligence into your vision-language pipelines.

## Dataset: PEC and CUFED
The PEC and CUFED dataset is available for research purposes:
- **Link:** [https://pan.baidu.com/s/1-kLIYzQKNMLRVCL4NubrlA](https://pan.baidu.com/s/1-kLIYzQKNMLRVCL4NubrlA)
- **Extraction Code:** `qrls`

## Inference
We provide a unified script `infer.py` to evaluate images using our trained model. The script supports distributed inference via `torchrun` and offers two evaluation modes: **Global** and **Local**.
### Local 
```bash
torchrun --nnodes 1 --nproc_per_node 8 infer.py \
    --task local \
    --base_model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --lora_path "AHR-PEC" \
    --test_data_path <path_to_test_data_json> \
    --local_output_dir <path_to_output_directory>
```
### Global 
```bash
torchrun --nnodes 1 --nproc_per_node 8 infer.py \
    --task global \
    --base_model_path "Qwen/Qwen3-VL-8B-Instruct" \
    --lora_path "AHR-PEC" \
    --regression_result <path_to_regression_results_jsonl> \
    --image_root <path_to_dataset_root> \
    --global_output_dir <path_to_output_directory> \
    --ref_count 5
```
