# unified_inference.py
import os
import json
import re
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from peft import PeftModel
SCORE_TOKEN_IDS = []

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Inference Script for Qwen3-VL-8B")
    parser.add_argument("--task", type=str, required=True, choices=["local", "global"],
                        help="Choose which inference task to run: 'local' for single image, 'global' for album context.")
    parser.add_argument("--base_model_path", type=str, default="/model/path")
    parser.add_argument("--lora_path", type=str, default="/model/path_lora")
    
    # global
    parser.add_argument("--regression_result", type=str, default="local.jsonl")
    parser.add_argument("--image_root", type=str, default="dataset")
    parser.add_argument("--global_output_dir", type=str, default="/result")
    parser.add_argument("--output_filename", type=str, default="save")
    parser.add_argument("--ref_count", type=int, default=5)
    
    # local
    parser.add_argument("--test_data_path", type=str, default="path/test.json")
    parser.add_argument("--local_output_dir", type=str, default="path/save_results")
    
    return parser.parse_args()

def init_score_token_ids(tokenizer):
    global SCORE_TOKEN_IDS
    scores = ["0", "1", "2", "3", "4", "5"]
    SCORE_TOKEN_IDS = []
    print("\n[Token Check] Mapping scores to token IDs:")
    for s in scores:
        ids = tokenizer.encode(s, add_special_tokens=False)
        tid = ids[-1]
        SCORE_TOKEN_IDS.append(tid)
        print(f"  Score '{s}' -> Token ID: {tid}")
    print("------------------------------------------\n")

def extract_score_from_text(text):
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    if match:
        try:
            val = float(match.group())
            return max(0.0, min(5.0, val))
        except:
            return None
    return None

def init_distributed():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return local_rank, world_size, device
    else:
        return 0, 1, torch.device("cuda:0")

def build_model(base_path, lora_path, device):
    print(f"Loading Base: {base_path}")
    model = AutoModelForVision2Seq.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    
    print(f"Loading LoRA: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    
    init_score_token_ids(tokenizer)
    
    return model, tokenizer, processor

# ==========================================
CAPTION_PROMPT = "Briefly describe the specific event context and semantic meaning of this moment Keep it under 30 words."
TARGET_INSTRUCTION_SUFFIX = (
    "Focusing on composition, lighting, clarity, and emotional impact, compare this last image with the referenced standards. "
    "Rate the suitability of this image as a highlight cover on a scale of 0.0 to 5.0. "
    "Provide the score directly."
)
STANDARD_TEXT = "These images demonstrate the accepted standard for composition, lighting, clarity, and emotional impact in this specific album.\n"

def get_abs_path(image_root, rel_path):
    if rel_path.startswith("/"): return rel_path
    return os.path.join(image_root, rel_path)

@torch.inference_mode()
def generate_caption_base(model, processor, device, img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": CAPTION_PROMPT},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        
        with model.disable_adapter(): 
            outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.7)
        
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        caption = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        print(f"Caption Error {img_path}: {e}")
        return "High quality highlight image."

@torch.inference_mode()
def predict_score_with_lora(model, processor, device, ref_data_list, target_path):
    messages = [{"role": "user", "content": []}]
    images = []

    intro_text = (
        f"Here are {len(ref_data_list)} reference high-quality highlight images selected from the same album, "
        "along with their highlight scores (0.0-5.0) and event context:\n"
    )
    messages[0]["content"].append({"type": "text", "text": intro_text})

    for ref in ref_data_list:
        p, caption, score = ref['path'], ref['caption'], ref['score']
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            messages[0]["content"].append({"type": "image", "image": p})
            text_suffix = f" (Score: {score:.1f})\nAnalysis: {caption}\n"
            messages[0]["content"].append({"type": "text", "text": text_suffix})
        except Exception as e: 
            pass 

    messages[0]["content"].append({"type": "text", "text": "----------------\n" + STANDARD_TEXT})

    try:
        tgt_img = Image.open(target_path).convert("RGB")
        images.append(tgt_img)
        messages[0]["content"].append({"type": "image", "image": target_path})
    except: 
        return None, "read_error", [], 0.0

    final_text = f"\n{STANDARD_TEXT}{TARGET_INSTRUCTION_SUFFIX}"
    messages[0]["content"].append({"type": "text", "text": final_text})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    outputs = model.generate(
        **inputs, max_new_tokens=10, temperature=0.7, top_p=0.9,
        do_sample=True, output_scores=True, return_dict_in_generate=True
    )

    generated_ids = outputs.sequences[0][prompt_len:]
    ans = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    text_pred_score = extract_score_from_text(ans)

    score_dist, weighted_score_ref = [0.0] * 6, 0.0
    if len(outputs.scores) > 0:
        first_step_logits = outputs.scores[0][0]
        probs = F.softmax(first_step_logits, dim=-1)
        raw_probs = [probs[tid].item() if tid < len(probs) else 0.0 for tid in SCORE_TOKEN_IDS]
        total_p = sum(raw_probs)
        if total_p > 0:
            score_dist = [p / total_p for p in raw_probs]
            weighted_score_ref = sum(idx * p for idx, p in enumerate(score_dist))

    return text_pred_score, ans, score_dist, weighted_score_ref

def run_global_task(args, model, processor, local_rank, world_size, device):
    if local_rank == 0: 
        print(f"Starting GLOBAL Inference (Base->Caption, LoRA->Score+Dist)...")
        if not os.path.exists(args.global_output_dir): os.makedirs(args.global_output_dir)
    
    data = []
    with open(args.regression_result, 'r') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
    df = pd.DataFrame(data)
    
    all_album_ids = df['album_id'].unique()
    my_album_ids = [aid for i, aid in enumerate(all_album_ids) if i % world_size == local_rank]
    out_path = os.path.join(args.global_output_dir, f"{args.output_filename}_rank{local_rank}.jsonl")
    
    with open(out_path, 'w', encoding='utf-8') as f_out:
        for aid in tqdm(my_album_ids, desc=f"Rank {local_rank}"):
            group = df[df['album_id'] == aid]
            sorted_items = group.sort_values(by='weighted_score', ascending=False).to_dict('records')
            
            if len(sorted_items) <= args.ref_count:
                for item in sorted_items:
                    item.update({
                        'final_score': item.get('weighted_score', 0.0), 
                        'source': 'ref_fallback', 'gen_text': 'REF',
                        'raw_text_ref': 'REF', 'score_dist_ref': [], 'weighted_score_ref': 0.0
                    })
                    f_out.write(json.dumps(item) + "\n")
                continue

            refs = sorted_items[:args.ref_count]
            cands = sorted_items[args.ref_count:]
            
            ref_data_list = []
            for r in refs:
                path = get_abs_path(args.image_root, f"{r['album_id']}/{r['image_id']}")
                caption = generate_caption_base(model, processor, device, path)
                ref_score = r.get('weighted_score', 3.0) 
                ref_data_list.append({'path': path, 'caption': caption, 'score': ref_score})
                
                r.update({
                    'final_score': 100.0 + ref_score, 'source': 'ref', 'gen_text': caption,
                    'raw_text_ref': 'REF', 'score_dist_ref': [], 'weighted_score_ref': ref_score
                })

            processed_cands = []
            for c in cands:
                target_path = get_abs_path(args.image_root, f"{c['album_id']}/{c['image_id']}")
                try:
                    text_score, gen_text, score_dist, w_score_ref = predict_score_with_lora(
                        model, processor, device, ref_data_list, target_path
                    )
                    c.update({'raw_text_ref': gen_text, 'score_dist_ref': score_dist, 'weighted_score_ref': w_score_ref, 'gen_text': gen_text})
                    
                    if text_score is not None:
                        c['final_score'], c['source'] = text_score, 'model_text_score'
                    elif w_score_ref > 0:
                        c['final_score'], c['source'] = w_score_ref, 'model_dist_score'
                    else:
                        c['final_score'], c['source'] = c.get('weighted_score', 0.0), 'fallback_weighted'
                        
                except Exception as e:
                    print(f"Err: {e}")
                    c.update({'final_score': 0.0, 'gen_text': 'ERR', 'source': 'err'})
                
                processed_cands.append(c)

            processed_cands.sort(key=lambda x: x['final_score'], reverse=True)
            final_list = refs + processed_cands
            for item in final_list:
                f_out.write(json.dumps(item) + "\n")
            f_out.flush()
    print(f"Rank {local_rank} Finished.")

# ==========================================
@torch.inference_mode()
def predict_local(model, tokenizer, processor, device, img_path, prompt):
    image = Image.open(img_path).convert("RGB")
    if "<image>" not in prompt.lower(): prompt = "<image>\n" + prompt.strip()

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[-1]

    outputs = model.generate(
        **inputs, max_new_tokens=10, temperature=0.7, top_p=0.9,
        output_scores=True, return_dict_in_generate=True 
    )

    generated_ids = outputs.sequences[0][prompt_len:]
    ans = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    score_distribution, weighted_score = [0.0] * 6, 0.0 
    
    if len(outputs.scores) > 0:
        first_step_logits = outputs.scores[0][0]
        probs = F.softmax(first_step_logits, dim=-1)
        raw_probs = [probs[tid].item() for tid in SCORE_TOKEN_IDS]
        total_p = sum(raw_probs)
        if total_p > 0:
            score_distribution = [p / total_p for p in raw_probs]
            weighted_score = sum(idx * p for idx, p in enumerate(score_distribution))
        else:
            try: weighted_score = float(ans)
            except: weighted_score = 0.0

    return ans, score_distribution, weighted_score

def run_local_task(args, model, tokenizer, processor, local_rank, world_size, device):
    if local_rank == 0:
        print("Starting LOCAL Inference (Single Image Score)...")
        if not os.path.exists(args.local_output_dir): os.makedirs(args.local_output_dir)
        
    output_file = os.path.join(args.local_output_dir, f"result_rank_n{local_rank}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f: pass

    with open(args.test_data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for idx, sample in enumerate(tqdm(dataset, disable=local_rank != 0, desc=f"Rank{local_rank}")):
        if idx % world_size != local_rank: continue

        user_prompt = sample.get("instruction", "") + "\n" + sample.get("input", "")
        img_path    = sample.get("images", [])[0]
        gt          = sample.get("output", "").strip()
        
        try:
            album_id = img_path.split('/')[-2]
            image_id = img_path.split('/')[-1]
        except:
            album_id, image_id = "unknown", "unknown"

        ans, score_dist, w_score = predict_local(model, tokenizer, processor, device, img_path, user_prompt)

        result_item = {
            "album_id": album_id, "image_id": image_id,
            "raw_text": ans, "prediction": ans, 
            "score_dist": score_dist, "weighted_score": w_score, 
            "ground_truth": gt
        }
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    if local_rank == 0:
        print("Done. Check weighted_score in output files.")

# ==========================================
def main():
    args = parse_args()
    local_rank, world_size, device = init_distributed()
    
    model, tokenizer, processor = build_model(args.base_model_path, args.lora_path, device)
    
    if args.task == "global":
        run_global_task(args, model, processor, local_rank, world_size, device)
    elif args.task == "local":
        run_local_task(args, model, tokenizer, processor, local_rank, world_size, device)

if __name__ == "__main__":
    main()