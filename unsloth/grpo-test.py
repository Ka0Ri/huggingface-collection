from unsloth import FastModel
import torch
import re
from datasets import load_dataset


max_seq_length = 1024

# fourbit_models = [
#     # 4bit dynamic quants for superior accuracy and low memory use
#     "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
#     "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
#     "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
#     "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

#     # Other popular models!
#     "unsloth/Llama-3.1-8B",
#     "unsloth/Llama-3.2-3B",
#     "unsloth/Llama-3.3-70B",
#     "unsloth/mistral-7b-instruct-v0.3",
#     "unsloth/Phi-4",
# ] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
    cache_dir="/data/AISeed/huggingface",
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


# # model_id = "HuggingFaceTB/SmolVLM-Instruct"
# model_id = "google/gemma-3-4b-it"
# model_id = "HuggingFaceTB/SmolVLM-Instruct" # Use this for 4bit quantization

# BitsAndBytesConfig int-4 config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, 
#     bnb_4bit_use_double_quant=True, 
#     bnb_4bit_quant_type="nf4", 
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model = Gemma3ForConditionalGeneration.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     quantization_config=bnb_config,
#     _attn_implementation="flash_attention_2",
#     cache_dir="/data/AISeed/huggingface",
# )
# tokenizer = AutoTokenizer.from_pretrained(model_id)


# # Configure LoRA
# peft_config = LoraConfig(
#     r=8,
#     lora_alpha=8,
#     lora_dropout=0.1,
#     # target_modules=["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"],
#     target_modules=["q_proj", "v_proj"],
#     use_dora=True,
#     task_type="CAUSAL_LM",
#     init_lora_weights="gaussian",
# )

dataset = load_dataset("openai/gsm8k", "main", split = "train", cache_dir="/data/AISeed/huggingface")

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end)   == 1 else -0.5
        score += 0.5 if response.count(solution_start)  == 1 else -0.5
        score += 0.5 if response.count(solution_end)    == 1 else -0.5
        scores.append(score)
    return scores

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags = re.MULTILINE | re.DOTALL
)
match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>")

# REWARD function

def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.25
                else: score -= 1.0 # Penalize wrong answers
            except:
                score -= 0.5 # Penalize
        scores.append(score)
    return scores

def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            guess       = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
            continue
    return scores

# Train the model

import os
import mlflow
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Unsloth")

max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer


training_args = GRPOConfig(
    # use_vllm=True, # Use VLLM for training
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    logging_steps = 1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps = 50,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = "mlflow", # Can use Weights & Biases
    output_dir = "GSM8K-Qwen3-it", # Output directory for the model
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    # peft_config=peft_config,
    args = training_args,
    train_dataset = dataset,
)
trainer.train()