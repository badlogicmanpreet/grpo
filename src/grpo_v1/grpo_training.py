import random
import copy
import re
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import argparse
from tqdm import tqdm

# Modified by Manpreet Singh

# Constants
INSTRUCTION_TEMPLATE = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Utility functions
def initialize_random_seeds(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def construct_message_prompt(messages):
    """Build a prompt from a list of messages."""
    return "\n".join([m["content"].strip() for m in messages])

def parse_model_answer(text):
    """Extract the answer from the model's output."""
    parts = text.split("<answer>")
    if len(parts) < 2:
        return None
    last_part = parts[-1]

    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer

def parse_dataset_answer(text):
    """Extract the answer from a dataset example."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def find_trailing_number(text):
    """Extract the last number from a string."""
    text = text.replace('$', '').replace('%', '')
    pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def find_isolated_number(text):
    """Extract a single number from a string."""
    if text is None:
        return None
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None

# Data preparation functions
def format_training_data(split="train"):
    """Prepare dataset for training or evaluation."""
    data = load_dataset('openai/gsm8k', 'main')[split]
    formatted_data = []
    for example in data:
        prompt_str = construct_message_prompt([
            {"role": "system", "content": INSTRUCTION_TEMPLATE},
            {"role": "user", "content": example["question"]},
        ])
        formatted_example = {
            "prompt": prompt_str,
            "answer": parse_dataset_answer(example["answer"]),
        }
        formatted_data.append(formatted_example)
    return formatted_data

# Reward functions
def calculate_accuracy_reward(prompts, completions, answer, **kwargs):
    """Calculate a reward based on the correctness of the completion."""
    responses = [completion[0]['content'] for completion in completions]
    extracted = [parse_model_answer(response) for response in responses]
    rewards = []

    for r, a in zip(extracted, answer):
        if r == a:
            rewards.append(2.0)
        else:
            r_num = find_isolated_number(str(r))
            a_num = find_isolated_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
    return rewards

def calculate_structure_reward(completions, **kwargs):
    """Calculate a reward based on the format of the completion."""
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response:
            score += 0.2
        if "</reasoning>" in response:
            score += 0.2
        if "<answer>" in response:
            score += 0.2
        if "</answer>" in response:
            score += 0.2
        rewards.append(score)
    return rewards
    
def calculate_total_reward(prompts, completions, answer):
    """Combine correctness and format rewards."""
    accuracy_rewards = calculate_accuracy_reward(prompts, completions, answer)
    structure_rewards = calculate_structure_reward(completions)
    
    combined_rewards = []
    for acc_reward, struct_reward in zip(accuracy_rewards, structure_rewards):
        combined_rewards.append(acc_reward + struct_reward)
    
    return combined_rewards

# Model evaluation functions
def test_model_performance(model, tokenizer, eval_examples, device):
    """Evaluate the model on a set of examples."""
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print(f"Evaluation on {total} examples")
    print("="*50)

    for example in eval_examples:
        full_prompt = example["prompt"]
        expected = example["answer"]

        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            predicted = parse_model_answer(response)

            if predicted == expected:
                is_correct = True
            else:
                pred_num = find_isolated_number(str(predicted))
                exp_num = find_isolated_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    pred_num = find_trailing_number(str(predicted))
                    exp_num = find_trailing_number(str(expected))
                    is_correct = (pred_num is not None and exp_num is not None and pred_num == exp_num)

            if is_correct:
                correct += 1

            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nPredicted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-"*50)

        except Exception as e:
            print("\nFailed to parse model output for example:")
            print(full_prompt)
            print("Error:", e)
            print("-"*50)

    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.2f}%)")
    print("="*50)

    model.train()
    return accuracy

# GRPO functions
def compute_token_logprobs(logits, input_ids):
    """Compute log softmax for the given logits and select the probabilities of the input_ids."""
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def extract_sequence_logprobs(model, input_ids, attention_mask, logits_to_keep):
    """Compute log probabilities for the given input_ids."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return compute_token_logprobs(logits, input_ids)

def build_completion_mask(completion_ids, eos_token_id):
    """Create a mask for the completion ids."""
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()

def produce_model_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """Generate completions for the given prompts."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False
    )
    completion_ids = outputs[:, prompt_length:]
    completion_mask = build_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def prepare_training_rollouts(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """Generate rollout data for the GRPO algorithm."""
    device = next(model.parameters()).device
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = produce_model_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        old_log_probs = extract_sequence_logprobs(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = extract_sequence_logprobs(ref_model, input_ids, attention_mask, logits_to_keep)
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def compute_grpo_objective(model, ref_model, rollout_data, tokenizer, reward_function, beta=0.01, epsilon=0.2):
    """Compute the GRPO loss."""
    device = next(model.parameters()).device
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = extract_sequence_logprobs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=device
    )
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss, avg_reward

def setup_model_for_training(model):
    """Optimize model memory usage for training."""
    model.train()
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_input_requires_grad(module, input, output):
            output.requires_grad = True
        model.get_input_embeddings().register_forward_hook(make_input_requires_grad)

    model.gradient_checkpointing_enable()

    return model

def execute_grpo_training(model, tokenizer, train_data, num_iterations=1, num_steps=500, batch_size=4,
                    num_generations=4, max_completion_length=128, beta=0.1,
                    learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None, device_ids=None):
    """Train the model with the GRPO algorithm."""
    assert device_ids is not None and len(device_ids) > 1, "This code needs at least 2 GPU cores to run!"

    model = nn.DataParallel(model, device_ids=device_ids)
    print(f"Model wrapped with DataParallel across GPUs: {device_ids}")

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        ref_model = copy.deepcopy(model.module)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model created.")

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        for step in tqdm(range(num_steps), desc="Training"):
            batch_samples = random.sample(train_data, batch_size)
            with torch.no_grad():
                rollout_data = prepare_training_rollouts(
                    model.module,
                    ref_model,
                    tokenizer,
                    batch_samples,
                    num_generations,
                    max_completion_length
                )
            for grpo_iter in range(mu):
                loss, avg_reward = compute_grpo_objective(
                    model.module,
                    ref_model,
                    rollout_data,
                    tokenizer,
                    reward_function,
                    beta=beta,
                    epsilon=epsilon
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                wandb.log({
                    "loss": loss.item(),
                    "average_reward": avg_reward,
                    "iteration": iteration + 1,
                    "step": step + 1,
                    "grpo_iter": grpo_iter + 1
                })
                if (step + 1) % 50 == 0:
                    print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                          f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}, reward: {avg_reward:.4f}")

    return model.module

def main():
    """Main function to run the training and evaluation."""
    parser = argparse.ArgumentParser(description="Train a model with GRPO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                      help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="math_solver_model", 
                      help="Directory to save the model")
    parser.add_argument("--num_iterations", type=int, default=1, 
                      help="Number of GRPO iterations")
    parser.add_argument("--num_steps", type=int, default=500, 
                      help="Number of steps per iteration")
    parser.add_argument("--batch_size", type=int, default=7, 
                      help="Batch size for training")
    parser.add_argument("--num_generations", type=int, default=12, 
                      help="Number of generations per sample")
    parser.add_argument("--max_completion_length", type=int, default=400, 
                      help="Maximum completion length")
    parser.add_argument("--beta", type=float, default=0.04, 
                      help="Beta parameter for KL divergence")
    parser.add_argument("--learning_rate", type=float, default=5e-6, 
                      help="Learning rate")
    parser.add_argument("--mu", type=int, default=1, 
                      help="Number of GRPO iterations per step")
    parser.add_argument("--epsilon", type=float, default=0.1, 
                      help="Epsilon parameter for clipping")
    parser.add_argument("--eval_size", type=int, default=30, 
                      help="Number of examples for evaluation")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed")
    args = parser.parse_args()

    # Set random seed
    initialize_random_seeds(args.seed)

    # Initialize wandb
    os.environ["WANDB_PROJECT"] = "GRPO-Math-Solver"
    wandb.init(project=os.environ["WANDB_PROJECT"], config=vars(args))

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    device_ids = list(range(num_gpus)) if num_gpus > 1 else None

    # Prepare dataset
    print("Preparing dataset...")
    all_data = format_training_data("train")
    random.shuffle(all_data)
    size_of_eval_set = args.eval_size
    eval_data = all_data[:size_of_eval_set]
    train_data = all_data[size_of_eval_set:]

    # Initial evaluation
    print("Initial evaluation...")
    pre_grpo_accuracy = test_model_performance(model, tokenizer, eval_data, device)
    print(f"Initial accuracy: {pre_grpo_accuracy:.2f}%")

    # Optimize model memory
    model = setup_model_for_training(model)

    # Train with GRPO
    print("Training with GRPO...")
    training_config = {
        'num_iterations': args.num_iterations,
        'num_steps': args.num_steps,
        'batch_size': args.batch_size,
        'num_generations': args.num_generations,
        'max_completion_length': args.max_completion_length,
        'beta': args.beta,
        'learning_rate': args.learning_rate, 
        'mu': args.mu,
        'epsilon': args.epsilon
    }

    model = execute_grpo_training(
        model=model, 
        tokenizer=tokenizer,
        train_data=train_data,
        reward_function=calculate_total_reward,
        device_ids=device_ids,
        **training_config 
    )

    # Final evaluation
    print("Final evaluation...")
    post_grpo_accuracy = test_model_performance(model, tokenizer, eval_data, device)
    print(f"Final accuracy: {post_grpo_accuracy:.2f}%")

    # Save the model
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Finish wandb
    wandb.finish()

if __name__ == "__main__":
    main()