Write how GRPO works

Use the following setup with vscode/lambdalabs
8*A100 GPUs
install notebook plugin
pip install uv
uv venv rl --python 3.11 && source rl/bin/activate && uv pip install --upgrade pip
cmd-shift-P, reload window

For me the goal is to take a Qwen/Qwen2.5-7B-Instruct model, which is highly capable model for coding and mathematics with No. of parameters - 7.61B and has 28 layers. Instruct model is ideal here since the model is capable of handling intructions (Q/A type). I use the huggingface library to get the pretrained model and use torch.bfloat16 for memory efficiency. Let me explain why bfloat16 and not 32,

Explain
float32: 1 sign bit + 8 exponent bits + 23 mantissa bits = 32 bits
bfloat16: 1 sign bit + 8 exponent bits + 7 mantissa bits = 16 bits
bfloat is known as brain float, while representing a value in bfloat16, the exponent remains the same, which means it can represent very large and small values in similar way to float16 without underflowing or overflowing which is a problem in float16 (has 5 exponent bits). And finally by requiring only 16 bits per element or value, it leads to reduced VRAM usage and is relatively now having faster memory transfers. Precision drop is within a very tolerable level for a model.

Once the model is downloaded using from_pretrained, i get the tokenizer using huggingface AutoTokenizer by passing the model and setting the padding side to left, padding_side="left" here means that when the tokenizer pads sequences to a certain length, it inserts the padding tokens at the beginning (left side) of the sequence rather than at the end (right side). Just to note, Many decoder-based models often require or work best with left padding.

Since we run on 8*A100's or 8*H100s, we get the number of GPUs and initialize the device id's. Following which we prepare the training dataset, shuffle the dataset for randomness, declare the size of our evaluation set to 30 examples and finally create the evaluation and training datasets.

It is extremely important to evaluate the eval set with current model before we finetune using GRPO, this will give us an accuracy index of how well the base model was able to handle these math problems before it was tuned. In our case we get around 63% accuracy, which is not very bad.

Before we proceed with model training, we optimize our model memory, lets talk about it a little. First thing we do in the optimize_model_memory is to set the model to training mode, we then disable the KV caching, the KV caching is explained here /Users/manpreet.singh/git/open-r1/training/grpo_v2/kv-cache-description.md.
We then ensure that the input requires gradients, basically we are enabling require_grad=True on the input embedding layer. This ensures that when backpropagation occurs, the gradient computation doesn't stop at the first layer but properly flows through the input embeddings. Finally we enable gradient checkpointing, Gradient checkpointing reduces memory usage during neural network training by storing only select activation checkpoints instead of all intermediate values. When gradients are needed during backpropagation, activations between checkpoints are recomputed on-the-fly rather than stored. This trades additional computation time for significantly reduced memory requirements, enabling training of larger models on limited hardware.

Lets now move to the core of finetuning, first thing we do here is setup the training config, 
- num_iterations: This parameter determines how many complete cycles of the outer loop in the GRPO algorithm will be performed. Each iteration involves creating a new reference model (a deep copy of the current policy model) and then running multiple training steps.
- num_steps: For each step, the model samples a batch of data, generate completetions, applies GRPO updates. With 500 steps, the model is updated 500 times during the training process (subject to mu).
- batch_size: This determines how many prompts are sampled and processed together in a single training step. With a batch size of 7, each training step will involve generating completions for 7 different prompts. Larger batch sizes typically provide more stable training but require more memory.
   Note: The total number of completions generated during training would be:
         7 (batch size) × 12 (completions per prompt) × 500 (steps) = 42,000 completions
         These 42,000 completions are what the model actually learns from, as each completion receives a reward score that guides the optimization process. Also remember that at each step it randomly samples 7 prompts from the training data.
- num_generations: generate 12 completetions
- max_completion_length: 400, this is the max length of the generated completion.
- beta: is the KL penalty coefficient, this parameter controls how much the model is penalized for diverging from the original model. Lower value like .04 means it allows model to deviate more.
- learning rate: determines how large is the update to the model paramaeters, smaller value means smaller updates which is usually good for better stability of the model.
- mu: specifies the number of times the model is updated for each batch of data, with mu set to 1, the model is updated only once per batch, if mu was higher than the model will be updated that number of times before moving to the next batch.
- epsilon: this parameter limits how much the policy can change in single update by clipping the probabilites between new and old policies. 0.1 means clipped to the range [0.9, 1.1].

Affair of 'beta'(Kullback-Leibler (KL)) and 'epsilon'
    Both parameters serve as constraints that help stabilize training,
        - β (KL penalty coefficient): Controls how much the policy is allowed to diverge from the reference policy by adding a penalty based on the KL divergence between them. It's a "soft constraint" that discourages large changes.
        - ε (PPO clipping parameter): Directly limits how much the probability ratio between new and old policies can change by clipping the ratio to a range of [1-ε, 1+ε]. It's a "hard constraint" that prevents extreme updates.

    How they work together
    These parameters provide complementary guardrails:

    Different mechanisms: While ε provides a hard clipping that absolutely prevents updates outside a certain range, β provides a softer penalty that increases as the policy diverges more.
    
    Balance of exploration vs stability:
    - A lower β allows more exploration but requires a tighter ε to prevent instability
    - A higher β restricts exploration more, which might allow for a larger ε

    Optimization perspective: Both parameters affect the "trust region" - how far the algorithm can move from the current policy in a single update. In practice, these parameters are often tuned together.

    Example: If the original model predicted the token 'manpreet' with a probability of x (let's say 0.3), and after an update the new model predicts it with probability x+0.1 (0.4), this represents a divergence in the policy.

wandb is used for monitoring the training runs, it is an excellent source for viewing the progress. I have added relevant graphs during the training process.

finally, we now hit the train_with_grpo, lets dive into the details.

GRPO Details (train_with_grpo)
==============================

Initialization Phase
--------------------
1. We are passing the model, tokenizer, training data, reward function (combined in this case), device ids and training configs.
2. Since we are using 8*GPUs, we will wrap the model with DataParallel
    - dataparallel enables parallel processing, when we send the batch of data to the wrapped model, it automatically splits the batch into chunks and sends each chunk to different GPU.
    - during forward pass, the model is replicated onto each GPU deivice and each GPU processes its chunk independently
    - after the forward pass is done, the output is gathered back to the main GPU and combined
    - during backward pass the gradients are computed one each GPU and then averaged across all GPUs.
3. Main outer loop starts with num of iterations, 1 in our case
4. Create a deep reference copy of the original model, we use it later with Beta and Epsilon for controling the divergence for new policy (model).
    - set this reference model to evaluation mode
    - also set the requires_grad to False for each parameter of this reference model
    - explicit setting for each parameter here is more efficient
5. Now initialize the optimizer
6. set the model to training mode

Batch Processing Loop (num_steps)
---------------------
1. sample random batch from the training data
2. generate rollout data for batch
    - using the the context manager we temporarily disable the gradient computation
    - generate_rollout_data
        - from the batch we prepare the list of prompts, answers
        - again using the the context manager we temporarily disable the gradient computation
            - generate_completions
                - for this method just send the model, tokenizer, prompts, num of generations, max complete length
                - since we are generating completetions for each prompt, we dont need to send answers
                - first thing we do is encode the prompts with tokenizer, we use left padding, it is bit more efficient
                - input ids are the numerical token identifiers that represent our text (prompts in this case)
                - attention mask is a binary tensor, telling the model which token to pay attention to and which to ignore, 1 and 0 respectively, where 0 is usually for padded tokens
                - prompt length is the length(num tokens) of each prompt, 1 refers to second dimension i.e. sequence length, this may be used later to seperate prompt from generated completion
                - repeat interleave is helping to duplicate the prompts (both ids and attention masks) i.e. batch_size * number_of_generation times
                - use the model to generate the completions
                    - do_sample is True, meaning the model generates more diverse and creative outputs, if it was False the model will be greedy to select the next best token
                    - temperature controls the randomness in the generation, 1.0 means we use the original probability distribution generated by model, >1.0 means increase diversity whereas <1.0 means focused and determinitic
                    - early stopping of False means generation continuous until all sequences reach the maximum length, True would have stopped it early at EOS token
                - completetion ids are the numerical token identifiers of the generated completions, taken from [:,prompt_length:]
                - create completion mask, identifies EOS token position in the completions and create mask for valid tokens
            - we get back prompt_ids, prompt_mask, completion_ids, completion_mask
            - cat will combine the [prompt + completion] leading to fresh input_ids and attention_mask
            - logits to keep are the completion ids, these are used below
            - time to compute log probabilites using the new (policy) and reference model
                - we have passed model, input ids, attention mask, logits to keep
                - we're running the current policy model on the full input (prompt + completion) to get fresh logits for the completions that were previously generated. This might seem redundant at first, but it's actually necessary for the policy gradient update. Here's why:
                    1. Initial Generation: When we first called generate_completions(), we're using the model to sample completions. During this phase, the model is in inference mode, and we're not calculating or storing the full probability distribution over the vocabulary for each token.
                    2. Probability Estimation: For the GRPO update, you need the exact probabilities the current policy assigns to the tokens that were generated. This requires another forward pass through the model, but this time we're explicitly extracting the logits for every token.
                - finally use the selective log softmax on the logits and input_ids, in this case for completion part only, done for both current and reference policies.
                    1. Applies log softmax to convert logits to log probabilities over the vocabulary.
                    2. Uses gather to extract only the log probabilities corresponding to the input_ids.
                    3. Removes the extra dimension to match the original shape of input_ids.
            - finally we have old and reference log probabilities
            - create the formatted completetions
                    1. This converts each completion from token IDs back to human-readable text
                    2. tokenizer.decode(ids, skip_special_tokens=True) transforms the numerical token IDs back into text, removing special tokens like [PAD], [EOS], etc.
                    3. The structure [{'content': text}] creates a specific format that the reward function expects
3. Back to the inner loop

GRPO Loss Loop (num_steps)
--------------
1. mu defines the number of times we do backpropogation per batch
2. grpo_loss
    - we pass the model, reference model, rollout data, tokenizer, reward function, beta and epsilon
    - pulls all the variables from rollout data
    - computes token's log probabilities using the current model
        1. This calculates the log probabilities of the same completion tokens under the current state of the policy model. This is critical because during training, the model parameters are changing, so the probabilities assigned to the same tokens will differ from when they were originally generated.
    - compute the probability ratio between the current policy and the old policy:
        1. token_log_probs - old_log_probs: Subtracting log probabilities is equivalent to dividing probabilities
        2. torch.exp(...): Converting back from log space to probability space
        3. This ratio tells you how much more (or less) likely the model is to generate the same completions now compared to when they were originally sampled. A ratio greater than 1 means the model is now more likely to generate those tokens, less than 1 means it's less likely.
        4. This ratio is fundamental to the PPO algorithm (which GRPO is based on), as it measures how much the policy has changed and allows for controlled updates through clipping.
    - calculate the rewards, returns a rewards tensor
        1. prompts: The original prompts, repeated to match the number of completions
        2. completions: The generated text completions in a formatted structure
        3. answer: The reference/expected answers, also repeated to match completions
    - loss calculation
        1. first, the code reshapes the rewards tensor to separate rewards by prompt. Basically we get a matrix of size 7*12, 7 is the batch size and 12 is the completion length
        2. calculates the mean reward across its completions
        3. calculates the standard deviation of rewards
        4. Computes the advantages by standardizing the rewards:
            - Subtracts the mean reward for each prompt from its completion rewards
            - Divides by the standard deviation (with a small epsilon to prevent division by zero)
            - This normalization makes the algorithm more stable by comparing each completion against others from the same prompt
        5. now calculate the surr1 and surr2, this is ratio * advantage
            - in a way we are multiplying (the difference in completion probabilities at different model times for same prompt * normalized reward for same prompt), not intiutive, but pretty cool
        6. surrogate loss in the min of surr1, surr2
        7. Now, calculate the KL divergence, which is how much are we deviating from the reference model
        8. we then calculate the per token loss, which is surrogate loss - beta * kl
        9. finally, we put together the loss in scalar value
            - Applies the completion mask to only consider valid tokens (before EOS)
            - Takes the average loss per valid token for each sequence
            - Takes the mean across the batch
            - Negates the result because we want to maximize reward (but optimizers minimize)

Finally,
1. optimizer.zero_grad() - clears any older gradients from previous computations
2. loss.backward() - compute the gradients of the loss w.r.t to model parameters, gradients are stored in .grad attribute of the parameters.
3. clipping
    - Limits the magnitude of gradients to prevent excessively large updates
    - The max_norm=0.1 parameter sets a relatively tight constraint, which is common in fine-tuning large language models
    - Gradient clipping is especially important in reinforcement learning, where rewards can sometimes create large gradients
4. optimizer.step()
    - Actually updates the model parameters using the calculated gradients
    - The optimizer (AdamW in this case) applies the appropriate learning rate, momentum, and weight decay
    - This is what moves the model in the direction that reduces the loss function