import torch
import torch.nn.functional as F
from datasets import load_dataset, Value, DatasetDict, Features
from typing import Any, Dict, List, Tuple
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer)
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
from functools import partial
import os
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from trl import SFTTrainer

def prompt_fn(
    examples:Dict[str, List[Any]],
    tokenizer:PreTrainedTokenizer
    ) -> Dict[str, List[Any]]:

    prompts = []  
    for context, question, answer in zip(examples["context"], examples["question"], examples['answer']):
        prompt = tokenizer.apply_chat_template([
            {"role": "context", "content": context},
            {"role": "question", "content": question},
            {"role": "answer", "content": answer}],
            tokenize=False)
        prompt += tokenizer.eos_token
        prompts.append(prompt)

    return {
        "prompt": prompts}
    
if __name__ == '__main__':
    # dataset
    dataset = load_dataset('json', data_files='../../GPT_data_list.json')
    train_testvalid = dataset['train'].train_test_split(test_size=0.2)
    dataset = DatasetDict({
        'train': train_testvalid['train'],
        'valid': train_testvalid['test']
    })
    tokenizer = AutoTokenizer.from_pretrained("vaiv/GeM2-Llamion-14B-Chat")

    # prompt
    dataset = dataset.map(
        function=partial(prompt_fn, tokenizer=tokenizer),
        batched=True,
        batch_size=32,
        load_from_cache_file=False,
        drop_last_batch=False,
        num_proc=os.cpu_count() // 2,
        features=Features({
            "prompt": Value("string")
            }),
        remove_columns=dataset['train'].features.keys(),
        desc="Prompting")
    
    # dataset = dataset.filter(lambda x : len(x['prompt']) < 2000)
    
    print(dataset)
    print(dataset['train'][0].keys())
    print(dataset['train'][0]['prompt'])

    model = AutoModelForCausalLM.from_pretrained("vaiv/GeM2-Llamion-14B-Chat", device_map="auto")
    # model_ref = AutoModelForCausalLM.from_pretrained("vaiv/GeM2-Llamion-14B-Chat", device_map="auto")
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none',
        task_type='CAUSAL_LM',
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # Set up the training arguments
    trainingArgs = TrainingArguments(
        output_dir = 'SFT_result2',
        # num_train_epochs=3,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 8,
        optim = "adamw_hf",
        save_steps = 320,
        eval_steps = 320,
        logging_steps = 4,
        max_grad_norm = 0.3,  # for gradient clipping
        max_steps = 90000,  # epoch? or step? -> num_train_epochs...
        # warmup_steps=800,
        eval_strategy="steps", # epoch? or steps?
        save_strategy='steps',
        learning_rate=1e-4,
        dataloader_num_workers=16,
        # lr_scheduler_type="cosine",
        # warmup_ratio = 0.1,
    #     remove_unused_columns=False
    )

    def format_fc(prompts):
        output = []
        for prompt in prompts['prompt']:
            output.append(prompt)
        return output

    trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
    formatting_func=format_fc,
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=trainingArgs
    )
    trainer.train()