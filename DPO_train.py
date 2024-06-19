import gc
import os
import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Value, DatasetDict, Features
from functools import partial
from utils import orpo_prompt_fn, print_trainable_parameters
import bitsandbytes as bnb

os.environ['NCCL_P2P_DISABLE']='1'
os.environ['NCCL_IB_DISABLE']='1'
os.environ['WANDB_DISABLED'] = 'true'
# current_mse = float("inf")

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config


def create_peft_config(rank, modules):
    config = LoraConfig(
        use_dora=True,
        target_modules=modules+['ffn'],
        task_type="CAUSAL_LM",
        bias='lora_only', 
        lora_dropout=0.05,
        lora_alpha=32,
        r=rank
    )
    return config


if __name__ == '__main__':
    output_dir = 'DPO_result1'
    
    # Load Dataset
    dataset = load_dataset('junghyeon0427/real_dataset')
    tokenizer = AutoTokenizer.from_pretrained("vaiv/GeM2-Llamion-14B-Chat")
    dataset = dataset.map(
            function=partial(orpo_prompt_fn, tokenizer=tokenizer),
            batched=True,
            batch_size=32,
            load_from_cache_file=False,
            drop_last_batch=False,
            num_proc=os.cpu_count() // 2,
            features=Features({
            "prompt": Value("string"),
            "chosen": Value("string"),
            "rejected": Value("string")}),
            remove_columns=['context', 'question'],
            desc="Prompting")
    train_testvalid = dataset['train'].train_test_split(test_size=0.005)
    dataset = DatasetDict({
            'train': train_testvalid['train'],
            'valid': train_testvalid['test']
    })
    dataset = dataset.filter(lambda x: len(x['prompt']+x['chosen']) < 4000 or len(x['prompt']+x['rejected']) < 4000)
    ##########
    ##########
    ##########
    ##########
    ##########
    # dataset['train'] = dataset['train'].select(range(100))  # TODO
    ##########
    ##########
    ##########
    ##########
    ##########
    print(dataset)
    for key, val in dataset['train'][0].items():
        print(key)
        print(val)
        print('===' * 10)
    
    from huggingface_hub import HfApi, HfFolder
    # Load Model
    bnb_config = create_bnb_config()
    model = AutoModelForCausalLM.from_pretrained("MinsungKKK/SFT_Chat", 
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype='auto',
                                                 quantization_config=bnb_config,
                                                 device_map="auto",)
    model.config.use_cache = False
    model_ref = AutoModelForCausalLM.from_pretrained("MinsungKKK/SFT_Chat", 
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype='auto',
                                                 load_in_4bit=True,
                                                 device_map="auto")
    
    model = prepare_model_for_kbit_training(model)
    modules = find_all_linear_names(model)
    lora_config = create_peft_config(16, modules)
    model = get_peft_model(model, lora_config)
    
    # replace_lora_weights_loftq(model)
    
    print_trainable_parameters(model)
    print(lora_config)

    # Set up the training arguments
    orpo_args = DPOConfig(
        output_dir = output_dir,
        
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        auto_find_batch_size=True,
        
        learning_rate=1e-4,  # TODO
        lr_scheduler_type="cosine",  # TODO
        max_grad_norm = 3,  # TODO
        save_steps = 200,
        eval_steps = 200,
        max_steps = 90000,

        logging_steps = 8,
        warmup_steps=100,
        bf16=True,
        max_length=4096,
        max_prompt_length=4096,
        optim="paged_adamw_8bit",
        evaluation_strategy="steps",
        save_strategy='steps',
        dataloader_num_workers=16,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        args=orpo_args,
        beta=0.1,  # TODO
        # formatting_func=
    )
    
    trainer.train()
    trainer.save_model('orpo_trained')
