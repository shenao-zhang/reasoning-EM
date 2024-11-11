from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments
from trl import SFTTrainer
import os

os.environ["HF_TOKEN"] = "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"

# Define and parse arguments.
@dataclass
class ScriptArguments:
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=2)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.0)
    warmup_ratio: Optional[float] = field(default=0.1)
    model_name: Optional[str] = field(
        default="google/gemma-2-9b-it",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=5,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_seq_length: Optional[int] = field(default=2048)
    train_set_path: Optional[str] = field(
        default="YYT-t/gemma-2-9b-it-em-rational",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    output_dir: Optional[str] = field(
        default="./sft_gemma9b",
        metadata={"help": "The dir for output model"},
    )
    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    push_to_hub: Optional[bool] = field(
        default=True,
        metadata={"help": "Push to hub."},
    )
    hub_model_id: Optional[str] = field(
        default="gemma9b-sft-m",
        metadata={"help": "Hub model id"},
    )
    attn_implementation: Optional[str] = field(
        default="eager", #"flash_attention_2",
        metadata={"help": "Which attention implementation to use"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Define the trainer
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    weight_decay=script_args.weight_decay,
    warmup_ratio=script_args.warmup_ratio,
    do_eval=False,
    eval_strategy="no",
    save_strategy="epoch",
    save_steps=script_args.save_every_steps,
    overwrite_output_dir=True,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    deepspeed=script_args.deepspeed,
    remove_unused_columns=True,
    bf16=script_args.bf16,
    log_level="info",
    logging_strategy="steps",
    logging_steps=1,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    report_to='wandb'
)

model_kwargs = dict(
    attn_implementation=script_args.attn_implementation,
    torch_dtype=torch.bfloat16,
    use_cache=False if script_args.gradient_checkpointing else True
)

instruct_prompt = r"Answer the question based on the following example:"
example1 = r"""Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Answer: First find how many liters of the seawater are salt: 2 liters * 20% = 0.4 liters Then multiply that amount by 1000 ml/liter to find the number of ml of salt Jack gets: 0.4 liters * 1000 ml/liter = 400 ml."""
example2 = r"""Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name."""
few_shot_cot_prompt = instruct_prompt + '\n' + example2 + f'\nQuestion: '  #'\n' + example1
def cot_prefix(sample):
    replace_answer = sample["rational_answer"].replace("The answer is", "####")
    sample["text"] = 'Question: ' + sample["question"] + replace_answer #sample["rational_answer"]
#    sample["prompt"] = few_shot_cot_prompt + sample["question"]
#    sample["completion"] = sample["rational_answer"]
    return sample
train_dataset = load_dataset(script_args.train_set_path)["train"].shuffle(seed=42)
column_names = list(train_dataset.features)
train_dataset = train_dataset.map(cot_prefix, remove_columns=column_names, num_proc=16)

trainer = SFTTrainer(
    model=script_args.model_name,
    model_init_kwargs=model_kwargs,
    args=training_args,
    max_seq_length=script_args.max_seq_length,
    train_dataset=train_dataset,
    dataset_text_field="text",
    packing=True
)

train_result = trainer.train()

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")

if trainer.accelerator.is_main_process:
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(training_args.output_dir)