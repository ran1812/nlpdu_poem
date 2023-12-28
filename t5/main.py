from datasets import load_dataset
import torch
import transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,Trainer,default_data_collator
from transformers.testing_utils import CaptureLogger
from itertools import chain
import evaluate

dataset = load_dataset("wikitext",name='wikitext-2-v1')
config = AutoConfig.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

column_names = list(dataset["train"].features)
print(column_names)

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples['text'])
        return output

tokenized_datasets = dataset.map(tokenize_function,batched=True,remove_columns=column_names)
block_size = tokenizer.model_max_length
if block_size > 1024:
    block_size = 1024

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

metric = evaluate.load("accuracy")
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]
trainer = Trainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer,
        data_collator=default_data_collator, compute_metrics=compute_metrics, 
        preprocess_logits_for_metrics=preprocess_logits_for_metrics)

train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

