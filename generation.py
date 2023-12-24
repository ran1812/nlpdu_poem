import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('./ckpts_new')
model = GPT2LMHeadModel.from_pretrained('./ckpts_new')

from transformers import TextGenerationPipeline
text_generator = TextGenerationPipeline(model, tokenizer)   

while(1):
    prompt = input('begin:')
    ans = text_generator(prompt, max_length=200, do_sample=True)
    print(ans)