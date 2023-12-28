import torch
from model.simplet5 import SimpleT5
from transformers import T5Tokenizer, T5ForConditionalGeneration
import chinese_converter
import random

MODEL_PATH = './final_model'
class PoemModel(SimpleT5):
  def __init__(self) -> None:
    super().__init__()
    self.device = torch.device("cuda")

  def load_my_model(self):
    self.tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)


author_prompt = "模仿："
title_prompt = "主题："
cangtou_prompt = "藏头：欢 迎 来 到 北 京"
format_prompt = "格式："
EOS_TOKEN = '</s>'

poem_model = PoemModel()
poem_model.load_my_model()
poem_model.model = poem_model.model.to('cuda')


MAX_AUTHOR_CHAR = 4
MAX_TITLE_CHAR = 12
MIN_CONTENT_CHAR = 10
MAX_CONTENT_CHAR = 64

tmp = ['七言绝句','七言律诗','五言绝句','五言律诗']
tmp1 = ['七言绝句','五言绝句','七言律诗','五言律诗']
lst = ['qiyanjueju.txt','qiyanlvshi.txt','wuyanjueju.txt','wuyanlvshi.txt']

def poem(title_str, opt_author=None, model=poem_model,
         is_input_traditional_chinese=False,
         num_beams=2):
    i = random_integer = random.randint(2, 3)
    format_prompt = "格式：" + tmp1[i]
    model.model = model.model.to('cuda')
    if opt_author:
        in_request = format_prompt + EOS_TOKEN + title_prompt + title_str[:MAX_TITLE_CHAR] + EOS_TOKEN + author_prompt + opt_author[:MAX_AUTHOR_CHAR] + EOS_TOKEN + cangtou_prompt
    else:
        in_request =  format_prompt + EOS_TOKEN + title_prompt + title_str[:MAX_TITLE_CHAR] + EOS_TOKEN + author_prompt + EOS_TOKEN + cangtou_prompt
    if is_input_traditional_chinese:
        in_request = chinese_converter.to_simplified(in_request)
    out = model.predict(in_request,
                        max_length=MAX_CONTENT_CHAR,
                        num_beams=num_beams, temperature = 2)[0].replace(",", "，")
    if is_input_traditional_chinese:
        out = chinese_converter.to_traditional(out)
        print(f"標題： {in_request.replace('</s>', ' ')}\n詩歌： {out}")
    else:
        print(f"prompt: {in_request.replace('</s>', ' ')}\npoem： {out}")

#['','秋思', "百花", '佳人有约']
for title in ['', "皇宫", '咏史']:
    # Empty author means general style
    for author in ['', "王维", "高适"]:
        poem(title, author)
    print()
