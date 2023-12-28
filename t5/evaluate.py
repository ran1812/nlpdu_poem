import torch
from model.simplet5 import SimpleT5
from transformers import T5Tokenizer, T5ForConditionalGeneration
import chinese_converter
import random

from dataloader import two_char_split

MODEL_PATH = './t5-new-poem6/simplet5-epoch-1-train-loss-3.223-val-loss-2.5918'
class PoemModel(SimpleT5):
  def __init__(self) -> None:
    super().__init__()
    self.device = torch.device("cuda")

  def load_my_model(self):
    self.tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

lst = ['qiyanjueju.txt','qiyanlvshi.txt','wuyanjueju.txt','wuyanlvshi.txt']
authors = []
for i in range(len(lst)):
    dataset_names = lst[i]
    source_data = []
    target_data = []
    with open('./full_data3/'+dataset_names,mode='r',encoding='utf-8') as f:
        data = f.readlines()
        for k in range(len(data)):
            tmp = data[k][:-1].split('\t')
            author = tmp[1]
            authors.append(author)
authors = list(set(authors))
import random
selected_authors = random.sample(authors, 100)

with open('./test_prompt/title.txt',mode='r',encoding='utf-8') as f:
    titles = f.readlines()
    titles = [title.rstrip('\n') for title in titles]

with open('./test_prompt/cangtou.txt',mode='r',encoding='utf-8') as f:
    cangtou = f.readlines()
    cangtou = [title.rstrip('\n') for title in cangtou]

author_prompt = "模仿："
title_prompt = "主题："
cangtou_prompt = "藏头："
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

def evaluate_format():
    poem_model.model = poem_model.model.to('cuda')
    acc = 0
    tot = 100
    for _ in range(100):
        i = random_integer = random.randint(0, 3)
        format_prompt = "格式：" + tmp1[i]
        in_request = format_prompt + EOS_TOKEN + title_prompt + EOS_TOKEN + author_prompt + selected_authors[_]
        out = poem_model.predict(in_request,
                            max_length=MAX_CONTENT_CHAR,
                            num_beams=2, temperature = 2)[0].replace(",", "，")
        ans = two_char_split(out,'，','。')[:-1]
        if tmp1[i] == '七言绝句':
            flag = all(len(item) == 7 for item in ans)
            if len(ans) == 4 and flag:
                acc += 1
            else:
                print(out)
        if tmp1[i] == '五言绝句':
            flag = all(len(item) == 5 for item in ans)
            if len(ans) == 4 and flag:
                acc += 1
            else:
                print(out)
        if tmp1[i] == '五言律诗':
            flag = all(len(item) == 5 for item in ans)
            if len(ans) == 8 and flag:
                acc += 1
            else:
                print(out)
        if tmp1[i] == '七言律诗':
            flag = all(len(item) == 7 for item in ans)
            if len(ans) == 8 and flag:
                acc += 1
            else:
                print(out)
    print(acc)

def evaluate_title():
    poem_model.model = poem_model.model.to('cuda')
    acc = 0
    tot = 100
    for _ in range(100):
        i = random_integer = random.randint(0, 3)
        format_prompt = "格式：" + tmp1[i]

        in_request = format_prompt + EOS_TOKEN + title_prompt + titles[_] + EOS_TOKEN + author_prompt
        print(in_request)
        out = poem_model.predict(in_request,
                            max_length=MAX_CONTENT_CHAR,
                            num_beams=2, temperature = 2)[0].replace(",", "，")
        print(out)
        for i in titles[_]:
            if i in out:
                acc += 1 
                break
    print(acc)

def evaluate_cangtou():
    poem_model.model = poem_model.model.to('cuda')
    acc = 0
    tot = 100
    for _ in range(100):
        i = random_integer = random.randint(2, 3)
        format_prompt = "格式：" + tmp1[i]
        if i == 2 or i == 3:
            bias = 100
        else:
            bias = 0

        word = cangtou[_+bias]
        new_tmp = ''
        for i in word[:-1]:
            new_tmp += i + ' '
        new_tmp += word[-1]
        in_request = format_prompt + EOS_TOKEN + title_prompt + titles[_] + author_prompt + selected_authors[_] + EOS_TOKEN + cangtou_prompt + new_tmp
        out = poem_model.predict(in_request,
                            max_length=MAX_CONTENT_CHAR,
                            num_beams=2, temperature = 2)[0].replace(",", "，")
        ans = two_char_split(out,'，','。')[:-1]
        for i in range(len(ans)):
            if ans[i][0] != word[i]:
                tot -= 1
                print(in_request)
                print(out)
                break
    print(tot)

evaluate_cangtou()
exit()
evaluate_format()
evaluate_title()
evaluate_cangtou()
