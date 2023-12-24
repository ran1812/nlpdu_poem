import torch
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import DatasetDict, Dataset, ClassLabel
import json
import numpy as np
import datasets 

def two_char_split(string, delimiter1, delimiter2):
    first_split = string.split(delimiter1)
    result = [substr.split(delimiter2) for substr in first_split]
    flattened_result = [item for sublist in result for item in sublist]
    return flattened_result

def get_dataset(dataset_name, sep_token = '<S>', end_token = '<T>'):
    '''
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    dataset = DatasetDict()
    d_train = {'text': []}
    d_valid = {'text': []}
    d_test = {'text': []}
    lst = ['qiyanjueju.txt','qiyanlvshi.txt','wuyanjueju.txt','wuyanlvshi.txt']

    if isinstance(dataset_name,list):
        train_text = []
        validation_text = []
        test_text = []
        start_label = 0
        for name in dataset_name:			
            data = get_dataset(name)
            train_text += data['train'][:]['text']
            validation_text += data['validation'][:]['text']
            test_text += data['test'][:]['text']
        dataset['train'] = Dataset.from_dict({'text':train_text})
        dataset['validation'] = Dataset.from_dict({'text':validation_text})
        dataset['test'] = Dataset.from_dict({'text':test_text})
        return dataset

    name = ['七言绝句','七言律诗','五言绝句','五言律诗']
    if dataset_name in ['poem']:
        for i in range(len(lst)):
            dataset_names = lst[i]
            with open('./data/'+dataset_names,mode='r',encoding='utf-8') as f:
                data = f.readlines()
                for k in range(len(data)):
                    if k == 0:
                        data[k] = data[k][1:]
                    data[k] = name[i] + sep_token + data[k] + end_token
            d_train['text'] += data[:int(0.98*len(data))]
            d_valid['text'] += data[int(0.98*len(data)):int(0.99*(len(data)))]
            d_test['text'] += data[int(0.99*len(data)):]

        dataset['train'] = Dataset.from_dict(d_train)
        dataset['validation'] = Dataset.from_dict(d_valid)
        dataset['test'] = Dataset.from_dict(d_test)
    elif dataset_name in ['cangtou']:
        for i in range(len(lst)):
            dataset_names = lst[i]
            with open('./data/'+dataset_names,mode='r',encoding='utf-8') as f:
                data = f.readlines()
                for k in range(len(data)):
                    if k == 0:
                        data[k] = data[k][1:]
                    a = two_char_split(data[k],'，','。')
                    head = ''
                    for i in a[:-1]:
                        head += i[0]

                    data[k] = '藏头诗：' + head + sep_token + data[k] + end_token
            d_train['text'] += data[:int(0.98*len(data))]
            d_valid['text'] += data[int(0.98*len(data)):int(0.99*(len(data)))]
            d_test['text'] += data[int(0.99*len(data)):]

        dataset['train'] = Dataset.from_dict(d_train)
        dataset['validation'] = Dataset.from_dict(d_valid)
        dataset['test'] = Dataset.from_dict(d_test)
    else:
        raise ValueError("Need correct dataset name")
    print(dataset['train']['text'][0])
    print(dataset['train']['text'][-1])
    return dataset

