import torch
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import DatasetDict, Dataset, ClassLabel
import json
import numpy as np
import datasets 
import random

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

def get_poems(is_test=True, verbose=True):
    df_list = []
    for dynasty in POEM_CONTENT:
        size = 3 if is_test else POEM_CONTENT[dynasty]['total']
        pbar = tqdm(total=size, desc="Dynasty " + dynasty)
        for i in range(size):
            url = POEM_CONTENT[dynasty]['pattern'].format(i * 1000)
            if verbose:
                print(f"download {url} now")
            df_list.append(pd.read_json(url))
            pbar.update(1)
    return pd.concat(df_list)

def trim_author_fn(row):
    return row.s_author[:MAX_AUTHOR_CHAR]

def trim_title_fn(row):
    trimed_title = row.s_title[:MAX_TITLE_CHAR].replace(" ", "").replace("(", "").replace(")", "")
    return trimed_title

def trim_content_fn(row):
    trimed_content = row.s_content[:MAX_CONTENT_CHAR]
    # # End with a period to avoid partial ending to confuse model
    # last_period = trimed_content.rfind("。")
    # return trimed_content[:last_period+1]
    return trimed_content

def get_dataset_from_json(dataset_name):
    POEM_CONTENT = {
        'tang': {
            'total': 58,
            'pattern': "./raw_data/poet.tang.{0}.json"
        },
    }

    IS_TEST_FLOW = False

    df = get_poems(is_test=IS_TEST_FLOW, verbose=False)
    df['concat_paragraphs'] = [''.join(map(str, l)) for l in df['paragraphs']]
    df = df[['author', 'title', 'concat_paragraphs']]

    def convert_schinese(tchinese):
        return chinese_converter.to_simplified(tchinese)

    df['s_content'] = df.apply(lambda row: convert_schinese(''.join(row.concat_paragraphs)), axis=1)
    df['s_title'] = df.apply(lambda row: convert_schinese(''.join(row.title)), axis=1)
    df['s_author'] = df.apply(lambda row: convert_schinese(''.join(row.author)), axis=1)
    my_df = df
    print("my_df size", len(my_df))

    MAX_AUTHOR_CHAR = 4
    MAX_TITLE_CHAR = 12
    MIN_CONTENT_CHAR = 10
    MAX_CONTENT_CHAR = 64
    # Trim the size, a soft copy to avoid the view/copy conflict warning
    my_df['s_author_trim'] = my_df.copy().apply(trim_author_fn, axis=1)
    my_df['s_title_trim'] = my_df.copy().apply(trim_title_fn, axis=1)
    my_df['s_content_trim'] = my_df.copy().apply(trim_content_fn, axis=1)

    # Title cannot be empty
    empty_title_mask = (my_df['s_title_trim'].str.len() == 0)
    too_short_cotent_mask = (my_df['s_content_trim'].str.len() <= MIN_CONTENT_CHAR)
    invalid_mask = (('无正文' == my_df['s_content_trim']) | ('无正文' == my_df['s_author_trim']))
    too_short_mask =  empty_title_mask | too_short_cotent_mask | invalid_mask
    # filtered_my_df = my_df.loc[too_short_mask]
    # filtered_my_df

    qualitied_df = my_df.loc[~too_short_mask][[
    's_author_trim', 's_title_trim', 's_content_trim']]

    a = qualitied_df.sample(3)

    AUTHOR_PROMPT = "模仿："
    TITLE_PROMPT = "作诗："
    EOS_TOKEN = ''
    def build_dataset_df(df, include_author=True):
        dfc = df.copy()
        if include_author:
            dfc['source_text'] = TITLE_PROMPT + df['s_title_trim'] + EOS_TOKEN + AUTHOR_PROMPT + df['s_author_trim']
        else:
            dfc['source_text'] = TITLE_PROMPT + df['s_title_trim']
        dfc['target_text'] = df['s_content_trim']
        dfc = dfc[['source_text', 'target_text']]
        return dfc

    df_author_title_content = build_dataset_df(qualitied_df, True)
    df_title_content = build_dataset_df(qualitied_df, False)

    merged_df = pd.concat([df_author_title_content, df_title_content])
    return merged_df

author_prompt = "模仿："
title_prompt = "主题："
cangtou_prompt = "藏头："
format_prompt = "格式："

def get_dataset_for_full(dataset_name,sep_token = '</s>'):
    dataset = DatasetDict()
    d_train = {'source_text': [],'target_text':[]}
    d_valid = {'source_text': [],'target_text':[]}
    d_test = {'source_text': [],'target_text':[]}
    lst = ['qiyanjueju.txt','qiyanlvshi.txt','wuyanjueju.txt','wuyanlvshi.txt']

    for i in range(len(lst)):
        dataset_names = lst[i]
        source_data = []
        target_data = []
        with open('./full_data3/'+dataset_names,mode='r',encoding='utf-8') as f:
            data = f.readlines()
            for k in range(len(data)):
                tmp = data[k][:-1].split('\t')
                author = author_prompt + tmp[1]
                title = title_prompt + tmp[2]

                new_tmp = ''
                for i in tmp[3][:-1]:
                    new_tmp += i + ' '
                new_tmp += tmp[3][-1]
                cangtou = cangtou_prompt + new_tmp
                formats = format_prompt + tmp[4]

                source_data.append(formats + sep_token + title+ sep_token + author + sep_token + cangtou)
                target_data.append(tmp[0])
                source_data.append(formats + sep_token + title+ sep_token + author)
                target_data.append(tmp[0])
                source_data.append(formats + sep_token + title)
                target_data.append(tmp[0])
                source_data.append(formats)
                target_data.append(tmp[0])
                source_data.append(cangtou)
                target_data.append(tmp[0])
        zipped_lists = list(zip(source_data, target_data))
        random.shuffle(zipped_lists)
        source_data, target_data = zip(*zipped_lists)
        print(source_data[0])
        print(target_data[0])
        d_train['source_text'] += source_data[:int(0.95*len(source_data))]
        d_test['source_text'] += source_data[int(0.95*len(source_data)):]
        d_train['target_text'] += target_data[:int(0.95*len(target_data))]
        d_test['target_text'] += target_data[int(0.95*len(target_data)):]

    dataset['train'] = Dataset.from_dict(d_train)
    dataset['validation'] = Dataset.from_dict(d_valid)
    dataset['test'] = Dataset.from_dict(d_test)
    return dataset

get_dataset_for_full(1)
