import pandas
import numpy
import wordcloud
import webdataset
import os, sys
import pickle
import json
import matplotlib.pyplot as plt
import plotnine
from coronai.library.data.datasets.claws import preprocess_tweet
from tqdm import tqdm
import torch, numpy, torch.nn
from coronai.library.gadgets import remove_module_from_parameter_names
from transformers import BertTokenizer, BertForSequenceClassification


if __name__ == "__main__":
    repo = '/root'

    processed_data = pandas.read_csv(os.path.join(repo, 'df_with_symptoms.csv'))
    processed_data['preprocessed_tweets'] = processed_data['tweet'].copy().apply(
        lambda x: preprocess_tweet(x) if isinstance(x, str) else 'bad string')

    # processed_data = processed_data[~(processed_data.preprocessed_tweets == 'bad string')]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True, num_labels=4)
    ckpt = torch.load('..path..to../last_epoch_checkpoint.pt',
                      map_location='cpu')
    model.load_state_dict(remove_module_from_parameter_names(ckpt['model']))
    del ckpt

    softmax = torch.nn.Softmax(dim=-1)
    device = torch.device('cuda:0')
    device_list = [0, 1, 2, 3]

    model = model.to(device)
    model = torch.nn.DataParallel(model, [torch.device('cuda:%d' % i) for i in device_list])

    output_pdfs = []

    batch_size = 50
    cursor = 0
    while cursor < processed_data.shape[0]:
        print("{} / {}        ".format(cursor, processed_data.shape[0]), end='\r')
        end_index = cursor + batch_size
        if end_index > processed_data.shape[0]:
            end_index = processed_data.shape[0]
        list_of_texts = processed_data['preprocessed_tweets'].iloc[cursor:end_index].tolist()
        cursor += batch_size
        encoded_batch = tokenizer(list_of_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoded_batch['input_ids'].to(device)
        mask = encoded_batch['attention_mask'].to(device)
        outputs = softmax(model(input_ids, attention_mask=mask).logits).data.cpu().numpy().tolist()
        del input_ids
        del mask
        output_pdfs += outputs

    output_pdfs = numpy.array(output_pdfs)
    processed_data['hate_prob'] = output_pdfs[:, 0]
    processed_data['neutral_prob'] = output_pdfs[:, 1]
    processed_data['other_prob'] = output_pdfs[:, 2]
    processed_data['counterhate_prob'] = output_pdfs[:, 3]

    processed_data.to_csv(os.path.join(repo, 'df_with_symptoms_and_hate_scores.csv'))