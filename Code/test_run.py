import os
from os.path import exists, join
import json
import torch
from dataloader import MatchSumPipe
from metrics import MarginRankingLoss, ValidMetric, MatchRougeMetric
from utils import read_jsonl, get_data_path, get_result_path
from fastNLP.core.tester import Tester




save_path = "./Models"
models = os.listdir(save_path)
print(models)

def get_result_path(save_path, cur_model):
    result_path = join(save_path, '../result')
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, cur_model)
    if not exists(model_path):
        os.makedirs(model_path)
    dec_path = join(model_path, 'dec')
    ref_path = join(model_path, 'ref')
    os.makedirs(dec_path)
    os.makedirs(ref_path)
    return dec_path, ref_path

def get_data_path(mode, encoder):
    paths = {}
    if mode == 'train':
        paths['train'] = 'data/train_CNNDM_' + encoder + '.jsonl'
        paths['val']   = 'data/val_CNNDM_' + encoder + '.jsonl'
    else:
        paths['test']  = 'data/test_CNNDM_' + encoder + '.jsonl'
    return paths

path = get_data_path("test","bert")
print(path)
# # for name in path:
# #     assert exists(path[name])
# #     print(path[name])


datasets = MatchSumPipe(20, "bert").process_from_file(path)
print('Information of dataset is:')
print(datasets)
test_set = datasets.datasets['test']
device = int(0)
batch_size = 1

for cur_model in models:
    print('Current model is {}'.format(cur_model))

    # load model
    model = torch.load(join(save_path, cur_model))

    # configure testing
    dec_path, ref_path = get_result_path(save_path, cur_model)
    test_metric = MatchRougeMetric(data=read_jsonl(path['test']), dec_path=dec_path, 
                              ref_path=ref_path, n_total = len(test_set))
    tester = Tester(data=test_set, model=model, metrics=[test_metric], 
                     batch_size=batch_size, device=device, use_tqdm=False)
    tester.test()
