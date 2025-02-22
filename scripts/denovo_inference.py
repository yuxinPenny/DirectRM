import math, os, re, sys, csv,json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torcheval import metrics
from utils.dataset import denovoDataset
from utils.model import denovoModel
from random import sample

def prepare_dataloader(split):
    data = np.load(FLAGS.feature_dir + '/' + split + '.npz')
    batch_size = cfg['batch_size']
    seq = data['seq']
    stat = data['stat']
    bse = data['bse']

    col_ids = list(range(2,8*9,8))
    stat = stat.reshape(len(stat),-1)
    for id in col_ids:
        stat[:,id] = (stat[:,id] - np.mean(stat[:,id]))/np.std(stat[:,id])
    stat = stat.reshape(len(stat),9,-1)

    dataset = denovoDataset(seq,stat,bse)
    test_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)

    return test_loader

def predict(test_loader,model):
    device = cfg['device']
    model.eval()
    preds = torch.tensor([]).to(device)
    with torch.no_grad():
        for i, (b_seq,b_stat,b_bse) in enumerate(tqdm(test_loader)):
            b_seq = b_seq.to(device)
            b_stat = b_stat.to(device)
            b_bse = b_bse.to(device)
            b_preds = model(b_seq,b_stat,b_bse)
            preds = torch.cat([preds,b_preds],dim=0)
    return preds

if __name__ == '__main__':
    paser = argparse.ArgumentParser('find modified kmers')
    paser.add_argument('--feature_dir',required=True,help='directory of features')
    paser.add_argument('--model_path',required=False,help='path to pretrained model')
    paser.add_argument('--splits',nargs='*',help='splits to process')
    paser.add_argument('--prefix',default='',help='output prefix')
    paser.add_argument('--device',default='cuda:0',help='output prefix')

    args = paser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args

    global cfg
    cfg = {'device':'cuda:0',
           'batch_size':FLAGS.device}

    model = denovoModel().to(cfg['device'])
    model.load_state_dict(torch.load(FLAGS.model_path))

    if len(FLAGS.splits) > 2:
        splits = ['split' + str(s) for s in FLAGS.splits]
    else:
        s1 = int(FLAGS.splits[0])
        s2 = int(FLAGS.splits[1]) + 1
        splits = ['split' + str(s) for s in range(s1,s2)]

    for split in splits:
        test_loader = prepare_dataloader(split)
        preds = predict(test_loader,model)
        mod_status = preds[:,1].detach().cpu().numpy()
        print(np.sum(mod_status>0.5))
        print(len(mod_status))
        np.save(FLAGS.feature_dir + '/' + split +FLAGS.prefix +'_mod',mod_status)


