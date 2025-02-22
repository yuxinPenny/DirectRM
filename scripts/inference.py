import os, sys, json
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.dataset import pmlDataset
from utils.model import *

def prepare_dataloader(seq,stat,bse):

    col_ids = list(range(2,8*9,8))
    stat = stat.reshape(len(stat),-1)
    for id in col_ids:
        stat[:,id] = (stat[:,id] - np.mean(stat[:,id]))/np.std(stat[:,id])
    stat = stat.reshape(len(stat),9,-1)

    batch_size = 512
    dataset = pmlDataset(seq,stat,bse)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return data_loader

def predict(data_loader,model):
    device = FLAGS.device
    model.eval()
    tt_preds = torch.tensor([]).to(device)
    tt_attens = torch.tensor([]).to(device)

    with torch.no_grad():
        for i, (b_seq,b_stat,b_bse) in enumerate(data_loader):
            b_seq = b_seq.to(device)
            b_stat = b_stat.to(device)
            b_bse = b_bse.to(device)

            b_preds,b_attens = model(b_seq,b_stat,b_bse)

            tt_preds = torch.cat([tt_preds,b_preds],dim=0)
            tt_attens = torch.cat([tt_attens,b_attens])

    return tt_preds,tt_attens

def reg2base(regs):
    pos = range(regs['start']+1,regs['end']+1)
    if regs['strand'] == '-':
        pos = pos[::-1]
    return pos

def find_mod_base(seq,attens,preds,mod_type=0,ref_base=1):
    int_seq = np.argmax(seq,axis=-1)
    int_seq = torch.from_numpy(int_seq).to(FLAGS.device)
    mod_attens = attens[:,mod_type,:]
    mod_attens[int_seq!=ref_base] = 0
    tmp = torch.zeros_like(mod_attens).to(FLAGS.device)
    max_id = torch.argmax(mod_attens,axis=1)
    max_id = max_id.unsqueeze(1)
    tmp.scatter_(1,max_id,1)
    mod_attens = mod_attens * tmp
    base_results = mod_attens * torch.tile(preds[:,mod_type].reshape(-1,1),[1,9])
    base_results = base_results.reshape(-1,1)
    return base_results.detach().cpu().numpy()

def count_mod(x):
    return x.sum(skipna=True)
if __name__ == '__main__':
    paser = argparse.ArgumentParser('extract event signals')
    paser.add_argument('--feature_dir',required=True,help='directory of features')
    paser.add_argument('--outdir',required=True,help='output directory')
    paser.add_argument('--device',required=True,help='path to configuration files')
    # paser.add_argument('--model_path',required=False,help='path to pretrained model')
    paser.add_argument('--splits',nargs='*',help='splits to process')
    paser.add_argument('--config',required=False,help='path to configuration file')
    paser.add_argument('--ml',type=lambda x:(str(x).lower() == 'true'))
    paser.add_argument('--model_id',choices=['1','2','3','4','5','6','7'])
    paser.add_argument("--print_read_results",type=lambda x:(str(x).lower() == 'true'),default=False)

    args = paser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args

    os.system('mkdir -p ' + FLAGS.outdir)
    cfg = json.load(open(FLAGS.config,'r'))
    if FLAGS.model_id == '1':
        print("model 1: attention only")
        model = mlModel1().to(FLAGS.device)
        cfg = cfg['model1']
    elif FLAGS.model_id == '2':
        print("model 2: lstm")
        model = mlModel2().to(FLAGS.device)
        cfg = cfg['model2']
    elif FLAGS.model_id == '3':
        print("model 3: positional encoding")
        model = mlModel3().to(FLAGS.device)
        cfg = cfg['model3']
    elif FLAGS.model_id == '4':
        print("model 4: ltsm + positional encoding")
        model = mlModel4().to(FLAGS.device)
        cfg = cfg['model4']
    elif FLAGS.model_id == '5':
        print("model 5: attention only + feature extractor 1")
        model = mlModel5().to(FLAGS.device)
        cfg = cfg['model5']
    elif FLAGS.model_id == '6':
        print("model 6: attention only + feature extractor 2")
        model = mlModel6().to(FLAGS.device)
        cfg = cfg['model6']
    else:
        print('model 7: attention only + feature extractor 3')
        model = mlModel7().to(FLAGS.device)
        cfg = cfg['model7']



    if FLAGS.ml == True:
        print("integrated prediction")
        print(cfg['integrated'])
        print(cfg['integrated'])
        model.load_state_dict(torch.load(cfg['integrated']))
    else:
        print("independent prediction")
        model.ac4c.load_state_dict(torch.load(cfg['ac4c']))
        model.m1a.load_state_dict(torch.load(cfg['m1a']))
        model.m5c.load_state_dict(torch.load(cfg['m5c']))
        model.m6a.load_state_dict(torch.load(cfg['m6a']))
        model.m7g.load_state_dict(torch.load(cfg['m7g']))
        model.psi.load_state_dict(torch.load(cfg['psi']))

    if len(FLAGS.splits) != 2:
        splits = ['split' + str(s) for s in FLAGS.splits]
        print(splits)

    elif len(FLAGS.splits) == 2:
        s1 = int(FLAGS.splits[0])
        s2 = int(FLAGS.splits[1]) + 1
        splits = ['split' + str(s) for s in range(s1,s2)]
        print(splits)
    else:
        print("please specify correct splits")


    for split in tqdm(splits):

        # all prediction
        regs = pd.read_csv(FLAGS.feature_dir +'/' + split + '.csv')
        if regs.shape[1] == 6:
            regs.columns = ['read_id','seqnames','start','end','width','strand']
        else:
            print('labeled data')

        data = np.load(FLAGS.feature_dir +'/' + split + '.npz')

        seq = data['seq']
        stat = data['stat']
        bse = data['bse']

        data_loader = prepare_dataloader(seq,stat,bse)

        preds,attens = predict(data_loader,model)
        # np.savetxt(FLAGS.outdir + '/' + split + '.txt',preds.detach().cpu().numpy())
        # attens = attens.detach().cpu().numpy()

        regs = regs.loc[:,['read_id','seqnames','start','end','width','strand']]
        regs['pos'] = regs.apply(reg2base,axis=1)
        results = regs.explode('pos')
        results = results[['read_id','seqnames','pos','strand']].reset_index(drop=True)
        results['ac4c'] = find_mod_base(seq,attens,preds,mod_type=0,ref_base=1)
        results['m1a'] = find_mod_base(seq,attens,preds,mod_type=1,ref_base=0)
        results['m5c'] = find_mod_base(seq,attens,preds,mod_type=2,ref_base=1)
        results['m6a'] = find_mod_base(seq,attens,preds,mod_type=3,ref_base=0)
        results['m7g'] = find_mod_base(seq,attens,preds,mod_type=4,ref_base=2)
        results['psi'] = find_mod_base(seq,attens,preds,mod_type=5,ref_base=3)

        results.columns = ['read_id','seqnames','pos','strand','ac4c','m1a','m5c','m6a','m7g','psi']
        results = results.groupby(['read_id','seqnames','pos','strand']).aggregate('mean')
        results.reset_index(inplace=True)
        tmp = np.array(results[['ac4c','m1a','m5c','m6a','m7g','psi']])
        tmp[tmp==0] = np.nan
        results[['ac4c','m1a','m5c','m6a','m7g','psi']] = tmp

        # Nov30 updates, group read level results
        types = ['ac4c','m1a','m5c','m6a','m7g','psi']
        grouped = results.groupby('seqnames')
        for name,group in grouped:
            for type in types:
                os.system('mkdir -p ' + FLAGS.outdir + '/' + type)
                tmp = group[['read_id','seqnames','pos','strand',type]]
                tmp = tmp.dropna()
                if len(tmp) > 0:
                    path = FLAGS.outdir + '/' + type + '/' + name + '.csv'
                    if os.path.exists(path):
                        tmp.to_csv(path,index = False,mode='a',header=False)
                    else:
                        tmp.to_csv(path,index = False,mode='a')

        # if FLAGS.print_read_results==True:
        #     print("saving read-level results")
        #     results.to_csv(FLAGS.outdir + '/' + split + '_read_result.csv',index = 0)
        # else:
        #     print("skipping read-level results")

        # tmp = np.array(results[['ac4c','m1a','m5c','m6a','m7g','psi']])
        # tmp[tmp > 0.5] = 1
        # tmp[tmp <= 0.5] = 0
        # results[['ac4c','m1a','m5c','m6a','m7g','psi']] = tmp
        #
        # types = ['ac4c','m1a','m5c','m6a','m7g','psi']
        # for type in types:
        #     site_results = results.groupby(['seqnames','pos','strand']).agg(
        #         mod = (type,'sum'),
        #         coverage = (type,'count')
        #     )
        #     site_results.reset_index(inplace=True)
        #     site_results = site_results[site_results['coverage'] > 0]
        #     site_results.to_csv(FLAGS.outdir + '/' + split + '_' + type + '_result.csv',index = 0)

        # site_results = results.groupby(['seqnames','pos','strand']).agg(
        #     ac4c = ('ac4c',count_mod),
        #     m1a = ('m1a',count_mod),
        #     m5c = ('m5c',count_mod),
        #     m6a = ('m6a',count_mod),
        #     m7g = ('m7g',count_mod),
        #     psi = ('psi',count_mod),
        #     coverage = ('read_id','count')
        # )
        # site_results.reset_index(inplace=True)
        # site_results.to_csv(FLAGS.outdir + '/' + split + '_site_result.csv',index = 0)



