import os, sys, json
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.dataset import pmlDataset
from utils.model import *
from pathlib import Path
import json
import hashlib
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
    # gpu
    if FLAGS.max == False:
        # print("preds will be assigned to positions with scores larger than mean")
        mod_attens = attens[:,mod_type,:]
        mod_attens[seq[:,:,ref_base]==0] = float('nan')
        rm = torch.nanmean(mod_attens,axis=1,keepdims=True)
        mask = mod_attens > rm
        mod_arr = mask.to(torch.int16)
        base_results = mod_arr *(preds[:,mod_type].reshape(-1,1))
        base_results = base_results.reshape(-1,1)
        return base_results.detach().cpu().numpy()
    else:
        # print("preds will be assigned to positions with maximum scores")
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
    paser.add_argument('--device',required=True,help='device')
    paser.add_argument('--splits',nargs='*',help='splits to process')
    paser.add_argument('--model_dir',required=False,help='model dirs')
    paser.add_argument('--ml',type=lambda x:(str(x).lower() == 'true'))
    paser.add_argument('--model_id',choices=['1','2','3','4','5','6','7','8'],default='1')
    paser.add_argument('--max',type=lambda x:(str(x).lower() == 'true'),default=True)

    args = paser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args

    os.system('mkdir -p ' + FLAGS.outdir)
    # cfg = json.load(open(FLAGS.config,'r'))
    if FLAGS.model_id == '1':
        print("model 1: attention only")
        model = mlModel1().to(FLAGS.device)
    elif FLAGS.model_id == '2':
        print("model 2: lstm")
        model = mlModel2().to(FLAGS.device)
    elif FLAGS.model_id == '3':
        print("model 3: positional encoding")
        model = mlModel3().to(FLAGS.device)
    elif FLAGS.model_id == '4':
        print("model 4: ltsm + positional encoding")
        model = mlModel4().to(FLAGS.device)
    elif FLAGS.model_id == '5':
        print("model 5: attention only + feature extractor 1")
        model = mlModel5().to(FLAGS.device)
    elif FLAGS.model_id == '6':
        print("model 6: attention only + feature extractor 2")
        model = mlModel6().to(FLAGS.device)
    elif FLAGS.model_id == '7':
        print("model 7: attention only + feature extractor 3")
        model = mlModel7().to(FLAGS.device)
    else:
        print('model 8: attention only + shared feature extractor 1')
        model = mlModel8().to(FLAGS.device)



    if FLAGS.ml == True:
        print("integrated prediction")
        model.load_state_dict(torch.load(FLAGS.model_dir+'/ml'+FLAGS.model_id+'/model.pt'))
    else:
        print("independent prediction")
        model.ac4c.load_state_dict(torch.load(FLAGS.model_dir+'/ac4c_m'+FLAGS.model_id+'/model.pt'))
        model.m1a.load_state_dict(torch.load(FLAGS.model_dir+'/m1a_m'+FLAGS.model_id+'/model.pt'))
        model.m5c.load_state_dict(torch.load(FLAGS.model_dir+'/m5c_m'+FLAGS.model_id+'/model.pt'))
        model.m6a.load_state_dict(torch.load(FLAGS.model_dir+'/m6a_m'+FLAGS.model_id+'/model.pt'))
        model.m7g.load_state_dict(torch.load(FLAGS.model_dir+'/m7g_m'+FLAGS.model_id+'/model.pt'))
        model.psi.load_state_dict(torch.load(FLAGS.model_dir+'/psi_m'+FLAGS.model_id+'/model.pt'))

    splits = FLAGS.splits

    # Feb 6 updates: print prediction

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
        del data

        data_loader = prepare_dataloader(seq,stat,bse)

        preds,attens = predict(data_loader,model)

        del stat
        del bse

        # mod_status = np.load(FLAGS.feature_dir +'/' + split + '_mod.npy')
        # preds[mod_status < 0.5] = 0

        regs = regs.loc[:,['read_id','seqnames','start','end','width','strand']]
        regs['pos'] = regs.apply(reg2base,axis=1)
        results = regs.explode('pos')
        del regs
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
        for type in types:
            Path(FLAGS.outdir + "/" + type).mkdir(parents=True, exist_ok=True)
        if len(results['seqnames'].unique()) <= 200:
            for type in types:
                tmp = results[['read_id','seqnames','pos','strand',type]]
                tmp = tmp.dropna()
                if len(tmp) > 0:
                    grouped = tmp.groupby('seqnames')
                    for name, group in grouped:
                        if len(group) > 0:
                            path = Path(FLAGS.outdir + "/" + type + '/' + name + '.csv')
                            write_header = not path.exists()
                            group.to_csv(path, mode = 'a',index=False, header=write_header)
        else:
            print("too much seqnames, try to group some seqnames")
            metadata_path = Path(FLAGS.outdir + "/metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            def get_file_id(seqname: str) -> int:
                if seqname not in metadata:
                    file_id = int(hashlib.md5(seqname.encode()).hexdigest(), 16) % 100
                    metadata[seqname] = file_id
                return metadata[seqname]

            results['file_id'] = results['seqnames'].map(get_file_id)
            grouped = results.groupby('file_id')
            for file_id, group in grouped:
                for type in types:
                    tmp = group[['read_id','seqnames','pos','strand',type]]
                    tmp = tmp.dropna()
                    if len(tmp) > 0:
                        path = Path(FLAGS.outdir + "/" + type + '/' + str(file_id) + '.csv')
                        write_header = not path.exists()
                        tmp.to_csv(path, mode = 'a',index=False, header=write_header)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

