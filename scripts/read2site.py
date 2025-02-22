import math, os, re, sys, csv,json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == '__main__':
    paser = argparse.ArgumentParser('pool read results to site level')
    paser.add_argument('--indir',required=True,help='input directory')
    paser.add_argument('--outdir',required=True,help='output directory')
    args = paser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args

    os.system('mkdir -p ' + FLAGS.outdir)

    types = ['m6a','ac4c','m1a','m5c','m7g','psi']
    for type in types:
        chroms = os.listdir(FLAGS.indir+'/'+type)
        for chorm in tqdm(chroms):
            df = pd.read_csv(FLAGS.indir+'/'+type+'/'+chorm)
            tmp = df.groupby(['seqnames','pos','strand']).agg(
                max_prob = (type,'max'),
                noisyor_prob = (type,lambda x:(1-np.prod(1-x,axis=0))),
                count = (type,lambda x:(sum(x>0.5))),
                coverage = (type,'count')
            )
            tmp.reset_index(inplace=True)
            if len(tmp) > 0:
                path = FLAGS.outdir+'/'+type+'.csv'
                if os.path.exists(path):
                    tmp.to_csv(path,index=False,mode='a',header=False)
                else:
                    tmp.to_csv(path,index=False,mode='a')

