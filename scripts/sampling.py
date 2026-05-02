from itertools import count
import random
import remora
from remora import io, refine_signal_map, util
import pod5
import polars as pl
import pandas as pd
import numpy as np
import os,sys,argparse,re,tqdm,multiprocessing

if __name__ == '__main__':
    paser = argparse.ArgumentParser('extract event signals')
    paser.add_argument('--bam',required=True,help='bam file dir')
    paser.add_argument('--reg',required=True,help='csvs for regions of interested')
    paser.add_argument('--splits',nargs='*',help='splits to process')
    paser.add_argument('-o','--output',required=True,help='output directory')
    paser.add_argument('--min_coverage',default=30,help='minimum coverage per reg')
    paser.add_argument('--max_coverage',default=150,help='maximum coverage per reg')
    args = paser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args
    regs = pd.read_csv(FLAGS.reg)

    splits = FLAGS.splits

    sample_reads = [[] for _ in range(len(regs))]
    for split in splits:
        try:
            bam_path = FLAGS.bam + "/" + split + "_sorted.bam"
            bam_fh = io.ReadIndexedBam(bam_path)
            for i in range(0, len(regs)):
                try:
                    items = bam_fh.fetch(ctg=regs.loc[i,'seqnames'],
                                         strand=regs.loc[i,'strand'],
                                         start=regs.loc[i,'start'], end=regs.loc[i,'end'])
                    for item in items:
                        sample_reads[i].append(str(item).split('\t')[0])
                except Exception as e:
                    print(str(e))
                    continue
        except Exception as e:
            print(str(e))
            continue

    filtered_reads = []
    for i, reg_reads in enumerate(sample_reads):
        if len(reg_reads) <= int(FLAGS.min_coverage):
            continue
        elif len(reg_reads) >= int(FLAGS.max_coverage):
            filtered_reads.extend(random.sample(reg_reads, int(FLAGS.max_coverage)))
        else:
            filtered_reads.extend(reg_reads)


    sample_reads = list(set(filtered_reads))

    with open(FLAGS.output, 'w') as f:
        for item in sample_reads:
            f.write(f"{item}\n")