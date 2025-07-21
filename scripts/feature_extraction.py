import remora
from remora import io, refine_signal_map, util
import pod5
import polars as pl
import pandas as pd
import numpy as np
import os,sys,argparse,re,tqdm,multiprocessing
def extract_base_call_error(io_read):
    cigar = io_read.cigar
    seq_len = io_read.seq_len
    ref_len = len(io_read.ref_seq)
    seq_qual = io_read.full_align['qual']
    seq_qual = [ord(char) - 33 for char in seq_qual]
    seq_qual = (seq_qual - np.mean(seq_qual)) / np.std(seq_qual)
    inser = np.zeros(ref_len)
    delete = np.zeros(ref_len)
    mismatch = np.zeros(ref_len)
    qual = np.zeros(ref_len)
    ref_id = 0
    seq_id = 0
    for i in range(0, len(cigar)):
        tag = cigar[i][0]
        num = cigar[i][1]
        if tag == 0:
            # match
            qual[ref_id:ref_id + num] = seq_qual[seq_id:seq_id + num]
            seq1 = io_read.ref_seq[ref_id:ref_id + num]
            seq2 = io_read.seq[seq_id:seq_id + num]
            if seq1 != seq2:
                for j in range(0, num):
                    if seq1[j] != seq2[j]:
                        mismatch[ref_id + j] = 1
            ref_id = ref_id + num
            seq_id = seq_id + num
        elif tag == 1:
            # insertion to ref
            inser[ref_id:ref_id + num] = 1
            seq_id = seq_id + num
        elif tag == 2:
            delete[ref_id:ref_id + num] = 1
            ref_id = ref_id + num
        elif tag == 3:
            delete[seq_id:seq_id + num] = 1
            seq_id = seq_id + num
        elif tag == 4:
            # clipped
            # ref_pos[seq_id:seq_id+num] = np.nan
            seq_id = seq_id + num
        else:
            pass

    ref_bse = {'inser':inser,'delete':delete,'mismatch':mismatch,'qual':qual}
    return ref_bse

def quantiles(sig, seq_to_sig):
    tmp = [sig[seq_to_sig[i]:seq_to_sig[i + 1]] for i in range(0, len(seq_to_sig) - 1)]
    tmp = [h if h.size else np.nan for h in tmp]
    sig_min = np.array([np.min(h) for h in tmp]).astype(np.float32)
    sig_1q = np.array([np.quantile(h, 0.25) for h in tmp]).astype(np.float32)
    sig_3q = np.array([np.quantile(h, 0.75) for h in tmp]).astype(np.float32)
    sig_median = np.array([np.median(h) for h in tmp]).astype(np.float32)
    sig_max = np.array([np.max(h) for h in tmp]).astype(np.float32)
    return {'min': sig_min, 'q1': sig_1q, 'median': sig_median, 'q3': sig_3q, 'max': sig_max}

def extract_sig_stat(io_read):
    ref_reg = io.RefRegion(ctg=io_read.ref_reg.ctg,
                           strand=io_read.ref_reg.strand,
                           start=io_read.ref_reg.start,
                           end=io_read.ref_reg.end)
    sig_mean_sd_dwell = io_read.compute_per_base_metric("dwell_mean_sd", ref_anchored=True, region=ref_reg)
    sig_quantiles = io_read.compute_per_base_metric(metric_func=quantiles, ref_anchored=True, region=ref_reg)
    sig_mean_sd_dwell.update(sig_quantiles)
    return sig_mean_sd_dwell

def onehot_encoding(seq):
    alphabet = 'ACGT'
    char2int = dict((c,i) for i, c in enumerate(alphabet))
    int_seq = [char2int[n] for n in seq]
    int_seq = np.array(int_seq)
    onehot_seq = np.zeros((int_seq.size,4))
    onehot_seq[np.arange(int_seq.size), int_seq] = 1
    return onehot_seq

def extract_kmer_features(io_read):
    try:
        tmp = regs.loc[(regs['seqnames']==io_read.ref_reg.ctg)& (regs['strand'] == io_read.ref_reg.strand)]
        ref_pos = set(range(io_read.ref_reg.start,io_read.ref_reg.end))
        def findOverlaps(df):
            peak_pos = set(range(df['start']-1,df['end']-1))
            num = len(ref_pos.intersection(peak_pos))
            return num
        overlap = tmp.apply(findOverlaps,axis=1)
        tmp = tmp[overlap>0]
        tmp.reset_index(inplace=True,drop=True)

        peak_pos= []
        for i in range(len(tmp)):
            peak_pos.extend(list(range(tmp.loc[i,'start'],tmp.loc[i,'end'])))
        peak_pos = list(set(peak_pos))

        ref_pos = list(range(io_read.ref_reg.start,io_read.ref_reg.end))

        ref_bse = extract_base_call_error(io_read)
        sig_stat = extract_sig_stat(io_read)

        if io_read.ref_reg.strand == "+":
            ref_bse = np.vstack([ref_bse['inser'], ref_bse['delete'],ref_bse['mismatch'],ref_bse['qual']])
            ref_bse = ref_bse.transpose()

            sig_stat = np.vstack([sig_stat['mean'], sig_stat['sd'], sig_stat['dwell'],
                                  sig_stat['min'], sig_stat['q1'], sig_stat['median'], sig_stat['q3'],
                                  sig_stat['max']])
            sig_stat = sig_stat.transpose()

            onehot_seq = onehot_encoding(io_read.ref_seq)

            model_level = sig_map_refiner.extract_levels(util.seq_to_int(io_read.ref_seq))
        elif io_read.ref_reg.strand == "-":
            ref_bse = np.vstack([ref_bse['inser'][::-1], ref_bse['delete'][::-1],
                                 ref_bse['mismatch'][::-1],ref_bse['qual'][::-1]])
            ref_bse = ref_bse.transpose()

            sig_stat = np.vstack([sig_stat['mean'][::-1], sig_stat['sd'][::-1], sig_stat['dwell'][::-1],
                                  sig_stat['min'][::-1], sig_stat['q1'][::-1], sig_stat['median'][::-1],
                                  sig_stat['q3'][::-1],sig_stat['max'][::-1]])
            sig_stat = sig_stat.transpose()

            onehot_seq = onehot_encoding(io_read.ref_seq)[::-1]

            model_level = sig_map_refiner.extract_levels(util.seq_to_int(io_read.ref_seq))[::-1]

        overlap_ind = np.where(np.isin(ref_pos,peak_pos))[0].tolist()

        seq,stat,level,bse,coord = [],[],[],[],[]
        k = int(FLAGS.kmer)
        s = int(FLAGS.step)
        for i in overlap_ind[::s]:
            try:
                if io_read.ref_reg.strand == "+":
                    kmer_seq = onehot_seq[i:(i+k)]
                    kmer_level = model_level[i:(i+k)]
                elif io_read.ref_reg.strand == "-":
                    kmer_seq = onehot_seq[i:(i+k)][::-1]
                    kmer_level = model_level[i:(i+k)][::-1]
                else:
                    print("unknown strand")
                kmer_stat = sig_stat[i:(i+k)]
                if np.any(np.isnan(kmer_stat)):
                    aaaa = 0
                else:
                    kmer_bse = ref_bse[i:(i+k)]
                    kmer_coord = [io_read.read_id,io_read.ref_reg.ctg,ref_pos[i],ref_pos[i+k],
                                  k,io_read.ref_reg.strand]

                    seq.append(kmer_seq)
                    stat.append(kmer_stat)
                    level.append(kmer_level)
                    bse.append(kmer_bse)
                    coord.append(kmer_coord)

            except Exception as e:
                # print(str(e))
                pass

        return seq,stat,level,bse,coord

    except Exception as e:
        # print(str(e))
        pass

def iterate_pod5(pod5_dr,bam_fh,read_ids):
    seq2,stat2,level2,bse2,coord2 = [],[],[],[],[]
    failed_num = 0
    for read_id in read_ids:
        try:
            pod5_read = pod5_dr.get_read(read_id)
            bam_read = bam_fh.get_first_alignment(read_id)
            io_read = io.Read.from_pod5_and_alignment(pod5_read,bam_read)
            io_read.set_refine_signal_mapping(sig_map_refiner,ref_mapping=True)
            if int(io_read.seq_len) > 500:
                seq3,stat3,level3,bse3,coord3 = extract_kmer_features(io_read)
                seq2.extend(seq3)
                stat2.extend(stat3)
                level2.extend(level3)
                bse2.extend(bse3)
                coord2.extend(coord3)
            else:
                failed_num += 1
        except Exception as e:
            print(str(e))
            pass
    print(str(failed_num) + " failed")
    return seq2,stat2,level2,bse2,coord2


def iterate_split(split):
    bam_path = FLAGS.bam + "/" + split + "_sorted.bam"
    bam_fh = io.ReadIndexedBam(bam_path)

    bam_reads = []
    for i in range(0, len(regs)):
        try:
            items = bam_fh.fetch(ctg=regs.loc[i,'seqnames'],
                                 strand=regs.loc[i,'strand'],
                                 start=regs.loc[i,'start'], end=regs.loc[i,'end'])
            for item in items:
                bam_reads.append(str(item).split('\t')[0])
        except Exception as e:
            print(str(e))
            pass
    bam_reads = list(set(bam_reads))

    pod5_dir = FLAGS.pod5_dir + '/' + split
    fls = []
    for current_dirs,subdirs,files in os.walk(pod5_dir + '/'):
        for fl in files:
            if re.search(r'\.pod5',fl):
                relative_path = os.path.join(current_dirs,fl)
                absolute_path = os.path.abspath(relative_path)
                fls.append(absolute_path.rstrip())

    seq4,stat4,level4,bse4,coord4 = [],[],[],[],[]
    for fl in tqdm.tqdm(fls):
        pod5_dr = pod5.DatasetReader(fl)
        pod5_reads = [x for x in pod5_dr.read_ids]
        read_ids = list(set(pod5_reads).intersection(set(bam_reads)))
        seq5,stat5,level5,bse5,coord5 = iterate_pod5(pod5_dr,bam_fh,read_ids)
        seq4.extend(seq5)
        stat4.extend(stat5)
        level4.extend(level5)
        bse4.extend(bse5)
        coord4.extend(coord5)
    np.savez(file = FLAGS.output + "/" + split + ".npz",
             seq = np.array(seq4),stat = np.array(stat4),level = np.array(level4),bse = np.array(bse4))
    coord4 = pd.DataFrame(coord4)
    coord4.to_csv(FLAGS.output +"/" + split + ".csv",index=0)

if __name__ == '__main__':
    paser = argparse.ArgumentParser('extract event signals')
    paser.add_argument('--pod5_dir',required=True,help='pod5 dir')
    paser.add_argument('--bam',required=True,help='bam file dir')
    paser.add_argument('--reg',required=True,help='csvs for regions of interested')
    paser.add_argument('--level',required=True,help='kmer level tables in txt format')
    paser.add_argument('-o','--output',required=True,help='output directory')
    paser.add_argument('-t','--threads',default=5,help='number of threads')
    paser.add_argument('--splits',nargs='*',help='splits to process')
    paser.add_argument('--kmer',default=9,help='kmer size')
    paser.add_argument('--step',default=5,help='step size')
    args = paser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args
    os.system('mkdir -p ' + FLAGS.output)

    level_table = FLAGS.level
    sig_map_refiner = refine_signal_map.SigMapRefiner(
        kmer_model_filename=level_table,
        do_rough_rescale=True,
        scale_iters=0,
        do_fix_guage=True
    )
    regs = pd.read_csv(FLAGS.reg)

    if len(FLAGS.splits) > 2:
        splits = ['split' + str(s) for s in FLAGS.splits]
        print(splits)
        for split in tqdm.tqdm(splits):
            iterate_split(split)
    elif len(FLAGS.splits) == 2:
        s1 = int(FLAGS.splits[0])
        s2 = int(FLAGS.splits[1]) + 1
        splits = ['split' + str(s) for s in range(s1,s2)]
        print(splits)
        for split in tqdm.tqdm(splits):
            iterate_split(split)
    elif len(FLAGS.splits) == 1:
        print(FLAGS.splits[0])
        iterate_split('split'+FLAGS.splits[0])
    else:
        print("please specify correct splits")


