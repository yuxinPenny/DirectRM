# DirectRM

## 0 Brief introduction

DirectRM allows simultaneous detection of the six most abundant modifications, namely ac4C, m1A, m5C, m7G, m6A and pseudourdine, from nanopore direct RNA sequencing data. It employs a two-stage detection framework: 

- De novo detection: identification of potential modified kmers regardless of their modification types and positions.
- Modification type(s) and position(s) inference: the modification class(es) and its (their) position(s) within the modified k-mers are determined with an attention-based multi-instance multi-label framework.

It can confidently applied to diverse biological contexts and provide evaluable information to the complex epitranscriptome.

## 1 Features

- Support both SQK-RNA002 and newly introduced SQK-RNA004 sequencing data.
- Deep learning-based, can deal with data of high complexity (e.g., SQK-RNA004 data).
- Can provide read-level, transcriptome-level, or genome level modification profiles.
- Optimized data processing pipeline. 
  - Users could analyze data in parallel batches, which significantly reducing memory and computational demands. 
  - Less time and storage space required. 



## 2 Before running DirectRM

### Platforms

All processes related to DirectRM were tested on 

- Platform: Linux x86_64
- GPU: Nvidia GPUs
- CPUs

### Softwares

- Dorado: https://github.com/nanoporetech/dorado/tree/release-v0.6
- Remora: https://github.com/nanoporetech/remora
- Pytorch
- Python3
- Basic python packages: `math`, `os`, `re`, `sys`, `csv`, `json`, `argparse`, `tqdm`, `numpy`, `pandas`, `random`

### Input data

- ##### Raw data in Pod5 format

  Please make sure your data are in Pod5 format. If your data is in Fast5 format, please use the python module pod5 (https://pod5-file-format.readthedocs.io/en/latest/) to convert it to Pod5.

  ```bash
  pod5 convert fast5 ./input/*.fast5 --output outputs/
  ```

- ##### Reference sequence

- ##### Target regions

  Please provide a **CSV** file of regions you want to analyze. The table should contain following columns, and each line represents one region. The coordinates should be consistent to above reference file.
  
  | seqnames | start | end  | width | strand |
  | -------- | ----- | ---- | ----- | ------ |
  | chr1     | 207   | 307  | 101   | +      |
  | chr4     | 620   | 987  | 368   | +      |

## 3 Getting Started

### 3.1 Base calling and alignment

**Dorado** is used to perform base calling and alignment simultaneously. 

To optimize the memory usage, we process each `POD5` file as an independent split-parallel unit for both base calling and feature extraction. The script below will rearch the target directory for `.pod5` files, and for each `.pod5` file:
1) Perform base calling and alignment.
2) Compress, sort, and index the alignment file. 
3) Delete the intermediate SAM files to save space.

Each input `.pod5` file produces a corresponding `_sorted.bam` file with the same base name.

```bash
POD5_DIR="./pod5"
BAM_DIR="./bam"
REF="./reference.fa"
DEVICE="cuda:0"

mkdir -p ${BAM_DIR}
for pod5_fl in ${POD5_DIR}/*.pod5; do
  
    base_name=$(basename ${pod5_fl} .pod5)
    bam_fl=${BAM_DIR}/${base_name}.bam

    ./dorado-0.6.2-linux-x64/bin/dorado basecaller \
    ./dorado-0.6.2-linux-x64/model/rna004_130bps_hac@v3.0.1 \
    ${pod5_fl} --emit-sam --emit-moves \
    --reference ${REF} \
    --device ${DEVICE} > ${BAM_DIR}/${base_name}.sam

    samtools view -b -S ${BAM_DIR}/${base_name}.sam > ${BAM_DIR}/${base_name}.bam

    samtools sort ${BAM_DIR}/${base_name}.bam > \
    ${BAM_DIR}/${base_name}_sorted.bam

    samtools index ${BAM_DIR}/${base_name}_sorted.bam

    rm -rf ${BAM_DIR}/${base_name}.sam
    rm -rf ${BAM_DIR}/${base_name}.bam
done
```

Please download available Dorado models with: 
```
dorado download --model all
```
- `rna002_70bps_hac@v3` for data sequenced with SQK-RNA002 kit,
- `rna004_130bps_hac@v3.0.1` for data sequenced with SQK-RNA004 kit.

### 3.2 Feature extraction

**Remora** is used to extract features from above generated BAM file and Pod5 file. Remora will take the Pod5 file and corresponding bam file as input to form a Remora object. Then, signal refinement (re-squiggle) will be performed to correct the signal data to the reference sequence. Three groups of features will be extracted from the Remora object: 

1. kmer sequence: the 9-mer or 5-mer sequence will be extracted and encoded with One-Hot encoding
2. Signal features: for each nucleotide within the kmer, the minimum, maximum, 1st and 3rd quartiles, mean, median, standard deviation, and length of signal events will be extracted.
3. Base call error features: mismatch, insertion, deletion, and quality.

**Note**: To use remora, we recommend to create an independent conda environment with **python=3.9** for it.

#### Read sampling
To reduce unnecessary computation, users can specify a minimum and maximum coverage threshold for each region and then perform read sampling.
```bash
BAM_dir="./bam"
REG="./reg.csv"
READs="./sampled_reads.txt"

bam_fls=(${BAM_dir}/*.bam)
tmp=(${bam_fls[@]##*/})
split_array=(${tmp[@]%.bam})

python3 ./DirectRM/scripts/sampling.py \
--bam ${BAM_dir} \
--reg ${REG}\
-o ${READs} \
--splits ${split_array[@]} \
--min_coverage 30 \
--max_coverage 150
```
#### Feature extraction
```bash
POD5_DIR="./pod5"
BAM_DIR="./bam"
FEATURE_DIR='./data'
REG="./reg.csv"

mkdir -p ${FEATURE_DIR}

pod5_fls=(${POD5_DIR}/*.pod5)
tmp=(${pod5_fls[@]##*/})
split_array=(${tmp[@]%.pod5})


python3 ./DirectRM/scripts/feature_extraction.py \
--pod5_dir ${POD5_DIR} \
--bam ${BAM_DIR} \
--reg ${REG} \
--level <9mer_levels_v1.txt> \
-o {FEATURE_DIR} \
--splits ${splits[@]} \
--read_ids ${READs} \
--step 5 --kmer 9
```

- `--level`: the k-mers levels table. `./DirectRM/5mer_levels_v1.txt` for data sequenced with SQK-RNA002 kit, `./DirectRM/9mer_levels_v1.txt` for data sequenced with SQK-RNA004 kit.
- `--step`, `--kmer`: it will extract kmer features with a sliding window. `--kmer` specify the window size [5, 9], `--step` specify the step size of the sliding window.

**Note**: `feature_extraction.py` run in single-thread mode by default. For parallel execution, users can manually split `split_array` into multiple bacthes and submit multiple independent jobs. Each batch is processed separately, which allows flexible scaling according to availble computing resources.

#### Output
1. CSV files: coordinates of kmers

| read_id                              | seqnames | start     | end       | width | strand |
| ------------------------------------ | -------- | --------- | --------- | ----- | ------ |
| 4377a3e0-895e-4ee4-8eda-872a03e868aa | chr3     | 186788257 | 186788266 | 9     | +      |
| 4377a3e0-895e-4ee4-8eda-872a03e868aa | chr3     | 186788262 | 186788271 | 9     | +      |

2. npz files: features of kmers
   - seq: one hot encoded sequence 
   - stat: mean, median, and etc of signal events 
   - level: expected kmer levels
   - bse: base call errors

### 3.3 De novo modification detection

This step aims to detect modified kmers

```bash
FEATURE_DIR='./data'
OUT_DIR='./output'
DEVICE="cuda:0"

npz_fls=(${FEATURE_DIR}/*.npz)
tmp=(${npz_fls[@]##*/})
split_array=(${tmp[@]%.npz})

python3 ./DirectRM/scripts/denovo_inference.py \
--feature_dir ${FEATURE_DIR} \
--outdir ${OUT_DIR} \
--model_path <denovo_model> \
--splits ${split_array[@]} \
--device ${DEVICE}
```

#### Available models

We provided three binary de novo modification models, each model was trained with different label sources:

- **id1**: Wilcoxon test was used to compare the observed kmer signals and expected levels. K-mers with P-value < 0.01 (significantly differentiated) were labeled as positive, and vice versa.
- **id2**: K-mers with base call error frequence > 0 were labeled as positive, and vice versa.
- **id3 (recommended)**: Intersection of above two.

#### Parameter explaination

- `--model_path`: path to de novo detection model
  - `./DirectRM/model/RNA004/id3_binary/model.pt` for data sequenced with SQK-RNA004 kit
  - `./DirectRM/model/RNA002/id3_binary/model.pt` for data sequenced with SQK-RNA002 kit

#### Output
npy file speficy the probability of being modified

### 3.4 Modification type and position inference

This step aims to identify the modification type(s) and its(their) position within the modified kmers.

```bash
FEATURE_DIR='./data'
OUT_DIR='./output'
DEVICE="cuda:0"

npz_fls=(${FEATURE_DIR}/*.npz)
tmp=(${npz_fls[@]##*/})
split_array=(${tmp[@]%.npz})

python3 /gpfs/work/bio/yuxinzhang17/GlycoRNA/code/inference.py \
--feature_dir ${FEATURE_DIR} \
--outdir ${OUT_DIR} \
--device ${DEVICE} --splits ${split_array[@]} --ml True \
--model_dir ./DirectRM/model/RNA004 \
--model_id 5 
```

#### Available models

We provided four model artchitecture:

- model1: {attention} + {fcnn}
- model2: {attention + LSTM} + {fcnn}
- model3: {attention + positional encoding} + {fcnn}
- model4: {attention + positional encoding + LSTM} + {fcnn}
- model5 **(Recommended)**: {attention} + {biLSTM feature extractor} + {fcnn}
- model6: {attention} + {biLSTM + CNN feature extractor} + {fcnn}
- model7: {attention} + {CNN feature extractor} + {fcnn}
- model8 **(Recommended)**: {attention} + {shared biLSTM feature extractor} + {fcnn}
  - model8 only support integrated detection mode

#### Parameter explaination
- `--ml`: whether to use integrated detection model (`True`) or independent detection model (`False`)
- `--model_id`: id of model structure

#### Output

Read level prediction results for each class, grouped by modification type and chromsome/transcripts

| read_id                              | seqnames | pos(istion) | strand | ac4c(_probability) |
| ------------------------------------ | -------- | ----------- | ------ | ------------------ |
| 00172a9f-1aab-4044-a20d-87cc753ba3d0 | chr1     | 39004867    | +      | 0.3917313          |
| 00172a9f-1aab-4044-a20d-87cc753ba3d0 | chr1     | 39004882    | +      | 0.16477184         |

### 3.5 Read-level results to Site-level

This step aggregates the read-level results to provide site-level results

```bash
OUT_DIR='./output'

python3 /gpfs/work/bio/yuxinzhang17/DirectMultiRM/code/v07/read2site.py \
--indir ${OUT_DIR} \
--outdir ${OUT_DIR} \
--delete_read True
```

#### Parameter explaination
- `--delete_read`: whether to delete read-level prediction results (`True`) or not (`False`)

#### Output

Site-level results for each modification class

| seqnames | pos(istion) | strand | max_prob   | noisyor_prob | count | coverage |
| -------- | ----------- | ------ | ---------- | ------------ | ----- | -------- |
| chr22    | 15787218    | +      | 0.31133258 | 0.31133257   | 0     | 32       |
| chr22    | 15787232    | +      | 0.16512871 | 0.16512871   | 0     | 35       |

## 4 Getting help

We appreciate your feedback and questions. You can report an error or suggestions related to DirectRM as an issue on github.



