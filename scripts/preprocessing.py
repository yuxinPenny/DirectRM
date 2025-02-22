import os,sys,argparse
import re

paser = argparse.ArgumentParser('data proprocessing')
paser.add_argument('-i','--input',required=True,help='pod5 directory')
paser.add_argument('--new_dir',required=False,help='splitted new pod5 directory')
paser.add_argument('-o','--output',required=True,help='output bam directory')
paser.add_argument('--dorado',required=True,help='path to dorado package')
paser.add_argument('--ref',required=True,help='path to reference genome')
paser.add_argument('--model',required=True,help='path to model')
paser.add_argument('--device',required=True,help='cuda device')
paser.add_argument('--splitted',type=lambda x:(str(x).lower() == 'true'),default=False,help='whether the pod5 files have been splitted')
# paser.add_argument("--delete",default=True,help="whether to delete sam files")
args = paser.parse_args(sys.argv[1:])

global FLAGS
FLAGS = args

os.system('mkdir -p ' + FLAGS.output)
if FLAGS.splitted == False:
    total_fl = []
    pod5_dir = FLAGS.input + '/'
    for current_dir, subdirs, files in os.walk(pod5_dir):
        for filename in files:
            if re.search(r'\.pod5',filename):
                relative_path = os.path.join(current_dir,filename)
                absolute_path = os.path.abspath(relative_path)
                total_fl.append(absolute_path.rstrip())

    # split pod5 directory into multiple directory
    j = 0
    for i in range(0,len(total_fl),20):
        split_dir = FLAGS.new_dir + '/split' + str(j)
        os.system('mkdir -p ' + split_dir)
        split_fl = total_fl[i:(i+20)]
        for fl in split_fl:
            os.system('mv ' + fl + ' ' + split_dir)

        sam_fl = FLAGS.output + '/split' + str(j) + '.sam'
        bam_fl = FLAGS.output + '/split' + str(j) + '.bam'
        sorted_bam = FLAGS.output + '/split' + str(j) + '_sorted.bam'

        doroda_command = ' '.join([FLAGS.dorado,'basecaller',FLAGS.model,split_dir,'--emit-sam','--emit-moves','--reference',FLAGS.ref,'--device',FLAGS.device,'>',sam_fl])
        tobam_command = ' '.join(['samtools view -b -S',sam_fl,'>',bam_fl])
        sort_command = ' '.join(['samtools sort',bam_fl,'>',sorted_bam])
        index_command = ' '.join(['samtools index',sorted_bam])
        del_sam_command = ' '.join(['rm -rf',sam_fl])
        del_bam_commnd = ' '.join(['rm -rf',bam_fl])
        os.system(doroda_command)
        os.system(tobam_command)
        os.system(sort_command)
        os.system(index_command)
        os.system(del_sam_command)
        os.system(del_bam_commnd)
        j = j + 1

else:
    print('base call to new reference')
    splits = []
    pod5_dir = FLAGS.input + '/'
    filenames = os.listdir(pod5_dir)
    for filename in filenames:
        if re.search(r'split',filename):
            splits.append(filename)

    for split in splits:
        split_dir = FLAGS.input + '/' + split
        sam_fl = FLAGS.output + '/' + split + '.sam'
        bam_fl = FLAGS.output + '/' + split + '.bam'
        sorted_bam = FLAGS.output + '/' + split + '_sorted.bam'

        doroda_command = ' '.join([FLAGS.dorado,'basecaller',FLAGS.model,split_dir,'--emit-sam','--emit-moves','--reference',FLAGS.ref,'--device',FLAGS.device,'>',sam_fl])
        # doroda_command = ' '.join([FLAGS.dorado,'basecaller',FLAGS.model,split_dir,'--emit-sam','--emit-moves','--mm2-preset splice','--reference',FLAGS.ref,'--device',FLAGS.device,'>',sam_fl])
        tobam_command = ' '.join(['samtools view -b -S',sam_fl,'>',bam_fl])
        sort_command = ' '.join(['samtools sort',bam_fl,'>',sorted_bam])
        index_command = ' '.join(['samtools index',sorted_bam])
        del_sam_command = ' '.join(['rm -rf',sam_fl])
        del_bam_commnd = ' '.join(['rm -rf',bam_fl])
        os.system(doroda_command)
        os.system(tobam_command)
        os.system(sort_command)
        os.system(index_command)
        os.system(del_sam_command)
        os.system(del_bam_commnd)


# sh_file = open(FLAGS.sh,'a')
# for pod_fl in total_fl[0:2]:
#     sam_fl = pod_fl.replace(FLAGS.input,FLAGS.output)
#     sam_fl = sam_fl.replace('.pod5','.sam')
#     bam_fl = sam_fl.replace('.sam','.bam')
#     sorted_bam = bam_fl.replace('.bam','.sorted.bam')
#     doroda_command = ' '.join([FLAGS.dorado,'basecaller',FLAGS.model,pod_fl,'--emit-sam','--emit-moves','--reference',FLAGS.ref,'--device',FLAGS.device,'>',sam_fl])
#     tobam_command = ' '.join(['samtools view -b -S',sam_fl,'>',bam_fl])
#     sort_command = ' '.join(['samtools sort',bam_fl,'>',sorted_bam])
#     index_command = ' '.join(['samtools index',sorted_bam])
#     del_sam_command = ' '.join(['rm -rf',sam_fl])
#     del_bam_commnd = ' '.join(['rm -rf',bam_fl])
#     os.system(doroda_command)
#     os.system(tobam_command)
#     os.system(sort_command)
#     os.system(index_command)
#     os.system(del_sam_command)
    # sh_file.write(doroda_command+'\n')
    # sh_file.write(tobam_command+'\n')
    # sh_file.write(sort_command+'\n')
    # sh_file.write(index_command+'\n')
    # sh_file.write(del_sam_command+'\n')





