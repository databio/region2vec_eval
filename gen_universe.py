import numpy as np
import pandas as pd
import os
import argparse
from collections import defaultdict
from pybedtools import BedTool
import matplotlib
matplotlib.use('agg') # avoid tkinter import error
import matplotlib.pyplot as plt
import random
from config import DATA_FOLDER, UNIVERSES_FOLDER


# want to see what kind of consistencies there are between different region sets
def calc_stats(f):
    df = pd.read_csv(f, sep="\t", header=0).iloc[:,0:3]
    df.columns = ["chr","start","stop"]
    df['diffs'] = df['stop'] - df['start']
    stats = df.describe()
    mean = stats.iloc[1,2]
    std = stats.iloc[2,2]
    median = stats.iloc[5,2]
    maximum = stats.iloc[7,2]
    print('Number of regions: {}'.format(df.shape[0]))
    print('Region length statistics')
    print('-----------------------')
    print("max: ", maximum)
    print("mean: ", mean)
    print("median: ", median)


def group_regions():
    reg = defaultdict(list)
    def group(f):
        df = pd.read_csv(f, sep="\t", header=None).iloc[:,0:3]
        df.columns = ['chr', 'start', 'stop']
        grouped = df.groupby('chr')
        for name, group in grouped:
            reg[name].extend(list(zip(group['start'], group['stop'])))

    for c, l in reg.items():
        l.sort(key=lambda x: x[0])
        print(c, l)


def bedtools_merge(base, df, merge_dist):
    '''
    merges regions with overlap of merge_dist using the bedtools package
    base: first region set
    df: second region set to merge in with base
    merge_dist: the bedtools -d parameter
    returns: pandas dataframe
    '''

    combined = pd.concat([base, df])
    combined.sort_values(['chrom', 'start'], inplace=True)
    bed = BedTool.from_dataframe(combined)
    merged = bed.merge(d=merge_dist)
    return merged.to_dataframe()


def save_segmentation(name, segmentation, numfiles, overlap, folder='../segmentations/'):
    fp = "{}_{}overlap_{}files_tfbs_universe.txt".format(name, overlap, numfiles)
    path = os.path.join(folder, fp)
    print("saving segmentation to ", path)
    segmentation.to_csv(path, sep='\t', index=False, header=False)


def segmentation_merge(NUM_FILES, data_folder, merge_dist=-10, folder='../segmentations/'):
    '''
    creates a segmentation using bedtools merge and saves it to a file
    NUM_FILES:  the number of files used to create the segmentation
    merge_dist: the -d parameter in bedtools
    folder:     the folder the segmentation is written to
    returns: None
    '''

    # the first file will be the starting point for bedtools merge
    allfiles = sorted(os.listdir(data_folder))
    base_df = pd.read_csv(os.path.join(data_folder, allfiles[0]), sep='\t', header=None).iloc[:,0:3]
    base_df.columns = ['chrom', 'start', 'end'] # have to specify bedtools headers for pd.concat to work

    num_regions = [base_df.shape[0]]
    i = 0
    for f in allfiles[1:NUM_FILES]:
        if i % 50 == 0:
            print(i)
        if i >= NUM_FILES:
            break
        i += 1

        full_path = os.path.join(data_folder, f)
        df = pd.read_csv(full_path, sep='\t', header=None, usecols=[0,1,2])
        df.columns = ['chrom', 'start', 'end']
        base_df = bedtools_merge(base_df, df, merge_dist)
        num_regions.append(base_df.shape[0])

    # save the entire segmentation
    save_segmentation("ALL", base_df, NUM_FILES, merge_dist, folder)
    '''
    # save by each chromosome
    for name, group in base_df.groupby('chrom'):
        save_segmentation(name, group, NUM_FILES, merge_dist, folder)
    '''

    '''
    # visualize the number of regions in a segmentation in relation to the number of bedfiles used
    plt.plot(num_regions)
    plt.xlabel('# bedfiles used to build segmentation')
    plt.ylabel('# regions in the segmentation')
    plt.savefig('region_vs_bedfiles.png')
    '''


def segmentation_tile(tile_size, chrom_sizes, assembly, save_folder):
    '''
    creates a segmentation from tiling the genome and saves it to the folder
    returns: None
    '''
    save_path = os.path.join(save_folder, f"tiles{tile_size}.{assembly}.bed")
    count = 0
    with open(save_path, "w") as fout:
        with open(chrom_sizes, "r") as f:
            for line in f:
                chr_name, chr_size = line.split('\t')
                chr_name = chr_name.strip()
                chr_size = int(chr_size.strip())
                for t in range(0, chr_size, tile_size):
                    fout.write(f"{chr_name}\t{t}\t{t+tile_size}\n")
                    count += 1
    print(f"total regions: {count}")
    # save by each chromosome
    '''
    for name, group in base_df.groupby('chrom'):
        save_segmentation(name, group, NUM_FILES, merge_dist, folder)
    '''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str,
            help='choose tile or merge')
    parser.add_argument('-v', type=str, default="hg19",
            help='choose hg19 or hg38')
    parser.add_argument('-n', type=int,
            help='how many files to include in building the segmentation (default 100)',
            default=100)
    parser.add_argument('-d', type=int,
            help='how much overlap to use when merging regions with bedtools (default -10), negative values enforce the number of b.p. required for overlap.',
            default=-10)
    parser.add_argument('-t', type=int,
            help='tile size (default 300)',
            default=300)
    args = parser.parse_args()
    data_folder = os.path.join(DATA_FOLDER, "data")
    if args.m == "merge":
        segmentation_merge(args.n, data_folder, args.d, UNIVERSES_FOLDER)
    
    elif args.m == "tile":
        if args.v == "hg19":
            chrom_sizes = "./hg19_chrom.sizes"
            segmentation_tile(args.t, chrom_sizes, "hg19", UNIVERSES_FOLDER)
        else:
            raise ValueError(f"not implemented for args.v={args.v}")
    else:
        raise ValueError(f"args.m={args.m} not supported. Must choose from [merge, tile]")
   