from geniml.tokenization import hard_tokenization
from geniml.region2vec import region2vec
import argparse
import os
from config import  DATA_FOLDER, MODELS_FOLDER, UNIVERSES_FOLDER

def main(args):
    src_folder = os.path.join(DATA_FOLDER, 'data')
    dst_folder = os.path.join(MODELS_FOLDER, f'tokens/token_universe_{args.universe}')
    universe_types ={
       'Large': f'{UNIVERSES_FOLDER}/ALL_100overlap_690files_tfbs_universe.txt',
       'Medium':f'{UNIVERSES_FOLDER}/ALL_1000overlap_690files_tfbs_universe.txt',
       'Small': f'{UNIVERSES_FOLDER}/ALL_10000overlap_690files_tfbs_universe.txt',
       'Tiny': f'{UNIVERSES_FOLDER}/ALL_50000overlap_690files_tfbs_universe.txt',
       'tile1k': f'{UNIVERSES_FOLDER}/tiles1000.hg19.bed',
       'tile5k': f'{UNIVERSES_FOLDER}/tiles5000.hg19.bed',
       'tile25k': f'{UNIVERSES_FOLDER}/tiles25000.hg19.bed',
       'dhs': f'{UNIVERSES_FOLDER}/dhs_universe.bed',
    }
    
    universe_file = universe_types[args.universe]
    # must run tokenization first
    status = hard_tokenization(src_folder, dst_folder, universe_file, 1e-9, bedtools_path="bedtools")

    if status:
        save_dir = os.path.join(args.save_folder, 'expr_universe_{}/{}W{}D-{:.4f}r'.format(args.universe,args.win_size,args.embed_dim,args.init_lr))
        region2vec(dst_folder, save_dir, num_shufflings=args.num_shufflings, init_lr=args.init_lr, embedding_dim=args.embed_dim, context_win_size=args.win_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'region2vec training script')
    parser.add_argument('--save_dir', help='parent folder to generated shuffled datasets')
    parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('--init_lr', type=float, default=0.1, help='initial learing rate')
    parser.add_argument('--num_shufflings', type=int, default=1000, help='number of shufflings')
    parser.add_argument('--win_size', type=int, default=5, help='context window size')
    parser.add_argument('--universe', type=str, default='Large', help='universe type')
    parser.add_argument('--save_folder', type=str, default='/tfbs_region2vec_models/', help='save folder')

    
    args = parser.parse_args()

    main(args)