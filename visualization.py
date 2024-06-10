from visualization_utils import *
import os
import glob
import multiprocessing as mp
from time import time
import argparse
from config import MODELS_FOLDER

def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1)/60:.4f}min")
        return result

    return wrap_func

save_dir = './embed_visualization'
universes = ['tile1k', 'tile5k', 'tile25k', 'dhs', 'Large', 'Medium', 'Small']

@timer_func
def get_visualization(model, model_type, universe, save_dir, kwargs={}):
    if model_type == "region2vec":
        name = model.split('/')[-3]
        save_folder = os.path.join(save_dir, universe, name)
    elif model_type == "base":
        name = model.split('/')[-1].split('.')[0]
        save_folder = os.path.join(save_dir, universe,name)
    else:
        raise ValueError("invalid model_type")
    path2embed = generate_visualization(model, model_type, save_folder, 2, "umap", kwargs)

def main(universe_id):
    universe = universes[universe_id]
    result_path = MODELS_FOLDER
    model_paths = glob.glob(
        os.path.join(
            result_path,
            "expr_universe_{}/*r/models/region2vec_latest.pt".format(universe),
        )
    )
    batch = [(m, "region2vec") for m in model_paths]
    base_paths = [
            os.path.join(
                result_path, "expr_universe_{}/bin_embed.pickle".format(universe)
            ),
            os.path.join(
                result_path, "expr_universe_{}/pca_embed_10D.pickle".format(universe)
            ),
            os.path.join(
                result_path, "expr_universe_{}/pca_embed_100D.pickle".format(universe)
            ),
        ]
    batch = batch + [(p, "base") for p in base_paths]
    for model, model_type in batch:
        get_visualization(model, model_type, universe, save_dir)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description= 'region2vec visualization script')
    # parser.add_argument('--universe_id', type=int, default=0, help='select universe')
    # args = parser.parse_args()
    for i in range(len(universes)):
        main(i)