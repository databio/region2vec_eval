import argparse
import glob
import os
import pickle

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

from geniml.eval.ctt import *
from geniml.eval.gdst import *
from geniml.eval.npt import *
from geniml.eval.rct import *
from geniml.eval.utils import *

from config import MODELS_FOLDER, EVAL_RESULTS_FOLDER


def remap_name(name):
    return name.split('/')[-3]

def main(args):
    result_path = EVAL_RESULTS_FOLDER
    model_root_path = MODELS_FOLDER

    model_paths = glob.glob(
        os.path.join(
            model_root_path,
            "expr_universe_{}/*r/models/region2vec_latest.pt".format(args.universe),
        )
    )
    batch = [(m, "region2vec") for m in model_paths]
    row_labels = [remap_name(m) for m in model_paths]
    base_paths = [
        os.path.join(model_root_path, f"expr_universe_{args.universe}/bin_embed.pickle"),
        os.path.join(model_root_path, f"expr_universe_{args.universe}/pca_embed_10D.pickle"),
        os.path.join(model_root_path, f"expr_universe_{args.universe}/pca_embed_100D.pickle"),
    ]
    row_labels += ["Binary", "PCA-10D", "PCA-100D"]
    batch = batch + [(p, "base") for p in base_paths]
  
    if args.type == "GDSS":
        # Genome distance scaling test
        save_folder = os.path.join(result_path, "gdss_results/{}/".format(args.universe))
        gds_res = gdst_eval(
            batch,
            num_runs=20,
            num_samples=100000,
            save_folder=save_folder,
        )


    if args.type == "NPS":
        # Neighborhood preserving test
        save_folder = os.path.join(result_path, "nps_results/{}/".format(args.universe))
        npr_results = npt_eval(
            batch,
            args.K,
            num_samples=10000,
            num_workers=10,
            num_runs=20,
            resolution=50,
            dist="cosine",
            save_folder=save_folder,
        )

    if args.type == "CTS":
        save_folder = os.path.join(result_path, "cts_results/{}/".format(args.universe))
        ctt_res = ctt_eval(
            batch,
            num_runs=20,
            num_data=10000,
            save_folder=save_folder,
            num_workers=10,
        )
    if args.type == "RCS":
        if args.out_dim == -1:
            save_folder = os.path.join(result_path, f"rcs_results/{args.universe}/")
        else:
            save_folder = os.path.join(result_path, f"rcs_{args.out_dim}d_results/{args.universe}/")
        batch = [(m, "region2vec", base_paths[0]) for m in model_paths]
        batch = batch + [(p, "base", base_paths[0]) for p in base_paths]
        rct_eval(
            batch,
            num_runs=5,
            cv_num=5,
            out_dim=args.out_dim,
            save_folder=save_folder,
            num_workers=10,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe",
        default="Large",
        help="select universe type",
    )
    parser.add_argument(
        "--type",
        choices=["GDSS", "NPS", "CTS", "RCS"],
        default="RCT",
        help="select evaluation type",
    )
    parser.add_argument("--K", type=int, default=1000, help="number of neighbors")
    parser.add_argument("--out_dim", type=int, default=-1, help="number of output dimensions")
    parser.add_argument("--batch", type=int, default=-1, help="index of batch")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
        Large <-> Merge (100)
        Medium <-> Merge (1k)
        Small <-> Merge (10k)
    """
    args = parse_args()
    universes = [
        "tile1k",
        "tile5k",
        "tile25k",
        "dhs",
        "Large",
        "Medium",
        "Small",
    ]
    batches = [
        ("tile1k", "tile25k"),
        ("tile5k", "Small"),
        ("Large", "Medium","dhs"),
    ]

    if args.batch == -1:
        for u in universes:
            args.universe = u
            main(args)
    else:
        for u in batches[args.batch]:
            args.universe = u
            main(args)

