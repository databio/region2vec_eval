import os
import glob
import pickle
from config import MODELS_FOLDER
from geniml.eval.utils import *

def prepare_base_embeddings(model_root_path, universe):
    # binary embeddings
    bin_embed_path = os.path.join(model_root_path, f"expr_universe_{universe}/bin_embed.pickle")
    if not os.path.exists(bin_embed_path):
        vocab = get_vocab(
            os.path.join(model_root_path, f"expr_universe_{universe}/5W10D-0.0250r/models/region2vec_latest.pt"),
            "region2vec",
        )
        universe_file = os.path.join(model_root_path, f"expr_universe_{universe}/vocab.bed")

        write_vocab(vocab, universe_file)
        tokenized_files = glob.glob(os.path.join(model_root_path, f"tokens/token_universe_{universe}/*"))
        bin_embed = get_bin_embeddings(universe_file, tokenized_files)
        
        with open(bin_embed_path, "wb") as f:
            pickle.dump(bin_embed, f)
    else:
        with open(bin_embed_path,"rb") as f:
            bin_embed = pickle.load(f)
    pca10d_path = os.path.join(model_root_path, f"expr_universe_{universe}/pca_embed_10D.pickle")
    if not os.path.exists(pca10d_path):
        pca10d = get_pca_embeddings(bin_embed, 10)
        with open(pca10d_path,"wb") as f:
            pickle.dump(pca10d, f)
    
    pca100d_path = os.path.join(model_root_path, f"expr_universe_{universe}/pca_embed_100D.pickle")
    if not os.path.exists(pca100d_path):
        pca100d = get_pca_embeddings(bin_embed, 100)
        with open(pca100d_path,"wb") as f:
            pickle.dump(pca100d, f)


if __name__ == '__main__':
    """
    Large <-> Merge (100)
    Medium <-> Merge (1k)
    Small <-> Merge (10k)
    """
    
    universes = ['tile1k', 'tile5k', 'tile25k', 'dhs', 'Large', 'Medium', 'Small']
    for universe in universes:
        prepare_base_embeddings(MODELS_FOLDER, universe)