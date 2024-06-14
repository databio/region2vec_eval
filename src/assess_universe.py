import os

from geniml.assess.assess import get_f_10_score
from config import MODELS_FOLDER, DATA_FOLDER

def count_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        lines = f.readlines()
    return len(lines)
    
def main():
    universes = ['tile1k', 'tile5k', 'tile25k', 'dhs', 'Large', 'Medium', 'Small']
    for universe in universes:
        vocab_file = os.path.join(MODELS_FOLDER, f"expr_universe_{universe}/vocab.bed")
        F_10 = get_f_10_score(os.path.join(DATA_FOLDER, "data"),
                                os.path.join(DATA_FOLDER, "file_list.txt"),
                                vocab_file,
                                no_workers=8)
        num_vocab = count_vocab(vocab_file)
        print(f"{universe}: size={num_vocab}, F_10={F_10:.4f}")


if __name__ == '__main__':
    main()
