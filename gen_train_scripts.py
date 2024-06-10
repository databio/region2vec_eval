
import os
import glob
from config import TRAIN_SCRIPTS_FOLDER, MODELS_FOLDER
def combine_all(list_of_lists):
    if len(list_of_lists) == 1:
        return [[n] for n in list_of_lists[0]]
    first_list = list_of_lists[0]
    sub_list_of_lists = list_of_lists[1:]
    new_list = combine_all(sub_list_of_lists)
    merged_list = []
    for item in first_list:
        for nl in new_list:
            merged_list.append([item]+nl)
    return merged_list

"""
    Large <-> Merge (100)
    Medium <-> Merge (1k)
    Small <-> Merge (10k)
"""
universes = ['tile1k', 'tile5k', 'tile25k', 'dhs', 'Large', 'Medium', 'Small']
init_lrs = [0.025, 0.1, 0.5]
embed_dims = [10, 100]
win_sizes = [5, 50]
train_scripts_folder = TRAIN_SCRIPTS_FOLDER
os.makedirs(train_scripts_folder, exist_ok=True)

train_file_names = [os.path.join(train_scripts_folder,"{}_train.sh".format(u)) for u in universes]

file_ptrs = {}
for i,name in enumerate(train_file_names):
    file_ptrs[universes[i]] = open(name, 'w')
params = combine_all([universes, init_lrs, embed_dims, win_sizes])
save_folder = MODELS_FOLDER
program_path = glob.glob("train_region2vec.py")[0]
for param in params:
    universe, init_lr, embed_dim, win_size = param
    script = f'python {program_path} --embed_dim {embed_dim} --init_lr {init_lr} --win_size {win_size} --universe {universe} --num_shufflings 100 --save_folder {save_folder}'
    file_ptrs[universe].write(script)
    file_ptrs[universe].write('\n')
for u in file_ptrs:
    file_ptrs[u].close()