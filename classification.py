import numpy as np
import os
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from collections import Counter
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
from config import *

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

# get metadata
meta_path_path = "meta_data.txt"
metadata = {}
with open(meta_path_path, "r") as f:
    for line in f:
        info_dict = dict()
        str_line = line
        data = str_line.split(';')
        filename = data[0].split('\t')[0]
        item = data[0].split('\t')[1].split('=')
        info_dict[item[0]] = item[1]
        for entry in data[1:]:
            items = entry.split('=')
            info_dict[items[0].strip()] = items[1].strip()
        metadata[filename] = info_dict
        
        
# select files for classification
def sel_files(metadata, label_type):
    file_list = list(metadata.keys())
    labels = []
    for f in file_list:
        if metadata[f][label_type] == '':
            raise ValueError(f"No {label_type} type")
        labels.append(metadata[f][label_type])
    counter = Counter(labels)
    sel_classes = []
    for cls, freq in counter.most_common():
        if freq > 10:
            sel_classes.append(cls)

    sel_file_labels = {c:[] for c in sel_classes}
    total_files = 0
    for f in file_list:
        label = metadata[f][label_type]
        if label in sel_file_labels:
            sel_file_labels[label].append(f)
            total_files += 1
    # print(f"{label_type} classification, total files {total_files}, {len(sel_classes)} classes")
    return sel_file_labels
cell_data = sel_files(metadata, "cell")
antibody_data = sel_files(metadata, "antibody")


# split the data
def split_data(data_array, train_ratio=0.6):
    train_data = []
    test_data = []
    
    for key in data_array:
        num = len(data_array[key])
        train_num = int(num*train_ratio)
        assert train_num > 0 and train_num < num, "invalid training number"
        indexes = np.arange(num)
        np.random.shuffle(indexes)
        train_data.extend([(data_array[key][i],key) for i in indexes[0:train_num]])
        test_data.extend([(data_array[key][i],key) for i in indexes[train_num:]])
    return train_data, test_data

def gen_train_test_files(cell_data, antibody_data, train_ratio=0.6, num_runs=5):
    # save the training and test files
    os.makedirs("classification_data", exist_ok=True)
    for it in range(num_runs):
        train_cell_data, test_cell_data = split_data(cell_data, train_ratio)
        with open(os.path.join("classification_data", f"cell_train_train{train_ratio:.2f}_{it}.txt"), "w") as f:
            for fname, l in train_cell_data:
                f.write(f"{fname}\t{l}\n")
        with open(os.path.join("classification_data", f"cell_test_train{train_ratio:.2f}_{it}.txt"), "w") as f:
            for fname, l in test_cell_data:
                f.write(f"{fname}\t{l}\n")

    for it in range(num_runs):
        train_antibody_data, test_antibody_data = split_data(antibody_data, train_ratio)
        with open(os.path.join("classification_data", f"antibody_train_train{train_ratio:.2f}_{it}.txt"), "w") as f:
            for fname, l in train_antibody_data:
                f.write(f"{fname}\t{l}\n")
        with open(os.path.join("classification_data", f"antibody_test_train{train_ratio:.2f}_{it}.txt"), "w") as f:
            for fname, l in test_antibody_data:
                f.write(f"{fname}\t{l}\n")

# load training and test data from files
def load_data(data_path):
    fnames = []
    labels = []
    with open(data_path, "r") as f:
        for line in f:
            fname, label = line.strip().split('\t')
            fnames.append(fname)
            labels.append(label)
    return fnames, labels


# load embeddings
def get_file_embedding(file_path, model, model_type="region2vec"):
    sentence = []
    embedding = 0.0
    count = 0
    if model_type == "region2vec":
        vocab2index = model.wv.key_to_index
        embeddings = model.wv.vectors
    elif model_type == "base":
        vocab2index = {v:i for i,v in enumerate(model.vocab)}
        embeddings = model.embeddings
    with open(file_path, 'r') as f:
        for line in f:
            elements = line.strip().split('\t')[0:3]
            chr_name = elements[0].strip()
            start = elements[1].strip()
            end = elements[2].strip()
            word = chr_name+':'+start+'-'+end
            if word in vocab2index:
                embedding += embeddings[vocab2index[word]]
                count += 1
    if count > 0:
        embedding /= count
    return embedding
def get_embeddings(data, model_path, universe, model_type):
    if model_type == "region2vec":
        model = Word2Vec.load(model_path)
    elif model_type == "base":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    token_folder = os.path.join(f"{MODELS_FOLDER}/tokens", f"token_universe_{universe}")
    embeddings = []
    for fname in data:
        fname = fname[0:-3]
        file_path = os.path.join(token_folder, fname)
        embed = get_file_embedding(file_path, model, model_type)
        embeddings.append(embed)
    return embeddings
def get_label2index(labels):
    label2index = {}
    count = 0
    for l in labels:
        if l not in label2index:
            label2index[l] = count
            count += 1
    return label2index


def get_classification_score(X_train, y_train, X_test, y_test, classifier="svm"):
    if classifier == "svm":
        clf = svm.SVC(kernel = 'linear')
        clf.fit(X_train, y_train)
        f1 = f1_score((y_test), clf.predict(X_test), average = 'micro')
    return f1


def get_avg_classificatoin_score(model_path, universe, classifier="svm", model_type="region2vec", num_runs=5):
    all_scores = []
    for cls_type in ["cell", "antibody"]:
        cls_scores = []
        for n in range(num_runs):
            train_files = f"classification_data/{cls_type}_train_train0.60_{n}.txt"
            test_files = f"classification_data/{cls_type}_test_train0.60_{n}.txt"
            train_data, train_label = load_data(train_files)
            test_data, test_label = load_data(test_files)
            if n == 0: # get all embeddings and labels once
                file2idx = {f:i for i, f in enumerate(train_data+test_data)}
                label2number = get_label2index(train_label)
                train_embeddings = get_embeddings(train_data, model_path, universe, model_type)
                test_embeddings = get_embeddings(test_data, model_path, universe, model_type)
                embeddings = np.concatenate([train_embeddings, test_embeddings])
            train_embeddings = np.stack([embeddings[file2idx[f]] for f in train_data])
            test_embeddings = np.stack([embeddings[file2idx[f]] for f in test_data])
            train_label_number = [label2number[l] for l in train_label]
            test_label_number = [label2number[l] for l in test_label]
            f1 = get_classification_score(train_embeddings, train_label_number, test_embeddings, test_label_number, classifier)
            cls_scores.append(f1)
        cls_scores = np.array(cls_scores)
        avg_score = cls_scores.mean()
        std_score = cls_scores.std()
        all_scores.append((avg_score,std_score))
    return all_scores


def get_region2vec_scores(classifier_type="svm"):
    universes = [
        "tile1k",
        "tile5k",
        "tile25k",
        "dhs",
        "Large",
        "Medium",
        "Small",
    ]
    init_lrs = [0.025, 0.1, 0.5]
    embed_dims = [10, 100]
    win_sizes = [5, 50]
    
    params = combine_all([universes, init_lrs, embed_dims, win_sizes])
    with open(f"classificaton_results_{classifier_type}_dhs_region2vec.txt", "w") as fout:
        fout.write("universe,model,cell_acc,cell_std,antibody_acc,antibody_std\n")
        for param in tqdm(params):
            universe, init_lr, embed_dim, win_size = param
            model_config = f"{win_size}W{embed_dim}D-{init_lr:.4f}r"
            model_path = f"{MODELS_FOLDER}/expr_universe_{universe}/{model_config}/models/region2vec_latest.pt"
            cell_scores, antibody_scores = get_avg_classificatoin_score(model_path, universe, classifier_type, "region2vec")
            fout.write(f"{universe},{model_config},{cell_scores[0]:.6f},{cell_scores[1]:.6f},{antibody_scores[0]:.6f},{antibody_scores[1]:.6f}\n")
            fout.flush()

def get_base_scores(classifier_type="svm"):
    universes = [
        "tile1k",
        "tile5k",
        "tile25k",
        "dhs",
        "Large",
        "Medium",
        "Small",
    ]
    base_embeds = ["bin_embed.pickle", "pca_embed_10D.pickle", "pca_embed_100D.pickle"]
    params = combine_all([universes, base_embeds])
    
    with open(f"classificaton_results_{classifier_type}_dhs_base.txt", "w") as fout:
        fout.write("universe,model,cell_acc,cell_std,antibody_acc,antibody_std\n")
        for param in tqdm(params):
            universe, base_embed = param
            model_path = f"{MODELS_FOLDER}/expr_universe_{universe}/{base_embed}"
            cell_scores, antibody_scores = get_avg_classificatoin_score(model_path, universe, classifier_type, "base")
            fout.write(f"{universe},{base_embed},{cell_scores[0]:.6f},{cell_scores[1]:.6f},{antibody_scores[0]:.6f},{antibody_scores[1]:.6f}\n")
            fout.flush()
if __name__ == '__main__':
    get_region2vec_scores("svm")
    get_base_scores("svm")

