import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec


class BaseEmbeddings:
    def __init__(self, embeddings, vocab):
        self.embeddings = embeddings
        self.vocab = vocab


def plot_embeddings(path2embeddings, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(path2embeddings, "rb") as f:
        points = pickle.load(f)
    name = os.path.basename(path2embeddings).split(".")[0]
    fig_name1 = os.path.join(save_dir, "plot_{}.png".format(name))
    fig_name2 = os.path.join(save_dir, "plot_chromosomes_{}.png".format(name))

    fig, ax = plt.subplots(figsize=(9, 9))
    sns_plot = sns.scatterplot(data=points, x="x1", y="x2", ax=ax, alpha=0.2)
    plt.savefig(fig_name1)
    plt.close(fig)

    # fig, ax = plt.subplots(figsize=(12, 9))
    # sns_plot = sns.scatterplot(data=points,x='x1',y='x2', ax=ax,hue='label', alpha=0.2)
    # plt.savefig(fig_name2)
    # plt.close(fig)


def generate_visualization(path2model, model_type, save_dir, dim, method, kwargs={}):
    os.makedirs(save_dir, exist_ok=True)
    embed_name = "points_{}d_{}.pickle".format(dim, method)
    path2embeddings = os.path.join(save_dir, embed_name)
    if os.path.exists(path2embeddings):
        print("{} exists. Use the existing one".format(path2embeddings))
        return path2embeddings
    if model_type == "region2vec":
        model = Word2Vec.load(path2model)
        print("Vocabulary size is {}".format(len(model.wv.index_to_key)))
        vectors = model.wv.vectors
    elif model_type == "base":
        with open(path2model, "rb") as f:
            base_embed_obj = pickle.load(f)
            print("Vocabulary size is {}".format(len(base_embed_obj.vocab)))
            vectors = base_embed_obj.embeddings
    if method == "tsne":
        from sklearn.manifold import TSNE

        print("Using TSNE for dimension reduction")
        tsne_emebd = TSNE(n_components=dim, **kwargs)
        points = tsne_emebd.fit_transform(vectors)
    elif method == "umap":
        import umap.umap_ as umap
        import umap.plot

        umap_embed = umap.UMAP(n_components=dim, metric="cosine", **kwargs)
        mapper = umap_embed.fit(vectors)

        # with open(path2embeddings,'wb') as f:
        #     pickle.dump(mapper,f)
        name = os.path.basename(path2embeddings).split(".")[0]
        fig_name1 = os.path.join(save_dir, "plot_{}.png".format(name))

        fig, ax = plt.subplots(figsize=(9, 9))
        ax = umap.plot.points(mapper, ax=ax)
        plt.savefig(fig_name1, bbox_inches="tight")
        plt.close(fig)
        # points = umap_embed.fit_transform(model.wv.vectors)

    # points = pd.DataFrame(data=points, columns=['x%d'%(i+1) for i in range(dim)])
    # vocab = model.wv.index_to_key
    # label = []
    # for word in vocab:
    #     l = word.split(':')[0].strip()
    #     label.append(l)
    # points['label'] = np.array(label)

    # def region2tuple(x):
    #     eles = x.split(':')
    #     chr_name = eles[0].strip()
    #     start, end = eles[1].split('-')
    #     start, end = int(start.strip()), int(end.strip())
    #     return chr_name, start, end
    # regions = [region2tuple(v) for v in vocab]
    # with open(os.path.join(save_dir,'vocab.bed'),'w') as f:
    #     for chr_name, start, end in regions:
    #         f.write('{}\t{}\t{}\n'.format(chr_name, start, end))

    # with open(path2embeddings,'wb') as f:
    #     pickle.dump(points,f)
    # return path2embeddings


def plot_losses_vs_epoch(path2losses, save_dir):
    with open(path2losses, "rb") as f:
        losses = pickle.load(f)
    losses = np.array(losses)
    epochs = len(losses)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set(yscale="log")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Shuffling number")
    ax.plot(np.arange(1, epochs + 1), losses, "bo-", linewidth=2, markersize=8)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gene Embedding Visualization")
    parser.add_argument("--path2model", type=str, help="path to a saved model")
    # parser.add_argument('--path2losses', type=str, help='path to a saved loss')
    parser.add_argument("--save_dir", type=str, help="save visualization results to this folder")
    args = parser.parse_args()

    path2embeddings = generate_visualization(args.path2model, args.save_dir, 2, "tsne")
    plot_embeddings(path2embeddings, args.save_dir)
    # plot_losses_vs_epoch(args.path2losses)
