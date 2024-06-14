# Methods for evaluating unsupervised vector representations of genomic regions
This repository contains code and instructions to reproduce the results presented in the paper. The proposed evaluation metrics are implemented in [geniml.eval](https://github.com/databio/geniml/tree/master/geniml/eval).

## Requirements
- [geniml](https://github.com/databio/geniml)
- beautifulsoup4
- python=3.9
- pybedtools
- [bedtools](https://bedtools.readthedocs.io/en/latest/content/installation.html)

```
git clone git@github.com:databio/geniml.git
cd geniml
pip install -e .
```
After installing `geniml`, add `bedtools` binary to the environment variable `PATH`.

## Preparation
### Customize the configurations
Change the constants defined in `config.py`. Below are the descriptions for the constants.
```yaml
DATA_URL: link to the dataset
DATA_FOLDER: folder that stores the downloaded the dataset
TRAIN_SCRIPTS_FOLDER: folder that stores all the generated training scripts
MODELS_FOLDER: folder that stores all the trained models
UNIVERSES_FOLDER: folder that stores all the universes
EVAL_RESULTS_FOLDER: folder that stores all the evaluation results
```
### Download the dataset
Run the following command:
```bash
python -m src.download_dataset
```
Or download all the [content](http://big.databio.org/region2vec_eval/tfbs_dataset/) to `DATA_FOLDER`.
### Prepare universes
We provided all the seven universes used in the paper at [hg19 universes](http://big.databio.org/region2vec_eval/universes/). Download the universes to `UNIVERSES_FOLDER` specified in `config.py`.

We used the following code to generate the universes except the DHS universe, which is an external universe. You can use the same code to generate the universes based on your data, only to change `DATA_FOLDER` in `config.py` and the total number of files passed to `-n`.
```bash
# The Merge (100) universe
python -m src.gen_universe -m merge -n 690 -d 100
# The Merge (1k) universe
python -m src.gen_universe -m merge -n 690 -d 1000
# The Merge (10k) universe
python -m src.gen_universe -m merge -n 690 -d 10000
# The Tiling (1k) universe
python -m src.gen_universe -m tile -v hg19 -n 690 -t 1000
# The Tiling (5k) universe
python -m src.gen_universe -m tile -v hg19 -n 690 -t 5000
# The Tiling (25k) universe
python -m src.gen_universe -m tile -v hg19 -n 690 -t 25000
```
### Train embedding models
You can download all the trained models to `MODELS_FOLDER` (in `config.py`) at [models](http://big.databio.org/region2vec_eval/tfbs_models/). Note that `Large`, `Medium` and `Small` correspond to `Merge (100)`, `Merge (1k)` and `Merge (10k)`, respectively, in the paper.

We used the following steps to get all the models.

1. Generate training scripts via 
    ```bash
    python -m src.gen_train_scripts
    ```
2. Then, go to the `TRAIN_SCRIPTS_FOLDER` (specified in `config.py`) folder, and run all the scripts there to get trained models.

    Note that in `gen_train_scripts.py`, we include seven universes, three initial learning rates, two embedding dimensions, and two context window sizes.
    Therefore, for each universe, we will train 12 Region2Vec models, and in total, we will have 84 Region2Vec models.

3. After training Region2Vec models, run the following code to generate base embeddings, namely Binary, PCA-10D, and PCA-100D, for each of the seven universes.
    ```bash
    python -m src.get_base_embeddings
    ```

To obtain the results in Table S2, run the following code
```bash
python -m src.assess_universe
```
Note that we do not assess the original universes. Since Region2Vec will filter out some low-frequency regions in a universe based on the training data, we focused on the acutal universes with regions that have embeddings.

## Evaluate region embeddings
Run the following scripts to obtain the evaluation results.
```bash
python -m src.eval_script --type GDSS
python -m src.eval_script --type NPS
python -m src.eval_script --type CTS
python -m src.eval_script --type RCS
```

To speed up the process, you can split the universes into batches (Line 209, `eval_script.py`)
```python
batches = [
        ("tile1k", "tile25k"),
        ("tile5k", "Small"),
        ("Large", "Medium","dhs"),
    ]
```
Then, run the evaluation on each batch in parallel. For example, 
```bash
python -m src.eval_script --type GDSS --batch 0
```
will evaluate models for the Tiling (1k) and Tiling (25k) universes.

## Downstream tasks
We designed cell type and antibody type classification tasks for the trained region embeddings. We randomly selected 60% of all the BED files as the training files and the remaining as test files. We divided the BED files five times with different random seeds. The file splits are stored in `classification_data`. The code that generates the splits can be found in `classification.ipynb`.

Run the classification using the following script:
```bash
python -m src.classification
```

## Analyze results
We used the Jupyter notebook `result_analysis.ipynb` to generate all the figures and calculate the results.

## Generate embedding visualizations
The visualizations of different sets of region embeddings can be found at [embed_visualization](http://big.databio.org/region2vec_eval/embed_visualization/).

We used the following command to generate UMAP visualizations of all sets of region embeddings.
```bash
python -m src.visualization
```