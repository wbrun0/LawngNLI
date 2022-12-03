# **LawngNLI**

This is the repo for the Findings of EMNLP 2022 paper [LawngNLI: A Long-Premise Benchmark for In-Domain Generalization from Short to Long Contexts and for Implication-Based Retrieval]().



## **Abstract**

Natural language inference has trended toward studying contexts beyond the sentence level. An important application area is law: past cases often do not foretell how they apply to new situations and implications must be inferred. This paper introduces LawngNLI, constructed from U.S. legal opinions with automatic labels with high human-validated accuracy. Premises are long and multigranular. Experiments show two use cases. First, LawngNLI can benchmark for in-domain generalization from short to long contexts. It has remained unclear if large-scale long-premise NLI datasets actually need to be constructed: near-top performance on long premises could be achievable by fine-tuning using short premises. Without multigranularity, benchmarks cannot distinguish lack of fine-tuning on long premises versus domain shift between short and long datasets. In contrast, our long and short premises share the same examples and domain. Models fine-tuned using several past NLI datasets and/or our short premises fall short of top performance on our long premises. So for at least certain domains (such as ours), large-scale long-premise datasets are needed. Second, LawngNLI can benchmark for implication-based retrieval. Queries are entailed or contradicted by target documents, allowing users to move between arguments and evidence. Leading retrieval models perform reasonably zero shot on a LawngNLI-derived retrieval task. We compare different systems for re-ranking, including lexical overlap and cross-encoders fine-tuned using a modified LawngNLI or past NLI datasets. LawngNLI can train and test systems for implication-based case retrieval and argumentation.



## **(I) Setup**

**All scripts after (I) should be launched from the LawngNLI environment (`conda activate LawngNLI`) and file path 'LawngNLI/src'.**

```
git clone https://github.com/wbruno2/LawngNLI.git
cd LawngNLI
conda env create -n LawngNLI -f environment.yml
conda env create -n LawngNLI2 -f environment2.yml
conda create -n LawngNLI3 python=3.9 -y
cd src
conda activate LawngNLI2
pip install torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda activate LawngNLI3
pip install gdown==4.5.3
conda activate LawngNLI
git clone https://github.com/freelawproject/eyecite
cd eyecite
git reset --hard 7e58997d2f
cd ..
mv eyecite/eyecite eyecite2
rm -rf eyecite
mv eyecite2 eyecite
sed -i 's/\\ {Y/[ ]?{Y/' eyecite/regexes.py
#en-core-web-lg-3.1.0
python -m spacy download en_core_web_lg
python -m pip install fl-flint==0.2
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## **(II) Reconstruction of LawngNLI Datasets**

(1) Obtain an API key to download the bulk case opinion corpora (about 24 GB): create a research account for the [Caselaw Access Project](https://case.law/), obtain an API key, and add the API key after 'Token ' on line 5 of dataset.sh.

(2) `sh dataset.sh`

(3) Download '[LawngNLI_filter.xz](https://drive.google.com/file/d/1fPuOkMCSpJeoP4NYcVcbFIywnSNlAKKx/view?usp=share_link)' and '[LawngNLI_minus_case_text.xz](https://drive.google.com/file/d/1FltJduz-zD1n7eR3lQHufL_SMCpzEDAM/view?usp=share_link)' to the main directory.

(4) Run a dataset-specific script (A)-(C):



**(A) LawngNLI: main (NLI) version**

'LawngNLI.xz', from `CUDA_VISIBLE_DEVICES=-1 python reconstruct_nli.py`.

This dataset is used for the paper's 3-label NLI experiments.



**(B) LawngNLI: retrieval version**

'train.jsonl', 'dev.jsonl', and test sets ('data_long.xz' and 'data_long_BM25.xz') from `python reconstruct_retrieval.py` after reconstructing 'LawngNLI.xz'.

This dataset is used for the paper's retrieval experiments. The train and val sets are identical to the NLI version with BM25-filtered long premises only except that: (a) citations are not removed from the premises and (b) the Neutral premises are constructed by dense retrieval using the hypothesis rather than more complex criteria for dataset difficulty from the NLI version. The test set contains 999 negative pairs per positive pair. See the paper for more details.



**(C) unfiltered-LawngNLI2**

'unfiltered-LawngNLI2.csv', from `CUDA_VISIBLE_DEVICES=-1 python _dataset1.py --cpu #[maximum number of CPUs]#`.

This dataset is not used in the paper's experiments. It contains about 10.3 million untwinned candidate NLI examples. It is not balanced on labels and is left for future slicing.

To load:

```
import pandas as pd
data=pd.read_csv('unfiltered-LawngNLI2.csv', header=None, names=['caseid','stance','cited_case','cited_pages','citing_parenthetical','case_history_flag','contradicted_parenthetical'])
```


## **(III) Results Replication**

(1) From **(IV) Model Fine-Tuning** section, fine-tune all NLI models (for NLI Tables) or all retrieval models (for retrieval Tables). Models may need to be moved manually to the file paths in the scripts in (4).

(2) If not completed already, from **(II) Reconstruction of LawngNLI Datasets** section, reconstruct the NLI version of LawngNLI (for NLI Tables) or the retrieval version of LawngNLI (for retrieval Tables).

(3) `sh results.sh`

(4) Run a script for the given Table.

```
# NLI

#Table 2
python _stats.py

#Tables 5, 13, 14, 15
cd Nystromformer
python ../long_short.py
cd ..

#Table 12
cd Nystromformer
python ../baselines.py
cd ..

# Retrieval

#Table 17
cd Nystromformer
python ../zero_shot_retrieval.py
cd ..

#Table 6
cd Nystromformer
python ../retrieval.py
cd ..
```



## **(IV) Model Fine-Tuning**

(1) If not completed already, from **(II) Reconstruction of LawngNLI Datasets** section, reconstruct the NLI version of LawngNLI. For retrieval models, also reconstruct the retrieval version of LawngNLI.

(2) 

```
python _stats.py
sh setup.sh
```

(3) Run the command lines for the specific models in models.sh.



## **Citation**

If you find this repo useful, please cite the following paper:

```
@inproceedings{BrunoRo22,
    author = {William Bruno and Dan Roth},
    title = {{LawngNLI: A Long-Premise Benchmark for In-Domain Generalization from Short to Long Contexts and for Implication-Based Retrieval}},
    booktitle = {Findings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2022},
    url = "https://cogcomp.seas.upenn.edu/papers/BrunoRo22.pdf",
    funding = {BETTER, Google},
}
```



## **License**

Distributed under the MIT License.
