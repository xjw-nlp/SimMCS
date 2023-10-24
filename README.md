# SimMCS
This repository contains the code for our paper [Alleviating Exposure Bias via Multi-level Contrastive Learning and Deviation Simulation in Abstractive Summarization](https://aclanthology.org/2023.findings-acl.617/).
## Overview
In this paper, we present a simple yet effective framework to alleviate _exposure bias_ from two different perspectives in abstractive summarization.
![overview](./overview.png)
## Installation
- `conda create --name env --file spec-file.txt`
- `conda activate env`
- `pip install -r requirements.txt`
- `git clone https://github.com/neulab/compare-mt.git`
- `cd ./compare-mt`
- `pip install -r requirements.txt`
- `python setup.py install`

## Training
For CNNDM dataset, we run the code below:
```console
python ./src/main.py --cuda --gpuid 1 --config cnndm -l
```
## Citation
```console
@inproceedings{xie-etal-2023-alleviating,
    title = "Alleviating Exposure Bias via Multi-level Contrastive Learning and Deviation Simulation in Abstractive Summarization",
    author = "Xie, Jiawen  and
      Su, Qi  and
      Zhang, Shaoting  and
      Zhang, Xiaofan",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.617",
    doi = "10.18653/v1/2023.findings-acl.617",
    pages = "9732--9747",
    abstract = "Most Transformer based abstractive summarization systems have a severe mismatch between training and inference, i.e., exposure bias. From diverse perspectives, we introduce a simple multi-level contrastive learning framework for abstractive summarization (SimMCS) and a tailored sparse decoder self-attention pattern (SDSA) to bridge the gap between training and inference to improve model performance. Compared with previous contrastive objectives focusing only on the relative order of probability mass assigned to non-gold summaries, SimMCS additionally takes their absolute positions into account, which guarantees that the relatively high-quality (positive) summaries among them could be properly assigned high probability mass, and further enhances the capability of discriminating summary quality beyond exploiting potential artifacts of specific metrics. SDSA simulates the possible inference scenarios of deviation in the training phase to get closer to the ideal paradigm. Our approaches outperform the previous state-of-the-art results on two summarization datasets while just adding fairly low overhead. Further empirical analysis shows our model preserves the advantages of prior contrastive methods and possesses strong few-shot learning ability.",
}
```
