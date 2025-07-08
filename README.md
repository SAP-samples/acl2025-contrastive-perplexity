# Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2401.08491-29d634.svg)](https://arxiv.org/abs/2401.08491)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/acl2025-contrastive-perplexity)](https://api.reuse.software/info/github.com/SAP-samples/acl2025-contrastive-perplexity)


#### News
- **05/30/2025:** :confetti_ball: Initial repo created :tada:


## Description
This repository **will contain** the source code for our paper [**Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models**](https://arxiv.org/abs/2401.08491) to be presented at [ACL2025](https://2025.aclweb.org/).

### Abstract
The generation of toxic content by large language models (LLMs) remains a critical challenge for the safe deployment of language technology. We propose a novel framework for implicit knowledge editing and controlled text generation by fine-tuning LLMs with a prototype-based contrastive perplexity objective. Central to our method is the construction of hard negatives—toxic outputs that are generated through adversarial paraphrasing to be semantically similar and closely matched in length and model probability to their non-toxic counterparts. By training on these challenging and realistic pairs, our approach ensures robust and stable contrastive optimization. Experimental results in the domain of detoxification demonstrate that our method significantly reduces toxic generation while maintaining strong performance on downstream tasks such as commonsense reasoning and reading comprehension. Our findings highlight the effectiveness of leveraging hard negatives for attribute-aware language model fine-tuning.



#### Authors:
 - [Tassilo Klein](https://tjklein.github.io/)
 - [Moin Nabi](https://moinnabi.github.io/)

## Requirements
- [Python](https://www.python.org/) (version 3.6 or later)
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)


## Citations
If you use this code in your research or want to refer to our work, please cite:

```
@inproceedings{klein-nabi-2025-contrastive-perplexity,
    title = "Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models",
    author = "Klein, Tassilo  and
      Nabi, Moin",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",

    abstract = "This paper presents miCSE, a mutual information-based contrastive learning framework that significantly advances the state-of-the-art in few-shot sentence embedding.The proposed approach imposes alignment between the attention pattern of different views during contrastive learning. Learning sentence embeddings with miCSE entails enforcing the structural consistency across augmented views for every sentence, making contrastive self-supervised learning more sample efficient. As a result, the proposed approach shows strong performance in the few-shot learning domain. While it achieves superior results compared to state-of-the-art methods on multiple benchmarks in few-shot learning, it is comparable in the full-shot scenario. This study opens up avenues for efficient self-supervised learning methods that are more robust than current contrastive methods for sentence embedding.",
}
```

## How to obtain support
[Create an issue](https://github.com/SAP-samples/acl2025-contrastive-perplexity/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2025 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
