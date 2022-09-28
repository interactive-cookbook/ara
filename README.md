# Aligned Recipe Actions (ARA)

This repository contains the **A**ligned **R**ecipe **A**ctions corpus from the paper below. 

> Lucia Donatelli, Theresa Schmidt, Debanjali Biswas, Arne Köhn, Fangzhou Zhai, Alexander Koller (2021).
> [Aligning Actions Across Recipe Graphs](https://aclanthology.org/2021.emnlp-main.554/).
> Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).

The code for the most recent version of the recipe-action alignment model is available [here](https://github.com/interactive-cookbook/alignment-models).

Contact Person: Dr. Lucia Donatelli (donatelli@coli.uni-saarland.de)

## Licenses

Apache 2.0

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


## Data / Corpus

Our corpus consists of recipes for 10 different dishes from the [Microsoft Research Multimodal Aligned Recipe Corpus](https://github.com/microsoft/multimodal-aligned-recipe-corpus). For each dish, there are 11 recipes parsed into action graphs. We provide crowdsourced action alignments between the action phrases of **10 recipe pairings per dish**. These alignments indicate which action from the shorter of two recipes corresponds best to an action from the longer recipe. 

The data folder contains an updated version (ARA 1.1) of the originally created corpus in which part of the within-sentence punctuation was missing. This punctuation was added back in the updated version. For more details on the differences between ARA 1.0 and ARA 1.1 see [here](https://github.com/kastein/ara/blob/main/data/README.md#ara-10-vs-ara-11).
The originally created corpus version is available as Release [ARA 1.10 Corpus](https://github.com/kastein/ara/releases/tag/v1.0).

