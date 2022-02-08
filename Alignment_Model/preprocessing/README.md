# Tagger and Parser

The recipes in the ARA corpus were tagged and parsed by the here presented models. The models were trained on the data in [./data](./data).

## Environment setup
1. Create a conda environment with Python 3.7
```
conda create -n allennlp python=3.7
```
2. Activate the new environment
```
conda activate allennlp
```
3. Install allennlp using pip (we use version 0.8.4)
```
pip install allennlp==0.8.4
```
You may also need to install further packages with versions as specified in `requirements.txt`.

Usage:
```
pip install -r requirements.txt
```

**Internal note**: this is already set up on the Saarland servers, see details [here](https://github.com/coli-saar/am-parser/wiki/Setup-and-file-locations-on-the-Saarland-servers).

## Parameter configuration

Adjust parameters including file paths in the respective `.json` config files, as needed. By default, the paths point to datasets in [`./data`](./data). See there for details about the dataset. 

Both our models consume data in CoNLL format where each line represents a token and columns are tab-separated. The column DEPRELS contains additional dependency relations if a token has more than one head.The tagger requires data in the [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) format with the relevant columns being the first (TEXT) and the fourth (LABEL). The parser requires data in the [CoNLL-U](https://universaldependencies.org/format.html) format with the relevant columns being the second (FORM), the  fifth (LABEL), the seventh (HEAD) and the eighth (DEPREL). 

Models:
- `tagger_config.json` - BiLSTM-CRF tagger using English ELMo embeddings
- `parser_config.json` - Biaffine dependency parser (Dozat and Manning, 2017)

For the ELMo tagger, we use the following ELMo parameters (i.e. options and weights): [weights and options](https://allennlp.s3.amazonaws.com/models/ner-model-2018.12.18.tar.gz) (use the weights and options files under `fta/` after unzipping).

**Internal note**: the ELMo options and weight files can be found on the Saarland servers at `/proj/cookbook/`.

The weights and options files should be named and placed according to the paths specified in the `*config.json` files; alternatively, adjust the paths in the `*config.json` files.

## Training

Run `allennlp train [params] -s [serialization dir]` to train a model, where
- `[params]` is the path to the .json config file.
- `[serialization dir]` is the directory to save trained model, logs and other results.

## Evaluation
Run `allennlp evaluate [archive file] [input file] --output-file [output file]` to evaluate the model on some evaluation data, where
- `[archive file]` is the path to an archived trained model.
- `[input file]` is the path to the file containing the evaluation data.
- `[output file]` is an optional path to save the metrics as JSON; if not provided, the output will be displayed on the console.

### Performance

Tagger performance (on `./data/Tagger/test.conll03`):

Precision | Recall | F-Score
 --- | --- | ---
89.9 ± 0.5 | 89.2 ± 0.4 | 89.6 ± 0.3


<!--
Our tagger's performance compared to Y'20's performance and inter-annotator agreement (IAA).

Model | Corpus | Embedder | Precision  | Recall | F-Score  
--- | --- | --- | --- | --- | ---
IAA | 100-r by Y'20 | | 89.9 | 92.2 | 90.5
| | | | | 
Y'20 | 300-r by Y'20 | | 86.5 | 88.8 | 87.6
Our tagger  | [300-r by Y'20](https://github.com/coli-saar/interactive-cookbook/tree/master/data/English/Tagger) | [English ELMo](Taggers/tagger_with_english_elmo.json) | **89.9** ± 0.5 | **89.2** ± 0.4 | **89.6** ± 0.3
Our tagger  | [300-r by Y'20](https://github.com/coli-saar/interactive-cookbook/tree/master/data/English/Tagger) | [multilingual BERT](Taggers/tagger_with_bert_config.json) | 88.7 ± 0.4 | 88.4 ± 0.1 | 88.5 ± 0.2
| | | | | 
Our tagger  | [German](https://github.com/coli-saar/interactive-cookbook/tree/master/data/German/Tagger) | [German ELMo](Taggers/tagger_with_german_elmo.json) | 79.2 ± 1.4 | 81.2 ± 1.8 | 80.2 ± 1.6
Our tagger  | [German](https://github.com/coli-saar/interactive-cookbook/tree/master/data/German/Tagger) | [multilingual BERT](Taggers/tagger_with_bert_config.json) | 75.3 ± 0.8 | 76.0 ± 1.0 | 75.7 ± 0.9
-->



Parser performance (on `./data/Parser/test.conllu`):

Tag Source | Precision | Recall | F-Score
--- | --- | --- | ---
gold tags | 80.4 ± 0.0 | 76.1 ± 0.0 | 78.2 ± 0.0 
our tagger | 74.4 ± 0.5 | 70.4 ± 1.0 | 72.3 ± 0.8


<!--
Our parser's performance compared to Y'20's performance and inter-annotator agreement (IAA).

Model | Corpus |  Tag source | Precision  | Recall | F-Score 
--- | --- | --- | --- | --- | --- 
IAA | 100-r by Y'20 | gold tags | 84.4 | 80.4 | 82.3
 |  |  |  |  |
Y'20 | 300-r by Y'20 | gold tags | 73.7 | 68.6 | 71.1
Our parser | [300-r by Y'20](https://github.com/coli-saar/interactive-cookbook/tree/master/data/English/Parser) | gold tags | 80.4 ± 0.0 | 76.1 ± 0.0 | **78.2** ± 0.0
 |  |  |  |  |
Our parser  | [German](https://github.com/coli-saar/interactive-cookbook/tree/master/data/German/Parser) | gold tags | 69.3 ± 0.0 | 91.3 ± 0.0 | 78.8 ± 0.0

Our parser's performance on machine-tagged data:

Model | Corpus |  Tag source | Precision  | Recall | F-Score 
--- | --- | --- | --- | --- | --- 
Y'20 | 300-r by Y'20 | Y'20 tagger | 51.1 | 37.7 | 43.3
Our parser  | [300-r by Y'20](https://github.com/coli-saar/interactive-cookbook/tree/master/data/English/Parser) | [our ELMo tagger](Taggers/tagger_with_english_elmo.json) | 74.4 ± 0.5 | 70.4 ± 1.0 | **72.3** ± 0.8
Our parser  | [German](https://github.com/coli-saar/interactive-cookbook/tree/master/data/German/Parser) | [German ELMo tagger](Taggers/tagger_with_german_elmo.json) | 56.5 ± 1.1 | 82.8 ± 2.2 | 67.1 ± 0.5
-->

## Prediction

Run `allennlp predict [archive file] [input file] --use-dataset-reader --output-file [output file]` to parse a file with a pretrained model, where
- `[archive file]` i is the path to an archived trained model.
- `[input file]` is the path to the file you want to parse; this file should be in the same format as the training data, i.e. CoNLL-2003 for the tagger and CoNLL-U for the parser.
- `use-dataset-reader` tells the parser to use the same dataset reader as it used during training.
- `[output file]` is an optional path to save parsing results as JSON; if not provided, the output will be displayed on the console.

The ouput of the parser will be in JSON format. To transform this into the better readable CoNLL-U format, run the [read_prediction.py](./read_prediction.py) with the following arguments:
- `-m [mode]` where `[mode]` can be either `analysis` (for error analysis), `tagger_p2c` (translates tagger output into CoNLL-U file) or `parser_p2c` (translates parser output into CoNLL-U format).
- `-p [model output]` where `[model output]` is the tagger or parser output in json format. 
- `-o [file]` to state where you want to save the output. Default: `[model output].conllu`.

For sample inputs and outputs see [data/Samples](./data/Samples). 
