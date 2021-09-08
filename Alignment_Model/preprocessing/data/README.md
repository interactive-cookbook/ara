# Tagger and Parser training corpus

The **original data** by Yamakata et al. (2020) can be found at [flowgraph](https://sites.google.com/view/yy-lab/resource/english-recipe-flowgraph) and [NER](https://sites.google.com/view/yy-lab/resource/english-recipe-ner).

The data is **split** into train, development and test set with proportions 80:10:10.

| | Train | Development | Test
:--- | :---: | :---: | :---:
**# recipes** | 240 | 30 | 30

**POS tags** are included in the corpus but were not used for parsing. 

### Dependency parser

The dependency parser requires data in [CoNLL-U](https://universaldependencies.org/format.html) format where each line represents a token and columns are tab-separated. The relevant columns are FORM, LABEL, HEAD, DEPREL (only FORM and LABEL as input to prediction). The column DEPRELS contains additional dependency relations if a token has more than one head.

Excerpt from the corpus in CoNLL-U format:
```
ID	FORM 	(LEMMA)	POS	LABEL	(FEATS)	HEAD	DEPREL	DEPRELS	(MISC)

10	Bring	_	VV0	B-Ac	_	15	t	_	_
11	to	_	II	I-Ac	_	0	root	_	_
12	the	_	AT	I-Ac	_	0	root	_	_
13	boil	_	NN1	I-Ac	_	0	root	_	_
14	and	_	CC	O	_	0	root	_	_
15	cook	_	VV0	B-Ac	_	149	f-eq	_	_
16	till	_	NN1	O	_	0	root	_	_
17	tender	_	NN1	B-Sf	_	15	v-tm	_	_
18	.	_	.	O	_	0	root	_	_
19	Meanwhile	_	RR	O	_	0	root	_	_
20	,	_	,	O	_	0	root	_	_
21	add	_	VV0	B-Ac	_	29	d	[(35,	'f-eq')]
22	the	_	AT	O	_	0	root	_	_
23	carrots	_	NN2	B-F	_	21	t	_	_
```

### Tagger

The tagger requires data in CoNLL-2003 format where each line represets a token and columns are tab-separated. The relevant columns are TOKEN and LABEL (only TOKEN as input to prediction).

Excerpt from the corpus [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) format:

```
TOKEN POS (CHUNK) LABEL

Place	VV0	O	B-Ac
the	AT	O	O
potatoes	NN2	O	B-F
in	II	O	O
a	AT1	O	O
pan	NN1	O	B-T
of	IO	O	O
water	NN1	O	B-F
.	.	O	O
Bring	VV0	O	B-Ac
to	II	O	I-Ac
the	AT	O	I-Ac
boil	NN1	O	I-Ac
and	CC	O	O
```

### Samples

We provide some sample files for the input and output of our tagger and parser. All files contain the same single recipe which is part of the training split of the corpus.
