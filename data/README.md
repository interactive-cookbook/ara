# ARA corpus

## Recipes

This corpus comprises 10 different dishes with 11 recipes each. The recipes were obtained from the [Microsoft Research Multimodal Aligned Recipe Corpus](https://github.com/microsoft/multimodal-aligned-recipe-corpus) and parsed into action graphs, i.e. into dependency graphs over phrases of recipe text which describe actions. Typically, such actions are verb forms. 

### CoNLL-U format

The recipes are presented in the [CoNLL-U](https://universaldependencies.org/format.html) where each line represents a token and columns are tab-separated. The relevant columns are:
- the *first* column contains a token index beginning with 1 for the first token.
- the *second* column contains the text token.
- the *fifth* column contains a tag indicating whether the token belongs to an action phrase. The tags are in IOB2 format with the only label being `A` for 'action'.
- the *seventh* column contains a head index pointing to the first token of the action phrase that immediately depends on the current action phrase. The index `0` indicates that there is no such head action.
- the *eighth* column contains the dependency type between the current token and the head token. In the presented action graphs, this column is redundant as the only dependency types are `edge` (if there is a dependency) and `root` (if there isn't).
- the *ninth* column can contain additional `(head,dependency)` pairs if a token has more than one head.

## Alignments

This corpus presents crowdsourced alignments between the actions of two recipes at a time. For each dish, we compare 10 such recipe pairs in a cascading manner from the longest to the shortest recipe. Thus, the alignments indicate which action from the shorter of two recipes corresponds best to an action from the longer recipe.

### Data Format

The alignments are collected in one tsv file per dish with the following columns:
- the *first* column (`file1`) contains the name of the source recipe (i.e. the longer of the two recipes).
- the *second* column (`token1`) contains the token index of the first token of an action phrase in the source recipe.
- the *third* column (`file2`) contains the name of the target recipe (i.e. the shorter of the two recipes).
- the *fourth* column (`token2`) contains the token index of the first token of the aligned action in the target recipe.

## ARA 1.0 vs. ARA 1.1

The Microsoft Research Multimodeal Aligned Recipe Corpus comes in different versions. In the version used for the original ARA 1.0 corpus, some punctuation was missing, mostly commas and apostrophes. ARA 1.1 consists of exactly the same recipes and alignment information as ARA 1.0 with the difference that the missing punctuation marks were added to the recipe texts if available from the raw Microsoft corpus. Additionally, all token indices (column 1), head indices (column 7) and the alignment information was adpated to match the texts extended by the punctuation tokens. In particular, neither the tagger, parser nor the alignment model was run again on the recipe texts for ARA 1.1 but all information comes from ARA 1.0. 

Example extract from ARA 1.0 waffles_4.conllu 
```
6	Sift	_	_	B-A	_	17	edge	_	_
7	flour	_	_	O	_	0	root	_	_
8	salt	_	_	O	_	0	root	_	_
9	baking	_	_	O	_	0	root	_	_
10	powder	_	_	O	_	0	root	_	_
11	and	_	_	O	_	0	root	_	_
12	sugar	_	_	O	_	0	root	_	_
13	into	_	_	O	_	0	root	_	_
14	egg	_	_	O	_	0	root	_	_
15	mixture	_	_	O	_	0	root	_	_
16	.	_	_	O	_	0	root	_	_
17	Mix	_	_	B-A	_	25	edge	_	_
```

Corresponding example extract from ARA 1.1 waffles_4.connlu
```
7	Sift	_	_	B-A	_	20	edge	_	_
8	flour	_	_	O	_	0	root	_	_
9	,	_	_	O	_	0	root	_	_
10	salt	_	_	O	_	0	root	_	_
11	,	_	_	O	_	0	root	_	_
12	baking	_	_	O	_	0	root	_	_
13	powder	_	_	O	_	0	root	_	_
14	and	_	_	O	_	0	root	_	_
15	sugar	_	_	O	_	0	root	_	_
16	into	_	_	O	_	0	root	_	_
17	egg	_	_	O	_	0	root	_	_
18	mixture	_	_	O	_	0	root	_	_
19	.	_	_	O	_	0	root	_	_
20	Mix	_	_	B-A	_	28	edge	_	_
```

