# ARA corpus

## Recipes

This corpus comprises recipes for 10 different dishes with 11 recipes each. The recipes were obtained from the [Microsoft Research Multimodal Aligned Recipe Corpus](https://github.com/microsoft/multimodal-aligned-recipe-corpus) and parsed into action graphs, i.e. into dependency graphs over phrases of recipe text which describe actions. Typically, such actions are verb forms. 

### CoNLL-U format

The recipes are presented in the [CoNLL-U](https://universaldependencies.org/format.html) where each line represents a token and columns are tab-separated. The relevant columns are:
- the *first* column contains a token index beginning with 1 for the first token.
- the *second* column contains the text token.
- the *fifth* column contains a tag indicating whether the token belongs to an action phrase. The tags are in IOB2 format with the only label being `A` for 'action'.
- the *seventh* column contains a head index pointing to the first token of the action phrase that immediately depends on the current action phrase. The index `0` indicates that there is no such head action.
- the *eighth* column contains the dependency type between the current token and the head token. In the presented action graphs, this column is redundant as the only dependency types are `edge` (if there is a dependency) and `root` (if there isn't).
- the *ninth* column can contain additional `(head,dependency)` pairs if a token has more than one head.

## Alignments





## Sources 
- alignments crowdsourced
- tagging, parsing automated

## Data Format

### CoNLL-U

### Alignments
