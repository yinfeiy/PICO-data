# PICO-data for Participants

This repo contains raw annotaions for PICO dataset used in paper:

[Aggregating and Predicting Sequence Labels from Crowd Annotations](https://www.ischool.utexas.edu/~ml/papers/nguyen-acl17.pdf)
An Thanh Nguyen, Byron C. Wallace, Junyi Jessy Li, Ani Nenkova and Matthew Lease
Association for Computational Linguistics (ACL).

A SDK and sample codes are provided for retrieving the annotations.

## Description

The dataset is in [annotations/](./annotations/), it is splited into 4 parts:
1. train contains random selected 3549 abstracts.
2. dev contains random selected 500 abstracts.
3. test contains random selected 500 abstracts.
4. acl17-test contains 191 abstarcts with annotations by a medical student.

In each folder:
1. *ICO-annos-crowdsourcing.json* contains annotations from crowd sourced workers.
2. *ICO-annos-crowdsourcing-agg.json* contains aggregated results from crowd sourced annotations.
3. *ICO-annos-professional.json* for [acl17-test](./annotaions/acl17-test/) only, contains annotations from a medical student.

## Environment and dependencies:

- Python 2
- [spaCy](http://spacy.io) for basic tokenization etc
- Sample code in [src/examples/](./src/examples/) folder

```bash
cd src
python -m examples.load_annotation
```
