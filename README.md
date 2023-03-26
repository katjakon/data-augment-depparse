# Data Augmentation for Dependency Parsing

This project augments dependency trees.

## Requirements
Written with Python 3.10

```
nltk 3.7 
pyyaml 6.0
tabulate 0.9.0
tqdm 4.65.0 
```

## Directory Structure

- `predictions/` contains the predictions on the test set made by the MaltParser trained on different augmented data sets.
- `corpora/data-26K` contains the down sampled corpus
- `augment/` contains the code used for augmenting the data
- `experiments.yaml` contains the paramter specifications for running different experiment. See Section Usage for more information.
- Run `main.py` to augment the data according to the experiments specified in `experiments.yaml`
- Run `eval.py` to evaluate the predictions made by the MaltParser models.

## Usage
All configurations for generating new augmented data can be adjusted in the file `experiments.yaml`<br>

- The key `output` stores the directory in which augmented data is stored.
- The key `input` is the path to the file with the data in the conll format that should be augmented
- Under the key `experiments` the name of the experiment is specified together with augmentations that should be run and their parameters

### Specify Augmentation Experiments
Under the key `experiments` in the config file `experiments.yaml`, add  a unique name for your experiment. You can then add one or more of the following augmentation techniques:
`rotate`, `crop`, `nonce`. Each method comes with a some parameters that can be adjusted:

- For `rotate`, `n` specifies the maximum number of sentences that can be generated. `informed` is a either True or False and decides if position statistics are taken into account. `flexible` is a list of labels that are allowed to move.
- For `crop`, `p` specifies the probability of a label being removed. `relations` is a list of labels that are allowed to be removed. Set to `False` to allow all labels to be removed.
- For `nonce`, `p` specifies the probability of a label being replaced. If `strict` is set to True, words are only replaced by words with the same POS tag.

For example:

```
experiments:
    comb-rot-crop-nonce:
        rotate:
            n: 2
            informed: True
            flexible:
                - "nsubj"
                - "obj"
                - "iobj"
                - "advmod"
        crop:
            p: 0.3
            relations:
                - "iobj"
                - "obj"
                - "advmod"
        nonce:
            p: 0.3
            strict: True
```