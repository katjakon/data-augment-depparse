import os
from random import seed

from tqdm import tqdm
import yaml

from augment import Augment
from data import Corpus

CONFIG = "experiments.yaml"
seed(1704)
with open(CONFIG, encoding="utf-8") as cfg:
    config_dict = yaml.safe_load(cfg)

# Load dictionary with all experiments configurations.
experiments = config_dict.get("experiments", dict())

# Make output directory.
out_dir = config_dict["output"]
path = os.path.join(out_dir)
if not os.path.exists(path):
    os.mkdir(path)

# Get input files.
in_dir = config_dict["input"]["dir"]
in_train = config_dict["input"]["train"]
in_val = config_dict["input"]["val"]

path_train = os.path.join(in_dir, in_train)
corpus = Corpus(path_train)
augment = Augment(corpus)

n_token = corpus.n_tokens()
print("Number of token in input data: ", n_token)
print("Number of sentences in input data: ", len(corpus))

for exp_name in experiments:
    aug_sent_dict = dict()
    augment_config = experiments[exp_name]
    rotate_kwargs = augment_config.get("rotate", False)
    crop_kwargs = augment_config.get("crop", False)
    nonce_kwargs = augment_config.get("nonce", False)
    n_augmented = 0
    for sent in tqdm(corpus, desc=f"Augmenting for configuration '{exp_name}'"):
        augs = set()
        if rotate_kwargs is not False:
            rotation_sents = augment.generate_rotations(sentence=sent, **rotate_kwargs)
            augs.update(rotation_sents)
        if crop_kwargs is not False:
            crop_sents = augment.generate_crops(sentence=sent, **crop_kwargs)
            augs.update(crop_sents)
        if nonce_kwargs is not False:
            nonce_sents = augment.generate_nonce(sentence=sent, **nonce_kwargs)
            augs.update(nonce_sents)
        if sent in augs: # Avoid writing the same sentence twice to the output.
            augs.remove(sent)
        aug_sent_dict[sent] = augs
        n_augmented += len(augs)
    print(f"Generated {n_augmented} sentences, {round(n_augmented/len(corpus), ndigits=2)} on average per input sentence.")
    out_exp_dir = os.path.join(out_dir, exp_name)
    if not os.path.exists(out_exp_dir): 
        os.mkdir(out_exp_dir)
    out_path = os.path.join(out_exp_dir, "augmented.conll")
    augment.write(aug_sent_dict, out_path)
        
