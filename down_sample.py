from random import seed, sample
import os



out_dir = "corpora/data-130K"
in_dir = "corpora/data-full"
SEED = "1704"

seed(SEED)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def read_conll(path):
    sents = []
    with open(path, encoding="utf-8") as file:
        sent = ""
        for line in file:
            if line.strip():
                if not line.startswith("#"):
                    sent += line
            else: # Empty line means a new sentence will start.
                sents.append(sent)
                sent = ""
    return sents


data_files = [f for f in os.listdir(in_dir) if "test" not in f]

for f in data_files:
    down_sampled = []
    path = os.path.join(in_dir, f)
    sents = read_conll(path)
    n_sent = len(sents)
    p = int(n_sent*0.5)
    new_sents = sample(sents, p)
    new_path = os.path.join(out_dir, f)
    with open(new_path, "w", encoding="utf-8") as o:
        for s in new_sents:
            o.write(s)
            o.write("\n")