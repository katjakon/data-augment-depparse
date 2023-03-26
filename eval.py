from nltk.parse import DependencyGraph, DependencyEvaluator
from tabulate import tabulate
import os

pred_path = "predictions"
gold_path = "corpora/data-full/de_gsd-ud-test.conllu"
filter = "baseline"

pred_exps = os.listdir(pred_path)


def read_conll(path):
    sent_dict = dict()
    with open(path, encoding="utf-8") as file:
        sent = ""
        sent_id = None
       
        for line in file:

            if line.strip():
                if line.startswith("#"):
                    sent_id = line
                else:
                    sent += line
            else: # Empty line means a new sentence will start.
                sent_dict[sent_id] = DependencyGraph(sent, top_relation_label="root")
                sent = ""

    return sent_dict

sent_gold = read_conll(gold_path)
results = []
for p_file in pred_exps:
    # if filter not in p_file:
    #     continue
    sent_pred = read_conll(os.path.join(pred_path, p_file))
    ordered_pred = []
    ordered_gold = []
    for key in sent_gold.keys():
        ordered_gold.append(sent_gold[key])
        ordered_pred.append(sent_pred[key])
    eval_dep = DependencyEvaluator(ordered_pred, ordered_gold)
    las, uas = eval_dep.eval()
    results.append((p_file.removesuffix(".conll"), round(las, 3), round(uas, 3)))

sort_res = sorted(results, key=lambda x: x[0], reverse=True)

print(tabulate(sort_res, headers=["Experiment", "LAS", "UAS"]))
    

