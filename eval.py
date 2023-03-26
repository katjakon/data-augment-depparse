from nltk.parse import DependencyGraph, DependencyEvaluator
from tabulate import tabulate
import os
import argparse

pred_path = "predictions"
gold_path = "corpora/data-full/de_gsd-ud-test.conllu"



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

def main():
    parser = argparse.ArgumentParser(
                    prog='EvalPred',
                    description='Evaluate predictions.')
    parser.add_argument('--gold', default="corpora/data-26k/de_gsd-ud-test.conllu")
    parser.add_argument("--pred_dir", default="predictions")
    parser.add_argument("--sort_by", choices=["name", "las", "uas"])
    args = parser.parse_args()
    sent_gold = read_conll(args.gold)
    pred_exps = os.listdir(args.pred_dir)
    sort_by = args.sort_by
    if sort_by == "name":
        sort_idx = 0
    elif sort_by == "las":
        sort_idx = 1
    else:
        sort_idx = 2
    results = []
    for p_file in pred_exps:
        sent_pred = read_conll(os.path.join(pred_path, p_file))
        ordered_pred = []
        ordered_gold = []
        for key in sent_gold.keys():
            ordered_gold.append(sent_gold[key])
            ordered_pred.append(sent_pred[key])
        eval_dep = DependencyEvaluator(ordered_pred, ordered_gold)
        las, uas = eval_dep.eval()
        results.append((p_file.removesuffix(".conll"), round(las, 3), round(uas, 3)))

    sort_res = sorted(results, key=lambda x: x[sort_idx], reverse=True)

    print(tabulate(sort_res, headers=["Experiment", "LAS", "UAS"]))

if __name__ == "__main__":
    main()
        

