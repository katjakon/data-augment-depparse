from math import factorial
from random import sample, random, choice, seed

from data import Sentence

seed(1704)

class Augment:

    ROOT = "root"
    FLEX = ["nsubj", "obj", "advmod", "iobj", "obl", "xcomp", "acl", "advcl", "ccomp", "case"]

    def __init__(self, corpus):
        self.corpus = corpus
        self.stats = self.corpus.position_statistics()

    def _rotate(self, sentence, flexible):
        chunks = sentence.chunks
        new_order_dict = {i: c for i, c in enumerate(chunks)}
        # Filter for flexible chunks
        flexible_chunks = []
        for idx, chunk in new_order_dict.items():
            head = chunk.head
            relation = sentence.dg.nodes[head]["rel"].split(":")[0] # We ignore further specifications for relations.
            if relation in flexible:
                flexible_chunks.append((idx, chunk))
        old_idx = [idx for idx, _ in flexible_chunks]
        new_idx = sample(old_idx, len(flexible_chunks))
        for (_, chunk), new_i in zip(flexible_chunks, new_idx):
            new_order_dict[new_i] = chunk
        new_order = list(new_order_dict.values())
        rotated = Sentence.from_new_order(sentence.dg, new_order)
        return rotated
    
    def generate_rotations(self, sentence, n=3,  informed=False, max_rotations=100, flexible=None):
        if flexible is None:
            flexible = self.FLEX
        new_sents = []
        poss_rots = factorial(len(sentence.chunks))
        n_rotations = min(poss_rots, max_rotations)
        for _ in range(n_rotations):
            rot_sent = self._rotate(sentence, flexible=flexible)
            new_sents.append(rot_sent)
        if informed is False:
            n_samples = min(len(new_sents), n)
            n_new = sample(new_sents, n_samples)
        else:
            pos_stats = self.stats[self.ROOT]
            sorted_sents = sorted(new_sents, key=lambda x: self._sentence_prob(x, pos_stats))
            n_new = sorted_sents[:n]
        return n_new

    def _sentence_prob(self, sentence, pos_stats):
        p = 1
        for chunk in sentence.chunks:
            head_rel = sentence.dg.nodes[chunk.head]["rel"]
            if head_rel not in pos_stats:
                continue
            if chunk.head < sentence.root:
                p *= pos_stats[head_rel]["left"]
            elif chunk.head > sentence.root:
                p *= pos_stats[head_rel]["right"]
        return p

    def generate_crops(self, sentence, relations=False, p=0.5):
        crops = []
        for chunk in sentence.chunks:
            head_relation = sentence.dg.nodes[chunk.head]["rel"]
            if relations is not False:
                if head_relation not in relations:
                    continue
            if random() <= p and chunk.head != sentence.root:
                cropped_sent = Sentence.from_removal(sentence.dg, chunk.head)
                crops.append(cropped_sent)
        return crops
    
    def generate_nonce(self, sentence, p=0.5, strict=False):
        nonces = []
        nonce_dict = self.corpus.nonce_features()
        for chunk in sentence.chunks:
            if random() <= p:
                chunk_head_rel = sentence.dg.nodes[chunk.head]["rel"]
                possible_nonces = nonce_dict.get(chunk_head_rel)
                if possible_nonces is None:
                    continue
                if strict is True:
                    original_tag = sentence.dg.nodes[chunk.head]["tag"]
                    possible_nonces = self.filter_nonces(original_tag, possible_nonces)
                if len(possible_nonces) == 0:
                    continue
                random_nonce = choice(list(possible_nonces))
                nonce_sent = Sentence.from_replacement(sentence.dg, chunk.head, random_nonce)
                nonces.append(nonce_sent)
        return nonces

    def filter_nonces(self, original_tag, nonces):
        return [n for n in nonces if dict(n)["tag"] == original_tag]
    
    def write(self, sentences, path):
        with open(path, "w", encoding="utf-8") as file:
            for sent, augmentation in sentences.items():
                conll_str = sent.dg.to_conll(style=10)
                file.write(conll_str)
                file.write("\n")
                for aug_s in augmentation:
                    conll_str = aug_s.dg.to_conll(style=10)
                    file.write(conll_str)
                    file.write("\n")





        


        