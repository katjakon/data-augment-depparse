# Chunks
from copy import  deepcopy

from nltk.parse import DependencyGraph

class Chunk:

    def __init__(self, dg, address, root=False):
        # Find chunks
        self.dg = dg
        self.head = address
        self.indices = [address]
        if root is False:
            self.indices = sorted(self.find_chunk(dg, address))
        self.min = self.indices[0]
        self.max = self.indices[-1]
        self.projective = True
        self.assert_full_range()

    def assert_full_range(self):
        if self.indices != list(range(self.min, self.max+1)):
            self.projective = False
 
    def find_chunk(self, dg, address):
        deps = dg.nodes[address]["deps"]
        all_deps = [address]
        for d in deps:
            for ad in deps[d]:
                all_deps += self.find_chunk(dg, ad)
        return all_deps
    
    def __len__(self):
        return len(self.indices)

    def __repr__(self) -> str:
        return "Chunk{}".format(self.indices)


class Sentence:

    def __init__(self, dg):
        self.dg = dg
        self.root = self._find_root()
        # Set top dependent to root in case it's been moved:
        self.dg.nodes[0]["deps"]= {"root": [self.root]}
        self.dg.nodes[0]["head"] = None
        self.direct_dependents = self._direct_dependents(self.root) 
        self.chunks = self._identify_chunks()
        self.ordered_nodes = sorted(self.dg.nodes.keys())
    
    def _find_root(self, name="root", dg=None):
        if dg is None:
            dg = self.dg
        root = None
        for node, feats in self.dg.nodes.items():
            if feats["rel"] == name:
               root = node
        if root is None:
            raise ValueError("Could not find root in dependency graph!")
        return root

    def _direct_dependents(self, address):
        node_deps = self.dg.nodes[address]["deps"]
        dependents = []
        for rel_dep in node_deps:
            for idx in node_deps[rel_dep]:
                dependents.append(idx)
        return dependents

    def _identify_chunks(self):
        root_dependents = sorted(self.direct_dependents+[self.root])
        chunks = []
        for dep in root_dependents:
            if dep != self.root:
                chunk = Chunk(self.dg, dep)
            else:
                chunk = Chunk(self.dg, dep, root=True)
            chunks.append(chunk)
        return chunks
    
    def word(self, address):
        return self.dg.nodes[address]["word"]

    def is_nonprojective(self):
        if all(chunk.projective for chunk in self.chunks):
            return False
        return True

    def __repr__(self):
        string = ""
        for i in self.ordered_nodes:
            word = self.word(i)
            if word is not None:
                string += word + " "
        return "Sentence({})".format(string)
    
    def __len__(self):
        return len(self.dg.nodes) - 1 # Subtract one to disregard TOP node.

    @classmethod
    def from_new_order(cls, dg, new_order):
        new_dg = DependencyGraph()
        new_address_dict = {}
        redirects = {}
        # Add TOP:
        new_address_dict[0] = dg.nodes[0]
        redirects[0] = 0
        idx = 1
        for chunk in new_order:
            for i in chunk.indices:
                redirects[i] = idx
                new_address_dict[idx] = deepcopy(dg.nodes[i]) # Keep original dg unchanged later on.
                idx += 1
        new_dg.nodes = new_address_dict
        for address in new_dg.nodes:
            features = new_dg.nodes[address]
            features["address"] = redirects[features["address"] ]
            features["head"] = redirects.get(features["head"], 0)
            for dep, index_list in features["deps"].items():
                new_deps_idx = [redirects[i] for i in index_list]
                features["deps"][dep] = new_deps_idx
        return cls(new_dg)

    @classmethod
    def from_removal(cls, dg, address):
        new_dg = DependencyGraph()
        address_chunk = Chunk(dg, address)
        new_address_dict = dict()
        # Remove dependency of chunk head from root.
        for a in dg.nodes:
            if a not in address_chunk.indices:
                feats = deepcopy(dg.nodes[a])
                for dep, indices in feats["deps"].items():
                    new_idx = [i for i in indices if i not in address_chunk.indices]
                    feats["deps"][dep] = new_idx
                new_address_dict[a] = feats
        redirects = dict()
        for idx, key in enumerate(sorted(new_address_dict.keys())):
            redirects[key] = idx
        final_address = dict()
        for address, features in new_address_dict.items():
            new_a = redirects[address]
            features["address"] = redirects[features["address"] ]
            features["head"] = redirects.get(features["head"], 0)
            for dep, index_list in features["deps"].items():
                new_deps_idx = [redirects[i] for i in index_list]
                features["deps"][dep] = new_deps_idx
            final_address[new_a] = features
        new_dg.nodes = final_address

        return cls(new_dg)

    @classmethod
    def from_replacement(cls, dg, address, updates):
        new_dg = deepcopy(dg)
        address_feats = new_dg.nodes[address]
        address_feats.update(updates)
        return cls(new_dg)

    def __eq__(self, __o: object) -> bool:
        return __o == self.dg.nodes

    def __hash__(self) -> int:
        return hash(str(self))

class Corpus:

    def __init__(self, data_file):
        self.sentences = self.read_conll(data_file)

    def read_conll(self, path):
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
        return [Sentence(DependencyGraph(s, top_relation_label="root")) for s in sents]
    
    def position_statistics(self):
        stats = dict()
        for sentence in self.sentences:
            dg = sentence.dg
            for address, feats in dg.nodes.items():
                rel = feats["rel"]
                stats.setdefault(rel, dict())
                deps = feats["deps"]
                for d in deps:
                    if d not in stats[rel]:
                        stats[rel][d] = {"left": 0, "right": 0}
                    for d_idx in deps[d]:
                        if d_idx < address:
                            stats[rel][d]["left"] += 1
                        else: 
                            stats[rel][d]["right"] += 1
        # Normalize:
        for rel in stats:
            for d in stats[rel]:
                sum_children = sum(stats[rel][d].values())
                if sum_children == 0:
                    continue
                stats[rel][d]["left"] /= sum_children
                stats[rel][d]["right"] /= sum_children
        return stats

    def nonce_features(self, keys={"word", "lemma", "ctag", "tag", "feats"}):
        nonce_dict = dict()
        for sent in self.sentences:
            for node_address in sent.dg.nodes:
                feats = sent.dg.nodes[node_address]
                rel = feats["rel"]
                nonce_dict.setdefault(rel, set())
                updates = {key: value for key, value in feats.items() if key in keys}
                nonce_dict[rel].add(tuple(updates.items()))
        return nonce_dict
    
    def n_tokens(self):
        return sum(len(s) for s in self.sentences)
    
    def __iter__(self):
        return iter(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]
    
    def __len__(self):
        return len(self.sentences)


if __name__ == "__main__":
    path = "data\de_gsd-ud-dev.conllu"
    corpus = Corpus(path)
    stats = corpus.position_statistics()
    nonce = corpus.nonce_features()
    for i in nonce["nsubj"]:
        print(dict(i))
    s = corpus.sentences[0]
    copy_sent = deepcopy(s)
    print({s, copy_sent}) 
    # # Manasse ist ein einzigartiger Parfümeur
    # sents = []
    # with open(path, encoding="utf-8") as file:
    #     sent = ""
    #     for line in file:
    #         if line.strip():
    #             if line.startswith("#"):
    #                 continue
    #             sent += line
    #         else:
    #             sents.append(sent)
    #             sent = ""   
    # example_sent = sents[2]
    # ex2 = sents[3]
    # dg = DependencyGraph(example_sent)
    # sent = Sentence(dg)
    # repl = Sentence(DependencyGraph(ex2))
    # print(sent)
    # print(repl)
    # nsubj_repl_dict = dict()
    # repl_address = None
    # for chunk in sent.chunks:
    #     chunk_head_feats = sent.dg.nodes[chunk.head]
    #     if chunk_head_feats["rel"] == "nsubj":
    #         repl_address = chunk.head
    # repl_keys = {"word", "lemma", "ctag", "tag", "feats"}
    # for chunk in repl.chunks:
    #     chunk_head_feats = repl.dg.nodes[chunk.head]
    #     if chunk_head_feats["rel"] == "nsubj":
    #         nsubj_repl_dict = {feat: val for feat, val in chunk_head_feats.items() if feat in repl_keys}
    # print(nsubj_repl_dict, repl_address)
    # new_sent = Sentence.from_replacement(sent.dg, repl_address, nsubj_repl_dict)
    # print(new_sent)
    # print(sent)
