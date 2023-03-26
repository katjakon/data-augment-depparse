# This file contains classes for handling conll data and dependency trees.
from copy import  deepcopy

from nltk.parse import DependencyGraph

class Chunk:

    def __init__(self, dg, address, root=False):
        """Create a chunk object.

        Args:
            dg (nltk.DependencyGraph): The Dependency Graph that contains the chunk.
            address (int): Address of the head of the chunk.
            root (bool, optional): Whether or not this chunk is the root. Defaults to False.
        """
        self.dg = dg
        self.head = address
        self.indices = [address]
        if root is False: # Finding chunks for root results in whole sentence.
            self.indices = sorted(self.find_chunk(dg, address))
        self.min = self.indices[0]
        self.max = self.indices[-1]
        self.projective = True
        self.assert_full_range()

    def assert_full_range(self):
        if self.indices != list(range(self.min, self.max+1)):
            self.projective = False
 
    def find_chunk(self, dg, address):
        """ Finds all dependents of an address in a dependency graph.
        Args:
            dg (nltk.DependencyGraph): The 
            address (int): Address of node in dependency graph.

        Returns:
            list: List of all dependents adresses and the node address itself
        """
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
        """Creates a Sentence object.

        Args:
            dg (nltk.DependencyGraph): Dependency graph that contains sentence information.
        """
        self.dg = dg
        self.root = self._find_root()
        # Set top dependent to root in case it's been moved:
        self.dg.nodes[0]["deps"]= {"root": [self.root]}
        self.dg.nodes[0]["head"] = None
        self.direct_dependents = self._direct_dependents(self.root) 
        self.chunks = self._identify_chunks()
        self.ordered_nodes = sorted(self.dg.nodes.keys()) # This is the real word order of the sentence.
    
    def _find_root(self, name="root", dg=None):
        """Finds address of root in Dependency Graph.

        Args:
            name (str, optional): String that signifies root. Defaults to "root".
            dg (nltk.DependencyGraph, optional): Dependency Graph where root is stored. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            int: Address of root node.
        """
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
        """Identifies all direct dependents of an label

        Args:
            address (int): Address of label for which direct dependents should be identified.

        Returns:
            list: list of address indices
        """
        node_deps = self.dg.nodes[address]["deps"]
        dependents = []
        for rel_dep in node_deps:
            for idx in node_deps[rel_dep]:
                dependents.append(idx)
        return dependents

    def _identify_chunks(self):
        """Gets all chunks from root.

        Returns:
            List[Chunk]: List of Chunks, each one being a direct dependent of the root with its children or the root itself.
        """
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
        """Gets word string from address indix."""
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
        """Instantiates a new Sentence object from a new word order.

        Args:
            dg (nltk.DependencyGraph): Dependency Graph with old chunk order.
            new_order (List[Chunk]): List of reorderd chunks

        Returns:
            Sentence: New Sentence object with reorderd nodes.
        """
        new_dg = DependencyGraph()
        new_address_dict = {}
        redirects = {}
        # Add TOP:
        new_address_dict[0] = dg.nodes[0]
        redirects[0] = 0
        idx = 1 # Top is always 0, stays at same position.
        # Store where addresses have been moved to.
        for chunk in new_order:
            for i in chunk.indices:
                redirects[i] = idx
                new_address_dict[idx] = deepcopy(dg.nodes[i]) # Keep original dg unchanged later on.
                idx += 1
        new_dg.nodes = new_address_dict
        # Change all values for address, head and dependents to new addresses.
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
        """Removes node and all its children from dependency graph.

        Args:
            dg (nltk.DependencyGraph: old Dependency Graph
            address (int): Address that should be removed.

        Returns:
            Sentence: New Sentence with removed node.
        """
        new_dg = DependencyGraph()
        address_chunk = Chunk(dg, address) # Identify all children.
        new_address_dict = dict()
        # Remove dependency of chunk head from root.
        for a in dg.nodes:
            if a not in address_chunk.indices: # Only add to new dg if its not in removed chunk.
                feats = deepcopy(dg.nodes[a])
                for dep, indices in feats["deps"].items():
                    # Remove addresses of removed chunk.
                    new_idx = [i for i in indices if i not in address_chunk.indices]
                    feats["deps"][dep] = new_idx
                new_address_dict[a] = feats
        redirects = dict()
        # Address need to be a full sequence, we need to change the address accordingly.
        # Ex: Original 1 2 3 --> Remove 2 --> 1 3 Full range --> 1 2
        for idx, key in enumerate(sorted(new_address_dict.keys())):
            redirects[key] = idx
        final_address = dict()
        # Change all values for address, head and dependents to new addresses.
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
        """Replace a word in a tree by a different one.

        Args:
            dg (nltk.DependencyGraph): _Old dependency graph
            address (int): Address that should be replaced.
            updates (dict): Values that should be updated, e.g. word, lemma etc.

        Returns:
            Sentence: new Sentence object with replaced word at given address.
        """
        new_dg = deepcopy(dg) # Do not change original dg.
        address_feats = new_dg.nodes[address]
        address_feats.update(updates) # This does not change dependents or head etc.
        return cls(new_dg)

    def __eq__(self, __o: object) -> bool:
        return __o == self.dg.nodes

    def __hash__(self) -> int:
        return hash(str(self))

class Corpus:

    def __init__(self, data_file):
        """Create a corpus object.

        Args:
            data_file (str): Path to conll file.
        """
        self.sentences = self.read_conll(data_file)

    def read_conll(self, path):
        """Reads in file in conll format.

        Args:
            path (str): Path to the conll file.

        Returns:
            List[Sentence]: List of Sentence objects
        """
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
        """Calculates how likely labels are to be to left or right of their head.

        Returns:
            dict: Nested dict 
        """
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
        """Generates dictionary with possible replacements for all relation labels.

        Args:
            keys (dict, optional): _description_. Defaults to {"word", "lemma", "ctag", "tag", "feats"}.

        Returns:
            dict: Dict[str, Set[Tuple[str, list]]], keys are relation labels, returns set of tuples 
            where first item is key from key and rest is value.
        """
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
        """Number of tokens in corpus"""
        return sum(len(s) for s in self.sentences)
    
    def __iter__(self):
        return iter(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]
    
    def __len__(self):
        return len(self.sentences)
