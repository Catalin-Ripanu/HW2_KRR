from typing import List, Dict, Tuple, Any
import pandas as pd
import networkx as nx
import itertools
import random
from copy import deepcopy
import numpy as np


def almost_equal(x: float, y: float, threshold: float = 1e-6) -> bool:
    return abs(x - y) < threshold


def factor_crossjoin(
    f1: pd.DataFrame, f2: pd.DataFrame, how: str = "outer", **kwargs
) -> pd.DataFrame:
    """
    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401
    :param f1 first factor represented as a pandas DataFrame
    :param f2 second factor represented as a pandas DataFrame
    :param how type of the join to perform on factors - for the crossjoin the default is "outer"
    :param kwargs keyword arguments that will be passed to pd.merge()
    :return cross join of f1 and f2
    """
    f1["_tmpkey"] = 1
    f2["_tmpkey"] = 1

    res = pd.merge(
        f1.reset_index(), f2.reset_index(), on="_tmpkey", how=how, **kwargs
    ).drop("_tmpkey", axis=1)
    res = res.set_index(keys=f1.index.names + f2.index.names)

    f1.drop("_tmpkey", axis=1, inplace=True)
    f2.drop("_tmpkey", axis=1, inplace=True)

    return res


def multiply_factors(f1: pd.DataFrame, f2: pd.DataFrame) -> pd.DataFrame:
    f1_vars = f1.index.names
    f2_vars = f2.index.names

    common_vars = [v for v in f1_vars if v in f2_vars]

    if not common_vars:
        ### we have to do a cross join
        f_res = factor_crossjoin(f1, f2)
        f_res["prob"] = f_res["prob_x"] * f_res["prob_y"]
        f_res = f_res.drop(columns=["prob_x", "prob_y"])

    else:
        ### there is a set of common vars, so we merge on them
        disjoint_vars = [v for v in f1_vars if v not in f2_vars] + [
            v for v in f2_vars if v not in f1_vars
        ]
        f_res = pd.merge(
            f1.reset_index(), f2.reset_index(), on=common_vars, how="inner"
        ).set_index(keys=disjoint_vars + common_vars)
        f_res["prob"] = f_res["prob_x"] * f_res["prob_y"]
        f_res = f_res.drop(columns=["prob_x", "prob_y"])

    return f_res


def sumout(f: pd.DataFrame, vars: List[str]) -> pd.DataFrame or float:
    f_vars = f.index.names
    remaining_vars = [v for v in f_vars if v not in vars]

    if remaining_vars:
        return f.groupby(level=remaining_vars).sum()
    else:
        # if we are summing out all values return the sum of all entries
        return f["prob"].sum()


def normalize(f: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a factor table with better numerical stability.
    """
    if not isinstance(f, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    f = f.copy()
    total = f["prob"].sum()

    if total == 0:
        # If all probabilities are zero, set uniform distribution
        f["prob"] = 1.0 / len(f)
    else:
        # Normalize
        f["prob"] = f["prob"] / total

    # Replace any NaN values with small probability
    f["prob"] = f["prob"].fillna(1e-10)

    return f.sort_index()


class Factor:
    """
    Place holder class for a Factor in a factor graph (implicitly also within a junction tree)
    """

    def __init__(self, vars: List[str], table: pd.DataFrame):
        """
        Instantiate a factor
        :param vars: random variables of a factor
        :param table: factor table that is proportional to probabilities
        """
        self.vars = vars
        self.table = table


class BayesNode:
    def __init__(
        self,
        var_name: str = None,
        parent_nodes: List["BayesNode"] = None,
        cpd: pd.DataFrame = None,
    ):
        """
        Defines a binary random variable in a bayesian network by
        :param var_name: the random variable name
        :param parent_nodes: the parent random variables (conditioning variables)
        :param cpd: the conditional probability distribution given in the form of a Pandas Dataframe which has a
        multilevel index that contains all possible binary value combinations for the random variable and its parents

        An example CPD is:
                   prob
            c a b
            1 1 1  0.946003
                0  0.080770
              0 1  0.664979
                0  0.223632
            0 1 1  0.751246
                0  0.355359
              0 1  0.688208
                0  0.994031

        The first level of the index is always the `var_name` random variable (the one for the current node)
        The next levels in the index correspond to the parent random variables
        """
        self.var_name = var_name
        self.parent_nodes = parent_nodes
        self.cpd = cpd

    def to_factor(self) -> Factor:
        factor_vars = [self.var_name] + [p.var_name for p in self.parent_nodes]
        return Factor(vars=factor_vars, table=self.cpd.copy(deep=True))

    def pretty_print_str(self):
        res = ""
        res += "Node(%s" % self.var_name
        if self.parent_nodes:
            res += " | "
            for p in [p.var_name for p in self.parent_nodes]:
                res += p + " "
            res += ")"
        else:
            res += ")"

        res += "\n"
        res += str(self.cpd)
        res += "\n"

        return res

    def __str__(self):
        res = ""
        res += "Node(%s" % self.var_name
        if self.parent_nodes:
            res += " | "
            for p in [p.var_name for p in self.parent_nodes]:
                res += p + " "
            res += ")"
        else:
            res += ")"

        return res

    def __repr__(self):
        return self.__str__()

    def copy(self) -> "BayesNode":
        """Create a deep copy of the BayesNode"""
        return BayesNode(
            var_name=self.var_name,
            parent_nodes=self.parent_nodes.copy() if self.parent_nodes else None,
            cpd=self.cpd.copy() if self.cpd is not None else None
        )


class BayesNet:
    """
    Representation for a Bayesian Network
    """

    def __init__(self, bn_file: str = "data/bnet"):
        # nodes are indexed by their variable name
        self.nodes, self.queries = BayesNet.parse(bn_file)

    @staticmethod
    def _create_cpd(
        var: str, parent_vars: List[str], parsed_cpd: List[float]
    ) -> pd.DataFrame:
        num_parents = len(parent_vars) if parent_vars else 0
        product_list = [[1, 0]] + [[0, 1]] * num_parents

        cpt_idx = list(itertools.product(*product_list))
        cpt_vals = parsed_cpd + [(1 - v) for v in parsed_cpd]

        idx_names = [var]
        if parent_vars:
            idx_names.extend(parent_vars)

        index = pd.MultiIndex.from_tuples(cpt_idx, names=idx_names)
        cpd_df = pd.DataFrame(data=cpt_vals, index=index, columns=["prob"])

        return cpd_df

    @staticmethod
    def parse(file: str) -> Tuple[Dict[str, BayesNode], List[Dict[str, Any]]]:
        """
        Parses the input file and returns an instance of a BayesNet object
        :param file:
        :return: the BayesNet object
        """
        bn_dict: Dict[str, BayesNode] = {}
        query_list: List[Dict[str, Any]] = []

        with open(file) as fin:
            # read the number of vars involved
            # and the number of queries
            N, M = [int(x) for x in next(fin).split()]

            # read the vars, their parents and the CPD
            for i in range(N):
                line = next(fin).split(";")
                parsed_var = line[0].strip()
                parsed_parent_vars = line[1].split()
                parsed_cpd = [float(v) for v in line[2].split()]

                parent_vars = [bn_dict[v] for v in parsed_parent_vars]
                cpd_df = BayesNet._create_cpd(
                    parsed_var, parsed_parent_vars, parsed_cpd
                )
                bn_dict[parsed_var] = BayesNode(
                    var_name=parsed_var, parent_nodes=parent_vars, cpd=cpd_df
                )

            # read the queries
            for i in range(M):
                queries, conds = next(fin).split("|")

                query_vars = queries.split()
                query_vars_dict = dict(
                    [(q.split("=")[0], q.split("=")[1]) for q in query_vars]
                )

                cond_vars = conds.split()
                cond_vars_dict = dict(
                    [(c.split("=")[0], c.split("=")[1]) for c in cond_vars]
                )

                query_list.append({"query": query_vars_dict, "cond": cond_vars_dict})

            # read the answers
            for i in range(M):
                query_list[i]["answer"] = float(next(fin).strip())

        return bn_dict, query_list

    def get_graph(self) -> nx.DiGraph:
        bn_graph = nx.DiGraph()

        # add nodes with random var attributes that relate the node name to the BayesNode instance
        # in the bayesian network
        for n in self.nodes:
            bn_graph.add_node(n, bn_var=self.nodes[n])

        # add edges
        for n in self.nodes:
            parent_vars = [v.var_name for v in self.nodes[n].parent_nodes]
            if parent_vars:
                for v in parent_vars:
                    bn_graph.add_edge(v, n)

        return bn_graph

    def prob(self, var_name: str, parent_values: List[int] = None) -> float:
        """
        Function that will get the probability value for the case in which the `var_name' variable is True
        (var_name = 1) and the parent values are given by the list `parent values'
        :param var_name: the variable in the bayesian network for which we are determining the conditional property
        :param parent_values: The list of parent values. Is None if var_name has no parent variables.
        :return:
        """
        if parent_values is None:
            parent_values = []

        index_line = tuple([1] + parent_values)

        return self.nodes[var_name].cpd.loc[index_line]["prob"]

    def sample_log_prob(self, sample: Dict[str, int]):
        logprob = 0
        for var_name in self.nodes:
            var_value = sample[var_name]
            parent_vals = None
            if self.nodes[var_name].parent_nodes:
                parent_names = [
                    parent.var_name for parent in self.nodes[var_name].parent_nodes
                ]
                parent_vals = [sample[pname] for pname in parent_names]

            prob = self.prob(var_name, parent_vals)
            if var_value == 0:
                prob = 1 - prob

            logprob += np.log(prob)

        return logprob

    def sample(self) -> Dict[str, int]:
        """
        Sample values for all the variables in the bayesian network and return them as a dictionary
        :return: A dictionary of var_name, value pairs
        """
        values = {}
        remaining_vars = [var_name for var_name in self.nodes]

        while remaining_vars:
            new_vars = []
            for var_name in remaining_vars:
                parent_vars = [p.var_name for p in self.nodes[var_name].parent_nodes]
                if all(p in values for p in parent_vars):
                    parent_vals = [values[p] for p in parent_vars]
                    prob = self.prob(var_name, parent_vals)
                    values[var_name] = int(np.random.sample() <= prob)
                else:
                    new_vars.append(var_name)
            remaining_vars = new_vars
        return values

    def pretty_print_str(self):
        res = "Bayesian Network:\n"
        for var_name in self.nodes:
            res += self.nodes[var_name].pretty_print_str() + "\n"

        return res

    def __str__(self):
        res = "Bayesian Network:\n"
        for var_name in self.nodes:
            res += str(self.nodes[var_name]) + "\n"

        return res

    def __repr__(self):
        return self.__str__()


class BayesNet2:
    """Class representing a Bayesian Network"""
    
    def __init__(self):
        """Initialize an empty Bayesian Network"""
        self.nodes = set()  # Set of nodes
        self.edges = set()  # Set of edges (parent, child)
        self.parents = {}   # Dictionary mapping node -> set of parents
        self.children = {}  # Dictionary mapping node -> set of children
        
    def add_node(self, node: str) -> None:
        """Add a node to the network"""
        self.nodes.add(node)
        if node not in self.parents:
            self.parents[node] = set()
        if node not in self.children:
            self.children[node] = set()
            
    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge from parent to child"""
        # Add nodes if they don't exist
        self.add_node(parent)
        self.add_node(child)
        
        # Add edge
        self.edges.add((parent, child))
        self.parents[child].add(parent)
        self.children[parent].add(child)
        
    def get_parents(self, node: str) -> set:
        """Get parents of a node"""
        return self.parents.get(node, set())
        
    def get_children(self, node: str) -> set:
        """Get children of a node"""
        return self.children.get(node, set())
        
    def get_markov_blanket(self, node: str) -> set:
        """Get the Markov blanket of a node (parents, children, and children's parents)"""
        blanket = set()
        # Add parents
        blanket.update(self.get_parents(node))
        # Add children
        children = self.get_children(node)
        blanket.update(children)
        # Add children's other parents
        for child in children:
            blanket.update(self.get_parents(child))
        # Remove the node itself if it's in the blanket
        blanket.discard(node)
        return blanket
        
    def get_ancestors(self, node: str) -> set:
        """Get all ancestors of a node"""
        ancestors = set()
        to_visit = {node}
        while to_visit:
            current = to_visit.pop()
            parents = self.get_parents(current)
            new_ancestors = parents - ancestors
            ancestors.update(new_ancestors)
            to_visit.update(new_ancestors)
        return ancestors
        
    def get_descendants(self, node: str) -> set:
        """Get all descendants of a node"""
        descendants = set()
        to_visit = {node}
        while to_visit:
            current = to_visit.pop()
            children = self.get_children(current)
            new_descendants = children - descendants
            descendants.update(new_descendants)
            to_visit.update(new_descendants)
        return descendants
        
    def get_all_paths(self, start: str, end: str) -> list:
        """Get all paths from start to end node"""
        def dfs(current: str, target: str, path: list, visited: set) -> list:
            if current == target:
                return [path]
            paths = []
            for child in self.get_children(current):
                if child not in visited:
                    new_visited = visited | {child}
                    new_paths = dfs(child, target, path + [child], new_visited)
                    paths.extend(new_paths)
            return paths
            
        return dfs(start, end, [start], {start})
        
    def get_cliques(self) -> list:
        """Get all maximal cliques in the moral graph"""
        # Create moral graph (add edges between parents and make undirected)
        moral_edges = set()
        for node in self.nodes:
            # Add original edges (undirected)
            for parent in self.get_parents(node):
                moral_edges.add(tuple(sorted([parent, node])))
            # Add edges between parents
            parents = list(self.get_parents(node))
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    moral_edges.add(tuple(sorted([parents[i], parents[j]])))
        
        # Find maximal cliques using Bron-Kerbosch algorithm
        def bronk(r: set, p: set, x: set) -> list:
            if not p and not x:
                return [r]
            cliques = []
            pivot = max(p.union(x), key=lambda v: len(p.intersection(
                n for n in self.nodes if tuple(sorted([v, n])) in moral_edges
            )))
            for v in p - {n for n in p if tuple(sorted([pivot, n])) in moral_edges}:
                neighbors = {n for n in self.nodes if tuple(sorted([v, n])) in moral_edges}
                new_r = r | {v}
                new_p = p.intersection(neighbors)
                new_x = x.intersection(neighbors)
                cliques.extend(bronk(new_r, new_p, new_x))
                p = p - {v}
                x = x | {v}
            return cliques
        
        return bronk(set(), self.nodes, set())


class JunctionTree:
    """
    Place holder class for the JunctionTree algorithm
    """

    def __init__(self, bn: BayesNet):
        self.bn = bn
        self.clique_tree = self._get_clique_tree()
        self._load_factors()
        self._factors_loaded = True

    def _moralize_graph(self, g: nx.DiGraph) -> nx.Graph:
        """
        Moralize a directed graph by:
        1. Adding edges between parents of each node
        2. Converting directed edges to undirected
        """
        # Create undirected copy of the graph
        moral_graph = nx.Graph()

        # Add all nodes and edges from original graph
        moral_graph.add_nodes_from(g.nodes(data=True))
        moral_graph.add_edges_from(g.edges())

        # Add edges between parents
        for node in g.nodes():
            parents = list(g.predecessors(node))
            # Add edges between all pairs of parents
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    moral_graph.add_edge(parents[i], parents[j])

        return moral_graph

    def _triangulate(self, h: nx.Graph) -> nx.Graph:
        """
        Triangulate an undirected graph using a more careful elimination algorithm
        that avoids creating unnecessary fill-in edges.
        """
        # Create a copy to triangulate
        g = h.copy()

        # Get elimination ordering using min-fill heuristic
        ordering = []
        remaining_nodes = list(g.nodes())

        while remaining_nodes:
            # For each remaining node, count how many fill edges would be needed
            fill_counts = {}
            for node in remaining_nodes:
                neighbors = list(g.neighbors(node))
                # Count missing edges between neighbors
                fill_edges = sum(
                    1
                    for i, u in enumerate(neighbors)
                    for v in neighbors[i + 1 :]
                    if not g.has_edge(u, v)
                )
                fill_counts[node] = fill_edges

            # Choose node that requires minimum number of fill edges
            min_fill_node = min(fill_counts.items(), key=lambda x: x[1])[0]

            # Add only necessary fill edges between neighbors
            neighbors = list(g.neighbors(min_fill_node))
            edges_added = 0
            for i, u in enumerate(neighbors):
                for v in neighbors[i + 1 :]:
                    if not g.has_edge(u, v):
                        g.add_edge(u, v)
                        edges_added += 1

            ordering.append(min_fill_node)
            remaining_nodes.remove(min_fill_node)

        return g

    def _create_clique_graph(self, th: nx.Graph) -> nx.Graph:
        """
        Create a clique graph from a triangulated graph.
        """
        # Find all maximal cliques using NetworkX
        cliques = list(nx.find_cliques(th))

        # Create clique graph
        c = nx.Graph()

        # Add nodes (cliques)
        for i, clique in enumerate(cliques):
            c.add_node(i, factor_vars=sorted(clique))

        # Add edges between cliques with non-empty intersection
        for i in range(len(cliques)):
            for j in range(i + 1, len(cliques)):
                intersection = set(cliques[i]) & set(cliques[j])
                if intersection:
                    c.add_edge(i, j)

        return c

    def _extract_clique_tree(self, c: nx.Graph) -> nx.Graph:
        """
        Extract a junction tree from the clique graph by finding a maximum spanning tree
        and properly setting up separator sets.
        """
        # Create edge weights based on the size of the intersection between cliques
        for u, v in c.edges():
            intersection = set(c.nodes[u]["factor_vars"]) & set(
                c.nodes[v]["factor_vars"]
            )
            c[u][v]["weight"] = len(intersection)
            c[u][v]["separator"] = sorted(intersection)

        # Find the maximum spanning tree
        t = nx.maximum_spanning_tree(c, weight="weight")

        # Copy node attributes
        for node in t.nodes():
            t.nodes[node].update(c.nodes[node])

        # Copy edge attributes (especially the separators)
        for u, v in t.edges():
            t[u][v].update(c[u][v])

        return t

    def _get_clique_tree(self) -> nx.Graph:
        """
        Generate the clique tree which is used to propagate "messages" (run belief propagation)
        within the cliques to balance the clique tree
        :return: The CliqueTree as a nx.DiGraph where each node has an attribute called "factor_vars", which
        is the list of random variables within the clique.
        """
        g = self.bn.get_graph()

        # TODO 1: moralize graph g
        #  see https://networkx.org/documentation/stable/_modules/networkx/algorithms/moral.html
        h = self._moralize_graph(g)

        # TODO 2: triangulate h
        th = self._triangulate(h)

        # TODO 3: create clique graph c - find maximal cliques
        #   see https://networkx.org/documentation/stable/reference/algorithms/chordal.html
        c = self._create_clique_graph(th)

        # TODO 4: create clique tree from clique graph c - find Maximum Weight Spanning Tree in c
        #   see https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.mst.maximum_spanning_tree.html#networkx.algorithms.tree.mst.maximum_spanning_tree
        t = self._extract_clique_tree(c)

        return t

    def _load_factors(self) -> None:
        """
        Load initial factors from Bayesian network into clique tree.
        """
        # Initialize all clique potentials to 1
        for node in self.clique_tree.nodes():
            # Create a factor table with all 1s
            vars = self.clique_tree.nodes[node]["factor_vars"]
            index = pd.MultiIndex.from_product(
                [[1, 0] for _ in vars], names=vars  # Note: Changed order to [1, 0]
            )
            self.clique_tree.nodes[node]["potential"] = pd.DataFrame(
                1.0, index=index, columns=["prob"]
            )

        # Assign each BN factor to a clique
        for var, node in self.bn.nodes.items():
            factor = node.to_factor()
            factor_vars = set(factor.vars)

            # Find smallest clique containing all factor variables
            containing_cliques = [
                n
                for n in self.clique_tree.nodes()
                if factor_vars.issubset(set(self.clique_tree.nodes[n]["factor_vars"]))
            ]

            if not containing_cliques:
                raise ValueError(
                    f"No clique found containing factor variables {factor_vars}"
                )

            best_clique = min(
                containing_cliques,
                key=lambda n: len(self.clique_tree.nodes[n]["factor_vars"]),
            )

            try:
                # Multiply factor into clique potential
                self.clique_tree.nodes[best_clique]["potential"] = multiply_factors(
                    self.clique_tree.nodes[best_clique]["potential"], factor.table
                )
                # Normalize to prevent numerical issues
                self.clique_tree.nodes[best_clique]["potential"] = normalize(
                    self.clique_tree.nodes[best_clique]["potential"]
                )
            except Exception as e:
                print(f"Error loading factor for variable {var}: {e}")
                raise

    def _get_junction_tree(self, root_name: str = None) -> nx.DiGraph:
        """
        Set a direction to the edges of the clique tree (which is an nx.Graph) such that the Junction Tree has
        a root. The root node is given by root_name.
        :param root_name: The name of the clique node that is the root of the Junction Tree
        :return: a nx.DiGraph representing the Junction Tree
        """
        if root_name is None or root_name not in self.clique_tree:
            root_name = random.choice(list(self.clique_tree.nodes()))

        t: nx.DiGraph = nx.bfs_tree(self.clique_tree, root_name)
        clique_tree_attrs = deepcopy(dict(self.clique_tree.nodes(data=True)))
        nx.set_node_attributes(t, clique_tree_attrs)

        return t

    def _incorporate_evidence(
        self, jt: nx.DiGraph, evidence: Dict[str, int]
    ) -> nx.DiGraph:
        """
        Incorporate evidence into the junction tree.
        """
        if not evidence:
            return jt

        jt = jt.copy()

        # For each evidence variable
        for var, value in evidence.items():
            # Find smallest clique containing the variable
            best_clique = None
            min_size = float("inf")

            for clique in jt.nodes():
                if "factor_vars" not in jt.nodes[clique]:
                    continue
                clique_vars = set(jt.nodes[clique]["factor_vars"])
                if var in clique_vars and len(clique_vars) < min_size:
                    min_size = len(clique_vars)
                    best_clique = clique

            if best_clique is None:
                continue

            # Update potential
            if "potential" not in jt.nodes[best_clique]:
                continue

            potential = jt.nodes[best_clique]["potential"]
            if var in potential.index.names:
                try:
                    # Create mask for evidence
                    mask = potential.index.get_level_values(var) == value
                    new_potential = potential.copy()
                    new_potential.loc[~mask, "prob"] = 0.0

                    # Normalize if possible
                    total = new_potential["prob"].sum()
                    if total > 0:
                        new_potential["prob"] /= total

                    jt.nodes[best_clique]["potential"] = new_potential
                except Exception as e:
                    print(f"Error incorporating evidence {var}={value}: {e}")

        return jt

    def _run_belief_propagation(self, jt: nx.DiGraph) -> nx.DiGraph:
        """
        Run belief propagation with improved probability handling.
        """

        def compute_message(
            from_node: Any, to_node: Any, messages: Dict
        ) -> pd.DataFrame:
            try:
                if not jt.has_edge(from_node, to_node):
                    return pd.DataFrame()

                separator = jt.edges[from_node, to_node]["separator"]
                result = jt.nodes[from_node]["potential"].copy()

                # Multiply incoming messages
                for neighbor in jt.predecessors(from_node):
                    if neighbor != to_node and (neighbor, from_node) in messages:
                        msg = messages[(neighbor, from_node)]
                        if not msg.empty:
                            result = multiply_factors(result, msg)

                # Eliminate variables
                vars_to_eliminate = [
                    v for v in result.index.names if v not in separator
                ]
                for var in vars_to_eliminate:
                    result = sumout(result, [var])

                return normalize(result)
            except Exception as e:
                print(f"Error in compute_message: {e}")
                return pd.DataFrame()

        try:
            jt = jt.copy()
            messages = {}

            # Store initial potentials
            for node in jt.nodes():
                jt.nodes[node]["initial_potential"] = jt.nodes[node]["potential"].copy()

            # Choose root
            root = max(jt.nodes(), key=lambda n: len(jt.nodes[n]["factor_vars"]))

            # Multiple passes
            for _ in range(2):
                # Reset potentials
                for node in jt.nodes():
                    jt.nodes[node]["potential"] = jt.nodes[node][
                        "initial_potential"
                    ].copy()

                # Collect messages
                for node in nx.dfs_postorder_nodes(jt, root):
                    for parent in jt.predecessors(node):
                        message = compute_message(node, parent, messages)
                        if not message.empty:
                            messages[(node, parent)] = message
                            parent_potential = multiply_factors(
                                jt.nodes[parent]["potential"], message
                            )
                            jt.nodes[parent]["potential"] = normalize(parent_potential)

                # Distribute messages
                for node in nx.dfs_preorder_nodes(jt, root):
                    for child in jt.successors(node):
                        message = compute_message(node, child, messages)
                        if not message.empty:
                            messages[(node, child)] = message
                            child_potential = multiply_factors(
                                jt.nodes[child]["potential"], message
                            )
                            jt.nodes[child]["potential"] = normalize(child_potential)

            return jt

        except Exception as e:
            print(f"Error in belief propagation: {e}")
            return jt

    def _eval_query(self, calibrated_jt: nx.DiGraph, query: Dict[str, int]) -> float:
        """
        Evaluate a query using the calibrated junction tree.
        """
        # Convert string values to integers
        query = {k: int(v) for k, v in query.items()}

        query_vars = set(query.keys())

        # Find smallest clique containing all query variables
        relevant_cliques = []
        for clique in calibrated_jt.nodes():
            clique_vars = set(calibrated_jt.nodes[clique]["factor_vars"])
            if query_vars.issubset(clique_vars):
                relevant_cliques.append(clique)

        if not relevant_cliques:
            raise ValueError("Query variables not found in single clique")

        # Choose smallest containing clique
        best_clique = min(
            relevant_cliques, key=lambda c: len(calibrated_jt.nodes[c]["factor_vars"])
        )

        # Get potential and marginalize
        potential = calibrated_jt.nodes[best_clique]["potential"].copy()

        # Sum out non-query variables
        for var in potential.index.names:
            if var not in query:
                potential = sumout(potential, [var])
                potential = normalize(potential)

        # Get probability for query assignment
        try:
            idx = tuple(query[var] for var in potential.index.names)
            return float(potential.loc[idx]["prob"])
        except KeyError as e:
            print(f"Error accessing index {idx} in potential:")
            print(potential)
            raise

    def run_query(self, query: Dict[str, int], evidence: Dict[str, int]) -> float:
        """
        Run a query on the Bayesian network using the junction tree algorithm.
        """
        # Convert string values to integers
        query = {k: int(v) for k, v in query.items()}
        evidence = {k: int(v) for k, v in evidence.items()}

        # Select root based on query and evidence variables
        all_vars = set(query.keys()) | set(evidence.keys())

        # Find best root clique that contains most query/evidence variables
        max_overlap = -1
        root_name = None
        for clique in self.clique_tree.nodes():
            clique_vars = set(self.clique_tree.nodes[clique]["factor_vars"])
            overlap = len(clique_vars & all_vars)
            if overlap > max_overlap:
                max_overlap = overlap
                root_name = clique

        # Get junction tree
        jt = self._get_junction_tree(root_name)

        # Incorporate evidence
        uncalibrated_jt = self._incorporate_evidence(jt, evidence)

        # Run belief propagation
        calibrated_jt = self._run_belief_propagation(uncalibrated_jt)

        # For conditional probability P(Q|E), we need P(Q,E)/P(E)
        if evidence:
            # Calculate P(Q,E)
            joint_query = {**query, **evidence}
            numerator = self._eval_query(calibrated_jt, joint_query)

            # Calculate P(E)
            denominator = self._eval_query(calibrated_jt, evidence)

            if denominator == 0:
                return 0.0  # Evidence has zero probability

            return numerator / denominator
        else:
            # For queries without evidence, just evaluate P(Q)
            return self._eval_query(calibrated_jt, query)

    def run_queries(self, queries) -> None:
        """
        Run queries.
        :param queries: queries in the original bayesian network
        """
        for query in queries:
            query_prob = self.run_query(query["query"], query["cond"])
            if almost_equal(query_prob, query["answer"]):
                print(
                    "Query %s OK. Answer is %.6f, given result is %.6f"
                    % (str(query), query["answer"], query_prob)
                )
            else:
                print(
                    "Query %s NOT OK. Answer is %.6f, given result is %.6f"
                    % (str(query), query["answer"], query_prob)
                )

if __name__ == "__main__":
    bn = BayesNet(bn_file="data/bn_learning")
    jt = JunctionTree(bn=bn)
    jt.run_queries(bn.queries)

    # get 20 samples from the Bayesian network and write the resulting dict to a file as space separated values
    samples_dict = {var: [] for var in sorted(bn.nodes.keys())}
    for _ in range(20):
        sample = bn.sample()
        for var in sorted(bn.nodes.keys()):
            samples_dict[var].append(sample[var])

    with open("data/samples_exam", "w") as f:
        f.write(" ".join(sorted(bn.nodes.keys())) + "\n")
        for i in range(20):
            f.write(
                " ".join([str(samples_dict[var][i]) for var in sorted(bn.nodes.keys())])
                + "\n"
            )

    bn2 = BayesNet2()
    bn2.add_edge("A", "B")
    bn2.add_edge("A", "C")
    bn2.add_edge("B", "D")
    bn2.add_edge("C", "D")
    print(bn2.get_cliques())

class AirQualityBayesNet:
    """
    Enhanced Bayesian Network structure for the Air Quality dataset.
    
    The network structure models the following key relationships:
    1. Sensor-Measurement Dependencies:
       - CO(GT) → PT08.S1(CO): CO sensor response
       - NMHC(GT) → PT08.S2(NMHC): NMHC sensor response
       - NOx(GT) → PT08.S3(NOx): NOx sensor response
       - NO2(GT) → PT08.S4(NO2): NO2 sensor response
       - O3 measured by PT08.S5(O3)
    
    2. Chemical Interactions:
       - NMHC(GT) → C6H6(GT): NMHC contribution to Benzene formation
       - NOx(GT) → NO2(GT): NOx as precursor to NO2
       - NOx(GT) → PT08.S5(O3): NOx influence on O3 formation
       - CO(GT) → NOx(GT): CO-NOx urban correlation
    
    3. Environmental Influences:
       - Temperature (T) and Relative Humidity (RH) → Absolute Humidity (AH)
       - Temperature (T) → NOx(GT): Temperature affects NOx chemistry
       - Temperature (T) → PT08.S5(O3): Temperature influences O3 formation
       - RH → PT08.S5(O3): Humidity affects O3 measurements
    """
    
    def __init__(self):
        # Define all nodes in the network
        self.nodes = [
            'CO(GT)', 'PT08.S1(CO)', 
            'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
            'NOx(GT)', 'PT08.S3(NOx)', 
            'NO2(GT)', 'PT08.S4(NO2)',
            'PT08.S5(O3)',
            'T', 'RH', 'AH'
        ]
        
        # Define edges (parent → child relationships)
        self.edges = [
            # 1. Sensor-Measurement Dependencies
            ('CO(GT)', 'PT08.S1(CO)'),
            ('NMHC(GT)', 'PT08.S2(NMHC)'),
            ('NOx(GT)', 'PT08.S3(NOx)'),
            ('NO2(GT)', 'PT08.S4(NO2)'),
            
            # 2. Chemical Interactions
            ('NMHC(GT)', 'C6H6(GT)'),      # NMHC → Benzene
            ('NOx(GT)', 'NO2(GT)'),        # NOx → NO2
            ('NOx(GT)', 'PT08.S5(O3)'),    # NOx influence on O3
            ('CO(GT)', 'NOx(GT)'),         # CO-NOx correlation
            
            # 3. Environmental Influences
            ('T', 'AH'),                   # Temperature → Absolute Humidity
            ('RH', 'AH'),                  # Relative Humidity → Absolute Humidity
            ('T', 'NOx(GT)'),              # Temperature effect on NOx
            ('T', 'PT08.S5(O3)'),          # Temperature effect on O3
            ('RH', 'PT08.S5(O3)')          # Humidity effect on O3
        ]
        
        # Define cliques for efficient inference
        self.cliques = [
            # CO and NOx related measurements
            {'CO(GT)', 'PT08.S1(CO)', 'NOx(GT)'},
            
            # NMHC and Benzene measurements
            {'NMHC(GT)', 'PT08.S2(NMHC)', 'C6H6(GT)'},
            
            # NOx, NO2, and O3 measurements
            {'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)'},
            
            # Environmental measurements and their effects
            {'T', 'RH', 'AH', 'PT08.S5(O3)', 'NOx(GT)'}
        ]
    
    def get_markov_blanket(self, node: str) -> set:
        """
        Get the Markov blanket for a given node (parents, children, and children's parents).
        
        The Markov blanket of a node contains all the variables that shield the node from 
        the rest of the network. This includes:
        - The node's parents
        - The node's children
        - The parents of the node's children (spouses)
        
        Args:
            node: Name of the node
            
        Returns:
            set: Markov blanket nodes
            
        Raises:
            ValueError: If node is not found in the network
        """
        if node not in self.nodes:
            raise ValueError(f"Node {node} not found in the network")
        
        markov_blanket = set()
        
        # Add parents
        parents = {parent for parent, child in self.edges if child == node}
        markov_blanket.update(parents)
        
        # Add children
        children = {child for parent, child in self.edges if parent == node}
        markov_blanket.update(children)
        
        # Add children's other parents (spouses)
        for child in children:
            child_parents = {parent for parent, c in self.edges if c == child}
            markov_blanket.update(child_parents)
        
        # Remove the node itself if it's in the blanket
        markov_blanket.discard(node)
        
        return markov_blanket
    
    def get_parents(self, node: str) -> set:
        """
        Get parents of a node.
        
        Args:
            node: Name of the node
            
        Returns:
            set: Parent nodes
        """
        return {parent for parent, child in self.edges if child == node}
    
    def get_children(self, node: str) -> set:
        """
        Get children of a node.
        
        Args:
            node: Name of the node
            
        Returns:
            set: Child nodes
        """
        return {child for parent, child in self.edges if parent == node}
    
    def get_clique_for_node(self, node: str) -> set:
        """
        Get the clique containing the given node.
        
        Args:
            node: Name of the node
            
        Returns:
            set: Clique containing the node
        """
        for clique in self.cliques:
            if node in clique:
                return clique
        return set()
    
    def get_all_cliques(self) -> list:
        """
        Get all cliques in the network.
        
        Returns:
            list: List of all cliques
        """
        return self.cliques
    
    def get_node_relationships(self, node: str) -> dict:
        """
        Get detailed relationships for a node.
        
        Args:
            node: Name of the node
            
        Returns:
            dict: Dictionary containing:
                - parents: Parent nodes
                - children: Child nodes
                - markov_blanket: Markov blanket nodes
                - clique: Clique containing the node
        """
        return {
            'node': node,
            'parents': self.get_parents(node),
            'children': self.get_children(node),
            'markov_blanket': self.get_markov_blanket(node),
            'clique': self.get_clique_for_node(node)
        }
    
    def write_to_file(self, output_file: str) -> None:
        """
        Write the network structure to a file in the format expected by BayesNet.
        
        Format:
        <num_nodes> <num_queries>
        <node>; <parent1> <parent2> ...; <cpt_values>
        
        Args:
            output_file: Path to write the network structure
        """
        try:
            # Create a mapping for special characters
            name_mapping = {}
            for node in self.nodes:
                if node == 'T':
                    safe_name = 'Temp'
                elif '(' in node and ')' in node:
                    # Keep the original name but replace special characters
                    safe_name = node.replace('(', '_').replace(')', '').replace('.', '_')
                else:
                    # Simple names like RH, AH stay as is
                    safe_name = node
                name_mapping[node] = safe_name
            
            # Get nodes in topological order to ensure parents are written first
            ordered_nodes = []
            visited = set()
            temp_visited = set()
            
            def visit(node):
                if node in temp_visited:
                    raise ValueError(f"Cycle detected in network at node {node}")
                if node in visited:
                    return
                temp_visited.add(node)
                
                # Visit all parents first
                parents = self.get_parents(node)
                for parent in sorted(parents):  # Sort for deterministic order
                    visit(parent)
                    
                temp_visited.remove(node)
                visited.add(node)
                ordered_nodes.append(node)
            
            # Visit all nodes to build topological order
            for node in sorted(self.nodes):  # Sort for deterministic order
                if node not in visited:
                    visit(node)
            
            # Print mapping for debugging
            print("\nNode name mapping:")
            for orig, safe in name_mapping.items():
                print(f"{orig} -> {safe}")
            
            with open(output_file, 'w') as f:
                # Write header: number of nodes and queries (0 for now)
                f.write(f"{len(self.nodes)} 0\n")
                
                # Process nodes in topological order
                for node in ordered_nodes:
                    # Get parents for this node
                    parents = self.get_parents(node)
                    # Use mapped names for parents
                    parent_str = " ".join(name_mapping[p] for p in sorted(parents)) if parents else ""
                    
                    # Calculate number of probabilities needed
                    num_parents = len(parents)
                    num_probs = 2 ** num_parents  # For each parent combination
                    
                    # Initialize with uniform probabilities
                    probs = [0.5] * num_probs
                    prob_str = " ".join(map(str, probs))
                    
                    # Write node definition using mapped name
                    f.write(f"{name_mapping[node]}; {parent_str}; {prob_str}\n")
                    
        except Exception as e:
            raise IOError(f"Error writing network structure to file: {str(e)}")
    
    def get_node_ordering(self) -> list:
        """
        Get a topological ordering of nodes based on dependencies.
        
        Returns:
            list: Nodes in topological order
        """
        # Create adjacency list representation
        adj_list = {node: set() for node in self.nodes}
        for parent, child in self.edges:
            adj_list[parent].add(child)
        
        # Track visited nodes and ordering
        visited = set()
        temp_visited = set()
        ordering = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError("Network contains a cycle")
            if node in visited:
                return
            
            temp_visited.add(node)
            for child in adj_list[node]:
                visit(child)
            temp_visited.remove(node)
            visited.add(node)
            ordering.insert(0, node)
        
        # Visit each node
        for node in self.nodes:
            if node not in visited:
                visit(node)
        
        return ordering
