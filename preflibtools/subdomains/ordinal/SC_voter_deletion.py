from preflibtools.instances import OrdinalInstance
import networkx as nx
from itertools import product, combinations
from single_crossing import is_single_crossing
from helper import get_voters, get_num_equal_voters, write_instance
import matplotlib.pyplot as plt

def get_conflict_pairs(vote_1: list, vote_2: list):
    """Get the conflict pairs between two votes.

    Args:
        vote_1 (list): the first vote.
        vote_2 (list): the second vote.

    Returns:
        (list): the conflict pairs (a, b).
    """
    conflict_pairs = []

    for a, b in combinations(vote_1, 2):
        if vote_1.index(a) < vote_1.index(b) and vote_2.index(a) > vote_2.index(b):
            conflict_pairs.append((a, b))

    return conflict_pairs

def voter_deletion(instance: OrdinalInstance):
    nr_orders = instance.num_unique_orders
    voters = get_voters(instance)
    G = nx.DiGraph()

    # get permutations z, i in [1, nr_voters] with order mattering
    combinations = product(range(nr_orders), range(nr_orders))
    for z, i in combinations:
        # construct vertex u_z_i to represent preference order i in a linear
        # order starting from z
        G.add_node((z, i))

    # TODO: combine this with the previous loop?
    for z, i in product(range(nr_orders), range(nr_orders)):
        # add edge from u_z_i to u_z_j where j != i if the set of conflict pairs
        # of u_z_i is a subset of the set of conflict pairs of u_z_j
        u_z_i = get_conflict_pairs(list(voters[z]), list(voters[i]))

        for j in range(nr_orders):
            if i != j:
                u_z_j = get_conflict_pairs(list(voters[z]), list(voters[j]))

                if set(u_z_i).issubset(set(u_z_j)):
                    G.add_edge((z, i), (z, j),
                            weight=get_num_equal_voters(instance,
                                                        list(voters[j])))

    # add root vertex
    G.add_node('root')

    # add edges from root to all vertices u_z_z
    for i in range(nr_orders):
        G.add_edge('root', (i, i), weight=get_num_equal_voters(instance,
                                                                list(voters[i])))

    # find the maximum weight path in the graph
    path = nx.dag_longest_path(G, weight='weight')

    # remove the root from the path
    path = path[1:]

    # TODO: remove G after testing
    return path, G


def reduce_instance_to_SC(instance: OrdinalInstance, path: list, write=False):
    """Reduce the instance based on the path.

    Args:
        instance (OrdinalInstance): the preference profile.
        path (list): the path to reduce the instance on.

    Returns:
        OrdinalInstance: the reduced preference profile.
    """
    # get the voters in the path
    voters = get_voters(instance)
    path_voters = [voters[z] for _, z in path]

    # convert vote to tuple of tuples
    path_voters = [tuple([(v, ) for v in vote]) for vote in path_voters]

    # reduce the instance to the voters in the path
    instance.num_unique_orders = len(path_voters)
    instance.multiplicity = {vote: instance.multiplicity[vote]
                             for vote in path_voters}
    instance.orders = path_voters
    instance.preferences = path_voters.copy()

    if write:
        name = instance.file_name + "_reduced_to_SC_voter_deletion"
        write_instance(instance, instance.num_voters, instance.num_alternatives,
                       instance.num_unique_orders, name)

    return instance


instance = OrdinalInstance()
instance.parse_file("preflibtools/preflibtools/subdomains/ordinal/SC_close.soc")
# instance.parse_file("PrefLib/00004-00000001.soc")

path, G = voter_deletion(instance)

reduce_instance_to_SC(instance, path, write=True)

print(is_single_crossing(instance))

# Draw the tree
labels = nx.get_edge_attributes(G,'weight')
pos = nx.spring_layout(G, seed=1)
nx.draw_networkx(G, with_labels=True, pos=pos)
nx.draw_networkx_edge_labels(G, edge_labels=labels, pos=pos)

ax = plt.gca()
ax.margins(0.05)
plt.tight_layout()
plt.axis('off')
plt.show()