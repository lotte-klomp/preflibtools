from preflibtools.instances import OrdinalInstance
from helper import get_alternatives, restrict_preferences
import random
import networkx as nx
import matplotlib.pyplot as plt

def get_bottom_alts(instance: list):
    """Get the set of last ranked alternatives in the instance.

    Args:
        instance (OrdinalInstance): the instance

    Returns:
        set: the set of last ranked alternatives
    """
    last_ranked = set()

    for voter in instance:
        last_ranked.add(voter[-1])

    return last_ranked

def get_B(instance: OrdinalInstance, alt_set: list, alternative: int):
    # B_vals = []
    B_a = set()

    # TODO: Check if B_vals workaround works
    instance = restrict_preferences(instance, alt_set)

    for i in instance:
        # Check if top(i) = a
        if alternative == i[0]:
            # B(i, a) = {second(i)}
            # B_vals.append(i[1])
            B_a.add(i[1])
        else:
            # get all alternatives before a
            # B_vals.append(i[:i.index(alternative)])
            if len(B_a) == 0:
                B_a = set(i[:i.index(alternative)])
            else:
                B_a = B_a.intersection(set(i[:i.index(alternative)]))

    return B_a

def is_SP_on_Tree(instance: OrdinalInstance):
    """Function to detect whether or not the instance is single-peaked on a
    tree. Returns True and the tree on the set of alternatives such that the
    instance is single-peaked on this tree, if one exists. Based on Trick's
    1989 algorithm.

    Args:
        instance (OrdinalInstance): the preference data
    """
    C_set = get_alternatives(instance)
    Tree = nx.Graph()

    while len(C_set) >= 3:
        L_set = get_bottom_alts(restrict_preferences(instance, C_set))

        for a in L_set:
            B_a = get_B(instance, C_set, a)
            if len(B_a) != 0:
                # Choose an arbitrary element from B(a)
                # TODO: efficiency gain ? unnecessary if small list ?
                b = random.choice(list(B_a))

                # add edge (a, b) to T
                Tree.add_edge(a, b)

            else:
                # instance is not single-peaked on any tree
                return False, None

        C_set = C_set - L_set

    if len(C_set) == 2:
        # add edge (a, b) to T
        a, b = C_set
        Tree.add_edge(a, b)

    return True, Tree



instance = OrdinalInstance()
instance.parse_file("preflibtools/preflibtools/subdomains/ordinal/SP_tree.soc")

_, Tree = is_SP_on_Tree(instance)

# Draw the tree
subax1 = plt.subplot(121)
nx.draw(Tree, with_labels=True, font_weight='bold')
plt.show()