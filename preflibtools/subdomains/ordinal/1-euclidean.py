"""
1-Euclidean domain must be single-peaked and single-crossing.
"""
from preflibtools.instances import OrdinalInstance
from preflibtools.properties.singlepeakedness import *
import itertools
from mip import *


# for plotting, delete later
import matplotlib.pyplot as plt
import numpy as np
from single_crossing import is_single_crossing

def get_alternatives(instance):
    """
    Get the set of alternatives from the instance.
    """
    alternatives = set()
    for alt,  _ in instance.alternatives_name.items():
        alternatives.add(alt)

    return alternatives

def append_to_axis(axis: list, a, b):
    """
    Append the alternatives a, b to the axis such that a is to the left of b.
    """
    if a not in axis:
        axis.append(a)

    idx_a = axis.index(a)

    if b in axis:
        idx_b = axis.index(b)
        if idx_b < idx_a:
            axis.remove(b)
            axis.append(b)
    else:
        axis.append(b)

    return axis

def restrict_preferences(instance, C_set_plus):
    """
    Restrict the preferences of the voters to elements in C_set_plus.
    """
    flattened = instance.flatten_strict()

    preferences = []
    for (pref, _) in flattened:
        pref = [c for c in pref if c in C_set_plus]
        preferences.append(pref)

    return preferences

def solve_LP(preferences, axis):
    # create pairs (a, b) such that a is to the left of b in the axis
    pairs = [(a, b) for a, b in itertools.permutations(axis, 2)
             if axis.index(a) < axis.index(b)]

    n = len(preferences)
    m = len(axis)

    # TODO: check if the total number of inequalities is correct
    total = int((n + 1) * m * (m - 1) / 2)

    model = Model()

    # add variables for the voters and alternatives
    vars = [model.add_var(var_type=CONTINUOUS, name=f"voter_{i}") for i in range(n)]
    vars += [model.add_var(var_type=CONTINUOUS, name=f"alternative_{i}") for i in range(m)]

    for pair in pairs:
        # add axis constraints
        model += vars[n + pair[0] - 1] + 1 <= vars[n + pair[1] - 1]

        # add voter constraints
        for i in range(n):
            # if voter prefers a to b
            if preferences[i].index(pair[0]) < preferences[i].index(pair[1]):
                model += vars[i] + 1 <= (vars[n + pair[0] - 1] + vars[n + pair[1] - 1]) / 2
            else:
                model += vars[i] >= (vars[n + pair[1] - 1] + vars[n + pair[0] - 1]) / 2 + 1

    model.objective = minimize(xsum(vars))

    # suppress log
    model.verbose = 0

    status = model.optimize()

    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:

        alternatives = dict()
        for i in range(m):
            alternatives[axis[i]] = model.vars[n + i].x

        voters = []
        for i in range(n):
            voters.append(model.vars[i].x)

        return status, voters, alternatives

    return None, None, None

def plot_results(mappings, n, m):
    """
    Plot the results of the linear program as a scatter plot with 1 dimension.
    """
    fig, ax = plt.subplots()

    # set labels for the alternatives and voters
    alternative_labels = [str(i) for i in range(1, m + 1)]
    voter_labels = ["v" + str(i) for i in range(1, n + 1)]

    # split the mappings into voters and alternatives
    voters = [mappings[i] for i in range(n)]
    alternatives = [mappings[i] for i in range(n, n + m)]

    # plot the voters
    ax.scatter(voters, [0] * len(voters), color='blue')

    for i, txt in enumerate(voter_labels):
        ax.annotate(txt, (voters[i], 0))

    # plot the alternatives
    ax.scatter(alternatives, [0] * len(alternatives), color='red')

    for i, txt in enumerate(alternative_labels):
        ax.annotate(txt, (alternatives[i], 0))

    # set the labels
    ax.set_title('1-Euclidean Domain')

    plt.show()

def gen_sets(v_1, C_set_plus, C_set_minus):
    f, g = dict(), dict()
    ind = 0

    # if C_set_minus is empty
    if not C_set_minus:
        return [C_set_plus], [], 1


    while C_set_minus and C_set_plus:
        tmp = None
        while C_set_plus:
            a = C_set_plus.pop()

            for b in C_set_minus:
                if v_1.index(a) > v_1.index(b):
                    if tmp is None:
                        ind += 1
                    tmp = b

                    if not ind in f:
                        f[ind] = set()
                    f[ind].add(a)
                else:
                    if not ind in f:
                        f[ind] = set()
                    f[ind].add(a)


        if tmp is not None:
            C_set_minus.remove(tmp)
            if not ind in g:
                g[ind] = set()
            g[ind].add(tmp)

        ind += 1
        g[ind] = C_set_minus

    return list(f.values()), list(g.values()), len(f)

def is_Euclidean(instance):
    is_SC, _ = is_single_crossing(instance)

    # veryify that E is single-crossing
    if is_SC:
        # get the first voter
        v_1 = list(instance.flatten_strict()[0][0])
        # get the top of the first voter
        c_minus = v_1[0]

        # get the last voter
        v_n = list(instance.flatten_strict()[-1][0])
        # get the top of the last voter
        c_plus = v_n[0]

        C_set = get_alternatives(instance)


        # TODO: more efficient way to get the number of alternatives
        # number of alternatives
        m = len(C_set)

        # number of voters
        n = instance.num_voters

        gamma = dict()

        # walk through the alternatives
        # colour the alternatives:
        #   - red: if c in C_M (0)
        #   - blue: if c in C_R (1)
        #   - green: if c in C_L (2)
        #   - grey: else (3)
        for c in C_set:
            if ((v_1.index(c) < v_1.index(c_plus)
                and v_n.index(c) < v_n.index(c_minus))
                or (c in set([c_minus, c_plus]))):
                # put c in C_M
                gamma[c] = 0
            else:
                # put in else
                gamma[c] = 3

        # walk through the alternatives in pairs
        # TODO: check if the order of the pairs is important
        for a, b in itertools.permutations(C_set, 2):
            if (v_1.index(a) < v_1.index(b)
                and v_n.index(b) < v_n.index(a)):
                if gamma[a] == 1 or gamma[b] == 2:
                    # Colouring stage cannot be completed
                    return None
                if gamma[a] == 3:
                    gamma[a] = 2
                if gamma[b] == 3:
                    gamma[b] = 1

        C_set_plus = set([c for c in C_set if gamma[c] != 3])
        C_set_minus = C_set - C_set_plus

        axis = list()

        # walk through the alternatives in C_set_plus in pairs
        for a, b in itertools.permutations(C_set_plus, 2):
            if ((gamma[a] == 2 and gamma[b] == 0)
                or (gamma[a] == 0 and gamma[b] == 1)
                or (gamma[a] == 2 and gamma[b] == 1)):
                axis = append_to_axis(axis, a, b)

            if ((gamma[a] == 0 and gamma[b] == 0
                 and v_1.index(a) < v_1.index(b))
                or (gamma[a] == 1 and gamma[b] == 1
                    and v_1.index(a) < v_1.index(b))
                or (gamma[a] == 2 and gamma[b] == 2
                    and v_n.index(b) < v_n.index(a))):
                axis = append_to_axis(axis, a, b)

        # restrict the preferences of the voters to elements in C_set_plus
        preferences = restrict_preferences(instance, C_set_plus)

        # TODO: return voters as dict for mapping?
        # TODO: rename alternatives to x for consistency
        results, voters, alternatives = solve_LP(preferences, axis)

        # check if the LP is feasible
        if (results is not None):
            # insert grey alternatives in the axis
            # partition the grey and non-grey alternatives into groups
            # according to the order of their appearance in the preference of
            # the first voter.
            f, g, k = gen_sets(v_1, C_set_plus, C_set_minus)

            # TODO: check if tmp construction is ever needed since mapping will only be constructed for non-grey value

            # union V and F1
            tmp = voters + [alternatives[i] for i in f[0]]


            # TODO: check if index or value is needed
            x_l = min(tmp)
            x_r = max(tmp)

            tmp = voters + [alternatives[i] for i in C_set_plus]

            # TODO: see previous comment, delta currently as value
            delta = max([abs(x - y) for x, y in itertools.permutations(tmp, 2)])

            y = dict()
            # add mapping of voters
            # index: 1 to n
            for i in range(n):
                y[i] = voters[i]

            # add F1 mapping
            for i in range(len(f[0])):
                # pop add method for efficiency
                tmp = f[0].pop()
                f[0].add(tmp)
                y[tmp + n - 1] = alternatives[tmp]

            # add mapping of grey alternatives in G1
            if g != []:
                for i in range(len(g[0])):
                    tmp = g[0].pop()
                    g[0].add(tmp)
                    y[tmp + n - 1] = x_r + 6 * delta + (i / m) * delta


            for i in range(1, k):
                for c in f[i]:
                    if alternatives[c] < x_l:
                        y[c + n - 1] = alternatives[c] - ((i + 1)**2) * delta
                    if alternatives[c] > x_r:
                        y[c + n - 1] = alternatives[c] + ((i + 1)**2) * delta

                for l in range(len(g[i])):
                    y[g[i].pop() + n - 1] = x_r + ((i + 1)**2) * delta + 2 * delta + l / m * delta

            plot_results(y, n, m)

            return y

    return None


instance = OrdinalInstance()
# instance.parse_file("./test_profiles/1-EU/SP_SC_EU.soc")
# instance.parse_file("./test_profiles/1-EU/SP_SC_not_EU.soc")
# instance.parse_file("./test_profiles/1-EU/EU_test_1.soc")
instance.parse_file("/home/lotte/Documents/Thesis/preflibtools/preflibtools/subdomains/ordinal/EU_test_1.soc")


is_Euclidean(instance)

