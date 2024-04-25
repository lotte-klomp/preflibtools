"""
Moldule that provides functions to check if a given preference profile is
single-crossing.
"""
from preflibtools.instances import OrdinalInstance
import numpy as np
import itertools

def kendall_tau_distance(order_a, order_b):
    if int(0) in order_a:
        pairs = itertools.combinations(range(0, len(order_a)), 2)
    else:
        pairs = itertools.combinations(range(1, len(order_a) + 1), 2)

    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    return distance

def get_alternatives(instance: OrdinalInstance):
    """Get the set of alternatives from the instance.

    Args:
        instance (OrdinalInstance): the instance

    Returns:
        set: the set of alternatives
    """
    alternatives = set()
    for alt,  _ in instance.alternatives_name.items():
        alternatives.add(alt)

    return alternatives

def get_voters(instance: OrdinalInstance):
    voters = []

    for (pref, _) in instance.flatten_strict():
        voters.append(pref)

    return voters

def Kendall_Tau(r_1: list, r_2: list):
    """Computes the Kendall-Tau distance between two rankings
    in O(n sqrt(log n)).

    Args:
        r_1 (list): first ranking
        r_2 (list): second ranking
        alternatives (set): set of alternatives

    Returns:
        int: Kendall-Tau distance between r_1 and r_2
    """

    l = len(r_1)

    # total number of pairs
    total_pairs = l * (l - 1) / 2

    i, v = 0, 0

    for i in range(l):
        for j in range(i + 1, l):
            a = (r_1[i] < r_1[j] and r_2[i] > r_2[j])
            b = (r_1[i] > r_1[j] and r_2[i] < r_2[j])

            if a or b:
                v += 1

    return min(v, int(total_pairs - v))

def is_single_crossing_order(profile: list):
    """Check if the given instance is single-crossing on the given order. Based
    on Algorithm 2 from "Preference Restrictions in Computational Social Choice:
    A Survey" (p. 54).

    Args:
        profile (list): preference profile consisting of votes in order.

    Returns:
        bool: whether or not the instance is single-crossing in the given order.
    """

    for i in range(1, len(profile) - 1):
        # K_1_i = Kendall_Tau(profile[0], profile[i])
        # K_i_i1 = Kendall_Tau(profile[i], profile[i + 1])
        # K_1_i1 = Kendall_Tau(profile[0], profile[i + 1])
        # TODO: fix Kendall_Tau to use the distance function
        K_1_i = kendall_tau_distance(profile[0], profile[i])
        K_i_i1 = kendall_tau_distance(profile[i], profile[i + 1])
        K_1_i1 = kendall_tau_distance(profile[0], profile[i + 1])

        if (K_1_i + K_i_i1) != K_1_i1:
            return False

    return True

def is_single_crossing(instance: OrdinalInstance):
    """Check if the given instance is single-crossing. Based on Algorithm 4 from
    "Preference Restrictions in Computational Social Choice: A Survey" (p. 55).

    Args:
        instance (OrdinalInstance): preference profile.

    Returns:
        array: voters in a single-crossing order if the instance is
               single-crossing, None otherwise.
    """

    voters = get_voters(instance)

    # v_1 and v_2 will always have different preferences due to multiplicity
    # being recorded in the instance
    v_1 = voters[0]
    v_2 = voters[1]

    # Kendall-Tau distance between the first two voters
    # K_dist = Kendall_Tau(v_1, v_2)
    K_dist = kendall_tau_distance(v_1, v_2)

    # array indexed by voters
    score = np.zeros(len(voters))
    score[1] = K_dist

    for i in range(2, len(voters)):
        # K_dist_1 = Kendall_Tau(v_1, voters[i])
        # K_dist_2 = Kendall_Tau(v_2, voters[i])
        K_dist_1 = kendall_tau_distance(v_1, voters[i])
        K_dist_2 = kendall_tau_distance(v_2, voters[i])

        if K_dist_1 + K_dist_2 == K_dist:
            # i goes in between 1 and 2
            score[i] = K_dist_1
        elif K_dist + K_dist_2 == K_dist_1:
            # i goes after 2
            score[i] = K_dist_1
        elif K_dist_1 + K_dist == K_dist_2:
            # i goes before 1
            score[i] = -K_dist_1
        else:
            # instance is not single-crossing
            return False, None

    # order voters by score
    n = instance.num_voters
    m = instance.num_alternatives

    P_order = []

    if n < m:
        # sort the voters in order of score[i]
        P_order = sorted(voters, key=lambda x: score[voters.index(x)])
    elif n >= m:
        B_arr = [[] for i in range(-m**2, m**2 + 1)]

        for i in range(n):
            B_arr[int(score[i] + m**2)].append(voters[i])

        # line 6: XOR on all the elements of B_arr
        P_order = [elem[0] for elem in B_arr if elem != []]

    # check if the determined order is single-crossing
    if is_single_crossing_order(P_order):
        return True, P_order
    else:
        return False, None

# instance.parse_file("./test_profiles/generated/instance.soc")

# res, _ = is_single_crossing(instance)
# print(res)

# print("KT distance: ", kendall_tau_distance([2, 1, 4, 3], [2, 4, 1, 3]))
# print("KT distance: ", kendall_tau_distance([1, 0, 3, 2], [1, 3, 0, 2]))
# print("KenTau: ", Kendall_Tau([1, 0, 3, 2], [1, 3, 0, 2]))

# print("KT distance: ", kendall_tau_distance([0, 1, 2, 3, 4] , [0, 3, 2, 1, 4]))
# print("KenTau: ", Kendall_Tau([0, 1, 2, 3, 4] , [0, 3, 2, 1, 4]))

# print("KenTau: ", Kendall_Tau([2, 1, 4, 3], [2, 4, 1, 3]))

# # expect 4
# assert(Kendall_Tau([1, 2, 3, 4, 5], [3, 4, 1, 2, 5]) == 4)

# # expect 0
# assert(Kendall_Tau([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 0)

# # expect 10
# assert(Kendall_Tau([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == 10)

# # expect 2
# assert(Kendall_Tau([1, 2, 3, 4], [2, 1, 4, 3]) == 2)

# # expect 5
# assert(Kendall_Tau([2, 4, 1, 3], [4, 1, 3, 2]) == 5)
