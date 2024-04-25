"""
Moldule that provides functions to check if a given preference profile is
single-crossing.
"""
from preflibtools.instances import OrdinalInstance
from helper import get_voters
import numpy as np
import itertools

def kendall_tau_distance(v1: list, v2: list):
    """Computes the Kendall-Tau distance between two voters.

    Args:
        v_1 (list): first ranking
        v_2 (list): second ranking

    Returns:
        int: Kendall-Tau distance between v_1 and v_2
    """
    l = len(v1)

    if int(0) in v1:
        pairs = itertools.combinations(range(l), 2)
    else:
        pairs = itertools.combinations(range(1, l + 1), 2)

    v = 0
    for a, b in pairs:
        if (v1.index(a) - v1.index(b)) * (v2.index(a) - v2.index(b)) < 0:
            v += 1
    return v


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

