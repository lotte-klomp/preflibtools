from preflibtools.instances import OrdinalInstance

def get_alternatives(instance: OrdinalInstance):
    """Get the set of alternatives from the instance.

    Args:
        instance (OrdinalInstance): the instance

    Returns:
        (set): the set of alternatives
    """
    alternatives = set()
    for alt,  _ in instance.alternatives_name.items():
        alternatives.add(alt)

    return alternatives

def get_voters(instance: OrdinalInstance):
    """Get the list of voters from the instance.

    Args:
        instance (OrdinalInstance): the instance

    Returns:
        (list): the list of voters
    """
    voters = []

    for (pref, _) in instance.flatten_strict():
        voters.append(pref)

    return voters

def restrict_preferences(instance: OrdinalInstance, alternatives_set: set):
    """Restrict the preferences of the voters to elements in alternatives_set.

    Returns:
        (list): the restricted preferences
    """
    flattened = instance.flatten_strict()

    preferences = []
    for (pref, _) in flattened:
        pref = [c for c in pref if c in alternatives_set]
        preferences.append(pref)

    return preferences

def get_num_equal_voters(instance: OrdinalInstance, vote: list):
    """Return the number of voters with the same preference order as the given
    vote.

    Args:
        instance (OrdinalInstance): the preference profile.
        vote (list): the preference order to compare.

    Returns:
        (int): the number of voters with the same preference order as the given
               vote.
    """
    vote_map = instance.vote_map()
    vote = tuple([(v, ) for v in vote])
    return vote_map[vote]

def write_instance(instance: OrdinalInstance, num_voters: int,
                   num_alternatives: int, num_unique_orders: int, name: str):
    with open(f"{name}.soc", "w") as file:
        file.write("# FILE NAME: " + name + "\n")
        file.write("# NUMBER ALTERNATIVES: " + str(num_alternatives) + "\n")
        file.write("# NUMBER VOTERS: " + str(num_voters) + "\n")
        file.write("# NUMBER UNIQUE ORDERS: " + str(num_unique_orders) + "\n")

        for i in range(num_alternatives):
            file.write(f"# ALTERNATIVE NAME {str(i + 1)}: {chr(i + 97)}" + "\n")

        # Iterate over the votes and their multiplicities
        for vote, multiplicity in instance.multiplicity.items():
            file.write(f"{multiplicity}: " + ", ".join([str(x[0]) for x in vote]) + "\n")

        file.close()
