from preflibtools.instances import OrdinalInstance

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
    """Get the list of voters from the instance.

    Args:
        instance (OrdinalInstance): the instance

    Returns:
        list: the list of voters
    """
    voters = []

    for (pref, _) in instance.flatten_strict():
        voters.append(pref)

    return voters