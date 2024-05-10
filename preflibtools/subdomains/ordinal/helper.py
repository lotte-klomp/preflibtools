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