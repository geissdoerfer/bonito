from .distributions import inverse_joint_cdf


def bonito(tcharges: tuple, dist_models: tuple, target_probability: float = 0.99):
    """Bonito connection protocol.

    With every new observation, each node updates a model of their charging time distribution.
    They exchange the model of their charging times and select a common connection interval according to the inverse joint cdf.
    If the connection interval is greater than the current charging time of both devices, the encounter is considered a success.

    Args:
        tcharges: tuple of 1-d arrays with charging time observations
        dist_models: tuple of classes of charging time distribution
    Yields:
        resulting connection interval and boolean specifying successful or failed encounter
    """
    dist1 = dist_models[0]()
    dist2 = dist_models[1]()

    for c1, c2 in zip(*tcharges):
        conn_int = inverse_joint_cdf((dist1, dist2), target_probability)
        if (c1 <= conn_int) and (c2 <= conn_int):
            yield conn_int, True
        else:
            yield conn_int, False

        dist1.sgd_update(c1)
        dist2.sgd_update(c2)


def modest(tcharges: tuple):
    """Modest connection protocol.

    Each node keeps track of the maximum observed charging time. They exchange the respective values and use the maximum of
    the two as connection interval. If the connection interval is greater than the current charging time of both devices,
    the encounter is considered a success.

    Args:
        tcharges: tuple of 1-d arrays with charging time observations
    Yields:
        resulting connection interval and boolean specifying successful or failed encounter
    """
    cmax1 = tcharges[0][0]
    cmax2 = tcharges[1][0]

    for c1, c2 in zip(*tcharges):
        conn_int = max(cmax1, cmax2)
        if (c1 <= conn_int) and (c2 <= conn_int):
            yield conn_int, True
        else:
            yield conn_int, False
        cmax1 = max(cmax1, c1)
        cmax2 = max(cmax2, c2)


def greedy(tcharges: tuple, max_offset: float = 680e-6):
    """Greedy protocol.

    Nodes wake up as soon as they have energy. If the difference of the charging time is less than the maximum tolerable offset,
    the encounter is considered a success.

    Args:
        tcharges: tuple of 1-d arrays with charging time observations
        max_offset: maximum tolerable time offset
    Yields:
        resulting connection interval and boolean specifying successful or failed encounter
    """
    for c1, c2 in zip(*tcharges):
        if abs(c1 - c2) <= max_offset:
            yield min(c1, c2), True
        else:
            yield min(c1, c2), False
