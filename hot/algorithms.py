
from itertools import combinations


def tighten(bounding_topo, i, threshold_fn, nonzero_fn):
    """
    Algorithm 2 from Smith 2022.
    Tightens a bounding topology using common cumulants of a given order.

    Parameters
    ----------
    bounding_topo : Collection[Set]
        Bounding topology to be tightened.
    i : int
        Common cumulant order.
    threshold_fn : Callable
        threshold_fn(len(B), i) is the min number of size-i subsets of B with a nonzero
        common cumulant in order for B to be accepted.
    nonzero_fn : Callable
        nonzero_fn(P) indicates whether P has a nonzero ith-order common cumulant.

    Returns
    -------
    tight_bounding_topo : List[Set]
        Tightened bounding topology.
    """

    # Sections of code are commented with line numbers from Algorithm 2 in the paper.

    # Line 1
    Pi = {P for B in bounding_topo for P in combinations(B, i)}
    accepted_Pi = {frozenset(P) for P in Pi if nonzero_fn(P)}   # \mathcal P in the paper
    processed = set()                                           # \mathcal X in the paper
    tight_bounding_topo = []                                    # \mathcal B' in the paper

    # Line 2
    B_stack = list(bounding_topo)                               # \mathcal B in the paper
    while len(B_stack) > 0:

        # Line 3
        B = B_stack.pop(0)
        processed.add(B)

        # Lines 4-5
        if len(B) < i:
            tight_bounding_topo.append(B)
            continue
        n_accepted_subsets = len({P for P in accepted_Pi if P <= B})
        if n_accepted_subsets >= threshold_fn(len(B), i):
            tight_bounding_topo.append(B)

        # Lines 6-10
        else:
            for p in B:
                B_sub = B - {p}
                if B_sub in processed:
                    continue
                is_maximal = True
                for M_acc in tight_bounding_topo + B_stack:
                    if B_sub <= M_acc:
                        is_maximal = False
                        break
                if is_maximal:
                    B_stack.insert(0, B_sub)

    return tight_bounding_topo


def bounding_topology(initial_bounding_topo, i0, imax, threshold_fn, nonzero_fn):
    """
    Algorithm 3 from Smith 2022.
    Tightens an initial guess for bounding topology using common cumulants
    up to a given order.

    Parameters
    ----------
    initial_bounding_topo : Collection[Set]
        Initial guess for the bounding topology
    i0 : int
        Initial common cumulant order.
    imax : int
        Final common cumulant order.
    threshold_fn : Callable
        threshold_fn(len(B), i) is the min number of size-i subsets of B with a nonzero
        common cumulant in order for B to be accepted.
    nonzero_fn : Callable
        nonzero_fn(P) indicates whether P has a nonzero common cumulant of order len(P).

    Returns
    -------
    bounding_topo : List[Set]
        Estimate of the bounding topology.
    """
    bounding_topo = initial_bounding_topo
    for i in range(i0, imax + 1):
        bounding_topo = tighten(bounding_topo, i, threshold_fn, nonzero_fn)
    return bounding_topo
