
import numpy as np
import networkx as nx
import scipy.special
import scipy.stats
from scipy.sparse import coo_matrix, csr_matrix
import json
import os
import cvxpy as cp
import hashlib
import logging
from tqdm import tqdm
from scipy.stats import ttest_1samp
from itertools import combinations, combinations_with_replacement

from PyMoments import kstat

from hot.constants import *
from hot.utils import bootstrap
from hot.algorithms import bounding_topology


def infer_topology(data_filename, output_directory, order, alphas, powers, thresholds, n_groups, resample_size, max_size, l1_weight, l1_exponent, nnz_threshold, solver_args, seed, verbose=False, **kwargs):
    """

    Parameters
    ----------
    data_filename
    output_directory : str
        Directory where output files from the inference are saved.
    order
    alphas : List[float]
        List of significance levels for the nonzero common cumulant hypothesis test.
        Provide one entry for path set sizes 2, 3, 4, ..., order, so that len(powers) == order - 1.
        Each entry should be in the range (0, 1].
    powers : float or List[float]
        List of estimates for the statistical power of the nonzero common cumulant hypothesis test.
        Provide one entry for path set sizes 3, 4, ..., order, so that len(powers) == order - 2.
        Each entry should be in the range (0, 1].
        If a single float is provided, then this power is used for all path set sizes.
    thresholds : float or List[float]
        List of thresholds.
        Provide one entry for path set sizes 3, 4, ..., order, so that len(thresholds) == order - 2.
        Each entry should be in the range (0, 1].
        If a single float is provided, then this threshold is used for all path set sizes.
    n_groups
    resample_size
    max_size : int
        Largest acceptable size of a non-maximal path set.
    l1_weight : float
    l1_exponent : float
    nnz_threshold : float
    solver_args : str
        JSON object of arguments for the CVXPY solver.
    seed : int
        Random seed.

    Returns
    -------
    """

    # Prepare directory for outputs
    write_outputs = output_directory is not None and len(output_directory) > 0
    if write_outputs:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # Set up logger
    logger_filename = os.path.join(output_directory, LOG_FILENAME)
    logging.basicConfig(filename=logger_filename, level=logging.DEBUG)
    logging.info('Starting topology inference with the following arguments:'.format(data_filename))
    logging.info(f'\tdata_filename: {data_filename}')
    logging.info(f'\toutput_directory: {output_directory}')
    logging.info(f'\torder: {order}')
    logging.info(f'\talphas: {alphas}')
    logging.info(f'\tpowers: {powers}')
    logging.info(f'\tthresholds: {thresholds}')
    logging.info(f'\tn_groups: {n_groups}')
    logging.info(f'\tmax_size: {max_size}')
    logging.info(f'\tl1_weight: {l1_weight}')
    logging.info(f'\tl1_exponent: {l1_exponent}')
    logging.info(f'\tnnz_threshold: {nnz_threshold}')
    logging.info(f'\tsolver args: {solver_args}')
    logging.info(f'\tseed: {seed}')

    # Load and bootstrap path delay data
    path_delays = np.loadtxt(data_filename, delimiter=',')
    if resample_size <= 0:
        resample_size = None
    bootstrapped_delays = bootstrap(path_delays, n_groups, sample_size=resample_size, seed=seed)
    h = hashlib.md5()
    h.update(bootstrapped_delays)
    data_hash = h.hexdigest()
    logging.info(f'Bootstrapped data to shape {bootstrapped_delays.shape} and hash {data_hash}')

    # Stage 1: bound support of f

    # Load pre-computed bounding topology, if the metadata matches
    bounding_topo_path = os.path.join(output_directory, BOUNDING_TOPO_FILENAME)
    bounding_topo = None
    if os.path.exists(bounding_topo_path):
        logging.info(f'A bounding topo file {bounding_topo_path} already exists. Comparing metadata:')
        with open(bounding_topo_path, 'r') as infile:
            payload = json.load(infile)
            use_file = True
            for metadata_var in ['data_hash', 'order', 'alphas', 'powers', 'thresholds']:
                local_val = eval(metadata_var)
                if payload[metadata_var] != local_val:
                    logging.info(f'\t{metadata_var} {payload[metadata_var]} does not match')
                    use_file = False
            if use_file:
                logging.info('Metadata matches; loading bounding topo from file')
                bounding_topo = list(map(frozenset, payload['bounding_topo']))
            else:
                logging.info('Metadata does not match; overwriting file with new bounding topo')

    # Compute bounding topology, if necessary, and save it to disk
    if bounding_topo is None:
        if verbose:
            print('Estimating bounding topology...')
        bounding_topo = estimate_bounding_topology(bootstrapped_delays, order, alphas, powers, thresholds)
        logging.info('Computed new bounding topo')
        if write_outputs:
            payload = {
                'bounding_topo': list(map(list, bounding_topo)),
                'data_hash': data_hash,
                'order': order,
                'alphas': list(alphas),
                'powers': list(powers),
                'thresholds': list(thresholds)}
            with open(bounding_topo_path, 'w') as outfile:
                json.dump(payload, outfile)
            logging.info(f'Wrote new bounding topo to {bounding_topo_path}')
    bounding_topo_hash = hash(frozenset(bounding_topo))
    logging.info(f'Using bounding topology with {len(bounding_topo)} sets and hash {bounding_topo_hash}')

    # Stage 2: bound support of g and estimate common cumulants

    # Load pre-computed data, if the metadata matches
    aux_data_path = os.path.join(output_directory, AUX_DATA_FILENAME)
    Xo, Xu, o_idx, u_idx, n_ancestors, link_candidates = None, None, None, None, None, None
    common_cumulant_ests = None
    if os.path.exists(aux_data_path):
        logging.info(f'An auxiliary data file {aux_data_path} already exists. Comparing metadata:')
        payload = np.load(aux_data_path)
        use_file = True
        for metadata_var in ['bounding_topo_hash', 'order', 'max_size', 'data_hash']:
            local_val = eval(metadata_var)
            if payload[metadata_var] != local_val:
                logging.info(f'\t{metadata_var} {payload[metadata_var]} does not match')
                use_file = False
        if use_file:
            logging.info('Metadata matches; loading auxiliary data from file')
            Xo = csr_matrix(
                (payload['Xo_data'], payload['Xo_indices'], payload['Xo_indptr']),
                shape=payload['Xo_shape'])
            Xu = csr_matrix(
                (payload['Xu_data'], payload['Xu_indices'], payload['Xu_indptr']),
                shape=payload['Xu_shape'])
            o_idx, u_idx, n_ancestors = payload['o_idx'], payload['u_idx'], payload['n_ancestors']
            link_candidates = json.loads(str(payload['link_candidates']))
            link_candidates = list(map(frozenset, link_candidates))
            common_cumulant_ests = payload['common_cumulant_ests']
        else:
            logging.info('Metadata does not match; overwriting file with new auxiliary data')

    # Compute inversion matrix and common cumulants, if necessary, and save them to disk
    if common_cumulant_ests is None:
        Xo, Xu, o_idx, u_idx, n_ancestors, link_candidates = construct_inversion_matrix(bounding_topo, order, max_size)
        common_cumulant_ests = np.ndarray((len(o_idx), n_groups))
        pbar = tqdm(range(len(o_idx)), desc='\tEstimating common cumulants...', disable=(not verbose))
        for i in pbar:
            common_cumulant_ests[i, :], _ = estimate_common_cumulant(
                bootstrapped_delays, link_candidates[o_idx[i]], order=order)
        if write_outputs:
            np.savez(aux_data_path,
                     Xo_indices=Xo.indices, Xo_indptr=Xo.indptr, Xo_data=Xo.data, Xo_shape=Xo.shape,
                     Xu_indices=Xu.indices, Xu_indptr=Xu.indptr, Xu_data=Xu.data, Xu_shape=Xu.shape,
                     o_idx=o_idx, u_idx=u_idx, n_ancestors=n_ancestors,
                     link_candidates=json.dumps(list(map(list, link_candidates))),
                     common_cumulant_ests=common_cumulant_ests,
                     bounding_topo_hash=bounding_topo_hash,
                     order=order,
                     max_size=max_size,
                     data_hash=data_hash)
            logging.info(f'Wrote new auxiliary data to {aux_data_path}')
    logging.info(f'Identified {len(link_candidates)} link candidates with '
                 f'{len(o_idx)} observed and {len(u_idx)} unobserved common cumulants')

    # Stage 3: invesion + lasso optimization + reconstruction of routing matrix

    # Optimization
    fo_hat = np.mean(common_cumulant_ests, axis=1)
    fo_sigma = np.std(common_cumulant_ests, axis=1)
    solver_args = json.loads(solver_args)
    g, fo, fu = solve_optimization(Xo, Xu, fo_hat, fo_sigma, o_idx, u_idx, n_ancestors, l1_weight, l1_exponent, **solver_args)
    logging.info(f'Solved lasso optimization problem')

    # Get routing matrix from zero-nonzero pattern of g
    predicted_links, g_idx = [], 0
    for link_candidate_idx in list(u_idx) + list(o_idx):
        P = link_candidates[link_candidate_idx]
        if np.abs(g[g_idx]) > nnz_threshold:
            predicted_links.append(P)
        g_idx += 1
    logging.info(f'Identified {len(predicted_links)} predicted links')

    # Save results to disk
    optim_results_path = os.path.join(output_directory, OPTIM_RESULTS_FILENAME)
    predicted_links_path = os.path.join(output_directory, PREDICTED_LINKS_FILENAME)
    if write_outputs:
        np.savez(optim_results_path, g=g, fo=fo, fu=fu)
        logging.info(f'Saved optimization results to {optim_results_path}')
        with open(predicted_links_path, 'w') as outfile:
            payload = {
                'predicted_links': list(map(list, predicted_links)),
                'bounding_topo': list(map(list, bounding_topo)),
                'order': order,
                'max_size': max_size,
                'l1_weight': l1_weight,
                'l1_exponent': l1_exponent,
                'solver': solver_args,
                'nnz_threshold': nnz_threshold}
            json.dump(payload, outfile)
        logging.info(f'Saved predicted links to {predicted_links_path}')

    logging.info('Done!')


# SPARSE MOBIUS INFERENCE STAGES


def estimate_bounding_topology(bootstrapped_delays, order, alphas, powers, thresholds):
    """
    Estimate the bounding topology from bootstrapped delay data.
    This method encapsulates Stage 1 of the sparse Mobius inference procedure.

    Parameters
    ----------
    bootstrapped_delays : np.ndarray
        (N, n, n_groups) array of resampled delay data.
    alphas : List[float]
        List of significance levels for the nonzero common cumulant hypothesis test.
        Provide one entry for path set sizes 2, 3, 4, ..., order, so that len(powers) == order - 1.
        Each entry should be in the range (0, 1].
    powers : float or List[float]
        List of estimates for the statistical power of the nonzero common cumulant hypothesis test.
        Provide one entry for path set sizes 3, 4, ..., order, so that len(powers) == order - 2.
        Each entry should be in the range (0, 1].
        If a single float is provided, then this power is used for all path set sizes.
    thresholds : float or List[float]
        List of thresholds.
        Provide one entry for path set sizes 3, 4, ..., order, so that len(thresholds) == order - 2.
        Each entry should be in the range (0, 1].
        If a single float is provided, then this threshold is used for all path set sizes.

    Returns
    -------
    bounding_topo : List[Set]
        Estimate of the bounding topology.
    """

    # Process inputs
    if len(powers) == 1:
        powers = [powers] * (order - 2)
    if len(thresholds) == 1:
        thresholds = [thresholds] * (order - 2)
    n_paths = bootstrapped_delays.shape[1]

    # Initial guess for bounding topology via covariances
    clique_graph = nx.Graph()
    clique_graph.add_nodes_from(range(n_paths))
    for i in range(n_paths):
        for j in range(i+1, n_paths):
            _, p = estimate_common_cumulant(bootstrapped_delays, [i, j])
            if p < alphas[0]:
                clique_graph.add_edge(i, j)
    initial_bounding_topo = [frozenset(clique) for clique in nx.find_cliques(clique_graph)]

    # Refine the bounding topology with Algorithm 3
    threshold_fn = create_threshold_fn(powers, thresholds)
    nonzero_fn = create_nonzero_fn(alphas, bootstrapped_delays)
    bounding_topo = bounding_topology(initial_bounding_topo, 2, order, threshold_fn, nonzero_fn)

    return bounding_topo


def construct_inversion_matrix(bounding_topo, i, max_size):
    """
    Construct the Mobius inversion matrix.
    This method encapsulates Stage 2 of the sparse Mobius inference procedure.

    Parameters
    ----------
    bounding_topo : List[Set]
        Estimate of the bounding topology.
    i : int
        Cumulant order.
    max_size : int
        Largest acceptable size of a non-maximal path set.

    Returns
    -------
    Xo : scipy.sparse.csr_matrix
        Sparse matrix encoding the Mobius transformation from "observed" common cumulants.
    Xu : scipy.sparse.csr_matrix
        Sparse matrix encoding the Mobius transformation from "unobserved" common cumulants.
    o_idx : np.ndarray
        Array of indices for "observed" path sets.
    u_idx : np.ndarray
        Array of indices for "unobserved" path sets.
    n_ancestors : np.ndarray
        Number of other path sets whose exact cumulant depends on the common cumulant of P.
    link_candidates : List[Set]
        List of candidate path sets, ordered by their index.
    """

    # Construct "support graph":
    #   Nodes are the support estimate of bounding_topo
    #   Edges point from subsets that are precisely one element smaller
    queue, processed = set(bounding_topo), set()
    Gf = nx.DiGraph()
    while len(queue) > 0:
        P = queue.pop()
        processed.add(P)
        for p in P:
            Q = frozenset(P - {p})
            Gf.add_edge(Q, P)
            if len(Q) > 1 and Q not in queue and Q not in processed:
                queue.add(Q)

    # Link candidates are path sets that are either maximal or smaller than the size cutoff.
    # Entries of f(P) and g(P) are only computed for candidates P.
    link_candidates = [P for P in Gf.nodes if
                       Gf.out_degree(P) == 0 or len(P) <= max_size]
    link_candidate_indices = {P: idx for idx, P in enumerate(link_candidates)}

    # Compute the sparse Mobius inversion matrix between link candidates.
    # Matrix entries are computed according to Lemma 9 in the paper.
    o_idx, u_idx = [], []                   # Indices for observed and unobserved common cumulants
    x_rows, x_cols, x_data = [], [], []     # Sparse matrix data for Mobius inversion matrix
    o_n_ancestors, u_n_ancestors = [], []   # Number of descendants
    for idx, P in enumerate(link_candidates):

        # Common cumulant corresp. to P is observed if and only if |P| <= order
        if len(P) <= i:
            o_idx.append(idx)
            o_n_ancestors.append(len(nx.ancestors(Gf, P)))
        else:
            u_idx.append(idx)
            u_n_ancestors.append(len(nx.ancestors(Gf, P)))

        # Record diagonal entry of inversion matrix
        x_rows.append(idx)
        x_cols.append(idx)
        x_data.append(1)

        # Record off-diagonal entries for sufficiently small path sets
        for Q in nx.descendants(Gf, P):
            if len(Q) <= max_size:
                x_rows.append(idx)
                x_cols.append(link_candidate_indices[Q])
                x_data.append((-1) ** (len(Q) - len(P)))

    # Correct columns of the inversion matrix corresponding to sets in the bounding topology.
    # Matrix entries are computed according to Lemma 9 in the paper.
    for B in bounding_topo:
        for P_len in range(1, 1 + min(max_size, len(B) - 1)):
            for P in combinations(B, P_len):
                sgn = 1 if (max_size + 1 - P_len) % 2 == 0 else -1
                coef = scipy.special.binom(len(B) - P_len - 1, max_size - P_len)
                x_rows.append(link_candidate_indices[frozenset(P)])
                x_cols.append(link_candidate_indices[B])
                x_data.append(sgn * coef)

    # Construct sparse inversion matrix
    X = scipy.sparse.coo_matrix((x_data, (x_rows, x_cols)))
    X = X.tocsr()

    # Partition inversion matrix
    o_idx, u_idx = np.array(o_idx, dtype=int), np.array(u_idx, dtype=int)
    Xuu = X[u_idx, :][:, u_idx]
    Xuo = X[u_idx, :][:, o_idx]
    Xou = X[o_idx, :][:, u_idx]
    Xoo = X[o_idx, :][:, o_idx]
    Xu = scipy.sparse.vstack([Xuu, Xou])
    Xo = scipy.sparse.vstack([Xuo, Xoo])

    n_ancestors = np.array(u_n_ancestors + o_n_ancestors)
    return Xo, Xu, o_idx, u_idx, n_ancestors, link_candidates


def solve_optimization(Xo, Xu, fo_hat, fo_sigma, o_idx, u_idx, n_ancestors, l1_weight, l1_exponent, **solver_args):

    # Prepare variables
    if len(u_idx) > 0:
        fu = cp.Variable(len(u_idx), )
    fo = cp.Variable(len(o_idx), )

    # Prepare L2 cost function
    fo_err = fo - fo_hat
    L2_cost = cp.quad_form(fo_err, np.diag(np.power(fo_sigma, -2)))
    if len(u_idx) > 0:
        g = Xu @ fu + Xo @ fo
    else:
        g = Xo @ fo

    # Prepare L1 regularizer
    l1_coefs = l1_weight * np.power(n_ancestors, l1_exponent)
    L1_cost = cp.sum(cp.multiply(l1_coefs, cp.abs(g)))

    # Solve
    prob = cp.Problem(cp.Minimize(L2_cost + L1_cost))
    prob.solve(**solver_args)

    # Return solution
    g = g.value
    fo = fo.value if len(o_idx) > 0 else np.zeros(0, )
    fu = fu.value if len(u_idx) > 0 else np.zeros(0, )
    return g, fo, fu


# HELPER FUNCTIONS


def create_threshold_fn(powers, thresholds):
    """
    Create a threshold function (for use with Algorithms 2 and 3).

    Parameters
    ----------
    powers : List[float]
        List of estimates for the statistical power of the nonzero common cumulant hypothesis test.
        Provide one entry for path set sizes 3, 4, ..., order, so that len(powers) == order - 2.
        Each entry should be in the range (0, 1].
    thresholds : List[float]
        List of thresholds.
        Provide one entry for path set sizes 3, 4, ..., order, so that len(thresholds) == order - 2.
        Each entry should be in the range (0, 1].

    Returns
    -------
    threshold_fn : Callable
        threshold_fn(len(B), i) is the min number of size-i subsets of B with a nonzero
        common cumulant in order for B to be accepted.
    """

    def threshold_fn(B_size, i):
        power, threshold = powers[i - 3], thresholds[i - 3]
        n_subsets = scipy.special.binom(B_size, i)
        return scipy.stats.binom(n_subsets, power).isf(threshold)

    return threshold_fn


def create_nonzero_fn(alphas, bootstrapped_delays):
    """
    Create a function to decide if path sets correspond to a nonzero common cumulant
    (for use with Algorithms 2 and 3).

    Parameters
    ----------
    bootstrapped_delays : np.ndarray
        (N, n, n_groups) array of resampled delay data.
    alphas : List[float]
        List of significance levels for the nonzero common cumulant hypothesis test.
        Provide one entry for path set sizes 2, 3, 4, ..., order, so that len(powers) == order - 1.
        Each entry should be in the range (0, 1].

    Returns
    -------
    nonzero_fn : Callable
        nonzero_fn(P) indicates whether P has a nonzero ith-order common cumulant.
    """

    def nonzero_fn(P):
        _, p_val = estimate_common_cumulant(bootstrapped_delays, P)
        return p_val < alphas[len(P) - 2]

    return nonzero_fn


def estimate_common_cumulant(X_batched, P, order=None):
    """
    Estimates the common cumulant associated with a given path set.
    Additionally, a 1-sample t-test is performed with the null hypothesis that the
    common cumulant is zero.

    Parameters
    ----------
    X_batched : np.ndarray
        (N, n, b) data array, with b batches of N observations of n variates.
    P : Collection[int]
        Collection of ints in {0, 1, ..., n-1} representing a path set.
    order : int
        Cumulant order. If None, k is the size of P.

    Returns
    -------
    kstats : np.ndarray
    p_val : float
        p-value for the hypothesis test.
    """
    if order is None:
        order = len(P)
    d = order - len(P)
    if d == 0:
        kstats = kstat(X_batched, tuple(P))
    else:
        kstats = []
        for P_sup in combinations_with_replacement(P, d):
            Q = tuple(P) + P_sup
            kstats.append(kstat(X_batched, Q))
        kstats = np.mean(kstats, axis=0)
    alt = 'greater' if order == 2 else 'two-sided'
    _, p_val = ttest_1samp(kstats, popmean=0, alternative=alt)
    return kstats, p_val
