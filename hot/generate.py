
from numpy.random import Generator, PCG64
import numpy as np
import scipy.stats
import random
import networkx as nx
import json


def generate_tomography_scenario(
        edgelist_filename, data_filename, links_filename, monitors, samples,
        delay_mean, delay_std, delay_scale, seed, verbose=False, **kwargs):
    """
    Randomly generate a tomography scenario based on an underlying network.

    Parameters
    ----------
    edgelist_filename : str
        Path to a file containing the edge list for the underlying network.
    data_filename : str
        Path for the output file of path delay samples.
    links_filename : str
        Path for the output file with the ground-truth links (as path sets).
    monitors : int
        Number of monitor nodes.
    samples : int
        Number of delay samples to generate.
    delay_mean : float
        Average of mean delays for links.
    delay_std : float
        Standard deviation of mean delays for links.
    delay_scale : float
        Scale parameter for link delay gamma distributions.
    seed : int
        Random seed.
    verbose : bool, optional, default=False
        Flag to print info to the command line.
    """

    # Seed RNGs
    random.seed(seed)
    random_state = Generator(PCG64(seed))

    # Load edgelist into NetworkX graph
    if verbose:
        print('\tConstructing graph from {}...'.format(edgelist_filename))
    G = nx.read_edgelist(edgelist_filename, create_using=nx.Graph)

    # Randomly select monitor nodes from the leaves
    if verbose:
        print('\tSelecting monitor paths...')
    leaves = [i for i in G.nodes if G.degree(i) == 1]
    monitor_nodes = random.sample(leaves, k=monitors)

    # Assign random mean delays to each link
    # Assign gamma distributions to each link as well
    # Compute link cumulants
    mean_delay_dist = scipy.stats.norm(delay_mean, delay_std)
    for i, j in G.edges:
        mean_delay = mean_delay_dist.rvs(random_state=random_state)
        G[i][j]['mean_delay'] = mean_delay
        G[i][j]['dist'] = scipy.stats.gamma(a=mean_delay / delay_scale, scale=delay_scale)

    # Compute shortest (delay-weighted) paths between monitor nodes
    monitor_paths = []
    for i in monitor_nodes:
        for j in monitor_nodes:
            if i < j:
                path = nx.shortest_path(G, source=i, target=j, weight='mean_delay')
                monitor_paths.append(path)

    # Store the set of links associated with each monitor path
    path_links = []
    for path in monitor_paths:
        edge_list = []
        for i in range(len(path) - 1):
            u, v = min(path[i], path[i + 1]), max(path[i], path[i + 1])
            edge = (u, v)
            edge_list.append(edge)
        path_links.append(edge_list)

    # Get set of unique links, preserving order
    unique_links, seen_links = [], set()
    for edge_list in path_links:
        for edge in edge_list:
            if edge not in seen_links:
                seen_links.add(edge)
                unique_links.append(edge)

    # Sample delays on each link
    if verbose:
        print('\tSampling delays...')
    link_delay_samples = dict()
    for link in unique_links:
        link_dist = G[link[0]][link[1]]['dist']
        link_delay_samples[link] = link_dist.rvs(samples, random_state=random_state)

    # Compute path delays
    path_delays = []
    for edge_list in path_links:
        path_delay = sum((link_delay_samples[link] for link in edge_list))
        path_delays.append(path_delay)
    path_delay_data = np.stack(path_delays, axis=1)

    # Compute the path sets associated with each link
    link_path_sets = dict()
    for path_idx, path in enumerate(monitor_paths):
        for i in range(len(path) - 1):
            u, v = min(path[i], path[i + 1]), max(path[i], path[i + 1])
            if (u, v) not in link_path_sets:
                link_path_sets[(u, v)] = set()
            link_path_sets[(u, v)].add(path_idx)
    link_path_sets = {link: frozenset(path_set) for link, path_set in link_path_sets.items()}
    link_path_sets = list(set(link_path_sets.values()))

    # # Evaluate routing matrix
    # R = np.zeros((len(paths), len(link_paths)))
    # for i, link in enumerate(link_paths):
    #     R[tuple(link), i] = 1

    # Save results
    if path_delay_data is not None:
        if verbose:
            print('\tWriting delay data to {}...'.format(data_filename))
        np.savetxt(data_filename, path_delay_data, delimiter=',')
    if links_filename is not None:
        if verbose:
            print('\tWriting routing matrix to {}...'.format(links_filename))
        with open(links_filename, 'w') as outfile:
            json.dump(list(map(list, link_path_sets)), outfile)

    if verbose:
        print('\tDone! Sampled {} delays from {} monitor paths traversing {} logical links.'.format(
            samples, len(monitor_paths), len(link_path_sets)))
