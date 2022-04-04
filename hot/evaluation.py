
import json
from hot.utils import power_set


def evaluate_topo_estimate(estimate_filename, true_filename, **kwargs):
    """
    Evaluate an estimated routing topology against the ground truth.

    Parameters
    ----------
    estimate_filename : str
        JSON file containing the predicted-links (as path sets).
    true_filename : str
        JSON file containing the ground-truth links (as path sets).
    """

    # Load estimated and true topologies
    with open(estimate_filename, 'r') as infile:
        payload = json.load(infile)
        predicted_links = list(map(frozenset, payload['predicted_links']))
        bounding_topo = list(map(frozenset, payload['bounding_topo']))
    with open(true_filename, 'r') as infile:
        true_links = json.load(infile)
        true_links = list(map(frozenset, true_links))

    # Evaluate bounding topology
    support_estimate = set()
    for B in bounding_topo:
        for P in power_set(B):
            support_estimate.add(frozenset(P))
    support_truth = set()
    for L in true_links:
        for P in power_set(L):
            support_truth.add(frozenset(P))
    tp = len(support_estimate & support_truth)
    fp = len(support_estimate - support_truth)
    fn = len(support_truth - support_estimate)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    print('Bounding topology precision: {:.4f}, recall: {:.4f}'.format(prec, rec))

    # Get true positives and false negatives for routing topology
    true_pos, false_pos, false_neg = [], [], []
    for L_true in true_links:
        false_negative = True
        for L_pred in predicted_links:
            if L_true == L_pred:
                false_negative = False
                break
        if false_negative:
            false_neg.append(L_true)
        else:
            true_pos.append(L_true)

    # Get false positives
    for L_pred in predicted_links:
        false_positive = True
        for L_true in true_links:
            if L_pred == L_true:
                false_positive = False
                break
        if false_positive:
            false_pos.append(L_pred)

    # Compute metrics
    tp, fp, fn = len(true_pos), len(false_pos), len(false_neg)
    f1 = tp / (tp + 0.5 * (fp + fn))
    if tp + fp == 0:
        prec = float('nan')
    else:
        prec = tp / (tp + fp)
    if tp + fn == 0:
        rec = float('nan')
    else:
        rec = tp / (tp + fn)

    # Report results
    print('Found {} true positives:'.format(len(true_pos)))
    for L in true_pos:
        print('  ', set(L))
    print('Found {} false positives:'.format(len(false_pos)))
    for L in false_pos:
        print('  ', set(L))
    print('Missed {} false negatives:'.format(len(false_neg)))
    for L in false_neg:
        print('  ', set(L))

    print('Routing topology precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(prec, rec, f1))
