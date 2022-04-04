
import argparse
import sys
from hot.generate import generate_tomography_scenario
from hot.inference import infer_topology
from hot.evaluation import evaluate_topo_estimate


def setup_parser():

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    # Arguments for topology inference
    parser_infer = subparsers.add_parser('infer', description='Infer routing topology from additive path data.')
    parser_infer.set_defaults(func=infer_topology, verbose=True)
    parser_infer.add_argument(
        'data_filename',
        help='CSV file containing the path data sample.',
        type=str)
    parser_infer.add_argument(
        'output_directory',
        help='Directory where output files from the inference are saved.',
        type=str)
    parser_infer.add_argument(
        '--order',
        help='Maximum cumulant order to evaluate.',
        type=int,
        default=3)
    parser_infer.add_argument(
        '--alphas',
        help='Significance threhsolds for nonzero common cumulant tests at orders 2, 3, 4, ..., <order>. '
             'E.g., "--alphas 1e-40 1e-30" accepts nonzero 2nd-order ccs to a significance of '
             '1e-40 and 3rd-order ccs to a significance of 1e-30.',
        nargs='+',
        type=float)
    parser_infer.add_argument(
        '--powers',
        help='Statistical power estimates for the nonzero common cumulant tests at orders 3, 4, ..., <order>. '
             'E.g., "--powers 0.95 0.9" estimates the test power to be 0.95 for 3rd-order ccs and '
             '0.9 for 4th-order ccs. If a single float is provided, this power is used for all orders.',
        nargs='+',
        type=float)
    parser_infer.add_argument(
        '--thresholds',
        help='Robustness thresholds for tightening the bounding topology at orders 3, 4, ..., <order>. '
             'Similar to --powers. If a single float is provided, this threshold is used for all orders.',
        nargs='+',
        type=float)
    parser_infer.add_argument(
        '--n-groups',
        help='Number of resamples to generate by bootstrapping.',
        type=int)
    parser_infer.add_argument(
        '--resample-size',
        help='Size of each resample to generate by bootstrapping. '
             'If 0, the resample sizes are the same size as the original sample.',
        type=int,
        default=0)
    parser_infer.add_argument(
        '--max-size',
        help='Largest acceptable size of a non-maximal path set.',
        type=int)
    parser_infer.add_argument(
        '--l1-weight',
        help='Weight for the lasso regularizer.',
        type=float,
        default=0.1)
    parser_infer.add_argument(
        '--l1-exponent',
        help='Exponent for the lasso regularizer.',
        type=float,
        default=0.0)
    parser_infer.add_argument(
        '--nnz-threshold',
        help='Threshold to decide if estimated exact cumulants are nonzero.',
        type=float,
        default=1e-3)
    parser_infer.add_argument(
        '--solver_args',
        help='JSON object of arguments for the CVXPY solver. '
             'See CVXPY documentation at https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options',
        type=str,
        default='{"verbose": false, "solver": "OSQP", "eps_abs": 1e-10, "eps_rel": 1e-10}'
    )
    parser_infer.add_argument(
        '--seed',
        help='Random seed.',
        type=int,
        default=1234)

    # Arguments for data generation
    parser_generate = subparsers.add_parser('generate', description='Generate synthetic path delay data.')
    parser_generate.set_defaults(func=generate_tomography_scenario, verbose=True)
    parser_generate.add_argument(
        'edgelist_filename',
        help='Edge list file for the underlying network.',
        type=str)
    parser_generate.add_argument(
        'data_filename',
        help='Path for the output file of path delay samples.',
        type=str)
    parser_generate.add_argument(
        'links_filename',
        help='Path for the output file with the ground-truth links (as path sets).',
        type=str)
    parser_generate.add_argument(
        '--monitors',
        help='Number of monitor nodes.',
        type=int,
        default=5)
    parser_generate.add_argument(
        '--samples',
        help='Number of path delay samples.',
        type=int,
        default=100000)
    parser_generate.add_argument(
        '--delay-mean',
        help='Average of mean delays for links.',
        type=float,
        default=10.0)
    parser_generate.add_argument(
        '--delay-std',
        help='Std. dev. of mean delays for links.',
        type=float,
        default=2.0)
    parser_generate.add_argument(
        '--delay-scale',
        help='Scale parameter for link delay gamma distributions.',
        type=float,
        default=4.0)
    parser_generate.add_argument(
        '--seed',
        help='Random seed.',
        type=int,
        default=1234)

    # Arguments for evaluation
    parser_eval = subparsers.add_parser('evaluate',
                                        description='Evaluate an estimated routing topology against ground truth.')
    parser_eval.set_defaults(func=evaluate_topo_estimate, verbose=True)
    parser_eval.add_argument(
        'estimate_filename',
        help='JSON file containing the predicted-links (as path sets).',
        type=str)
    parser_eval.add_argument(
        'true_filename',
        help='JSON file containing the ground-truth links (as path sets).',
        type=str)

    return parser


def main():

    parser = setup_parser()

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    arg_dict = vars(args)
    args.func(**arg_dict)


if __name__ == '__main__':
    main()
