
Getting Started
===============

Create a Synthetic Dataset
--------------------------

To begin, we will need some data. We can create synthetic measurements of path delays using the `generate`
command. Creating this dataset first needs an underlying network, with nodes representing routers and edges
corresponding to links. In the ``/high-order-tomography/data/rocketfuel/`` directory, we have provided several
real-world networks from the Rocketfuel database, stored as edgelists. Let's create a scenario based on the
AS2914 topology.

Navigate to the ``high-order-tomography`` root directory, and run the following:

.. code-block:: console

    $ hot generate \
        ./data/rocketfuel/AS2914.txt \
        ./data/generated/AS2914samples.csv \
        ./data/generated/AS2914links.json \
        --monitors 5 --samples 100000

Out::

        Constructing graph from ./data/rocketfuel/AS2914.txt...
        Selecting monitor paths...
        Sampling delays...
        Writing delay data to ./data/generated/AS2914samples.csv...
        Writing routing matrix to ./data/generated/AS2914links.json...
        Done! Sampled 100000 delays from 10 monitor paths traversing 18 logical links.

In this scenario, we have randomly designated 5 leaves from the AS2914 network as "monitor nodes".
For each of the 10 pairs of monitor nodes, we have selected a between these nodes as a "monitor path".
The monitor paths do not span the entire network; rather, as the output of the ``generate`` command
indicates, these 10 monitor paths only utilize 18 logical links, leading to a 10x18 routing matrix.
The contents of this ground-truth routing matrix are contained in the newly created file
``./data/rocketfuel/AS2914links.json``::

    [
        [0, 2, 6, 7],
        [2],
        [4, 6],
        [0, 1, 3, 8],
        [9],
        [0, 6, 7],
        [2, 4, 5],
        [2, 3, 4, 5],
        [8, 7],
        [3],
        [2, 4],
        [0, 1, 8],
        [1],
        [1, 4, 6, 9],
        [8, 9, 5, 7],
        [5],
        [4],
        [0, 8]
    ]

(Note the order of lists and of the numbers within them may be different, as these represent unordered sets.)
Each list corresponds to a link, and the numbers indicate which paths utilize the link. For example, the set
``[0, 2, 6, 7]`` represents that the routing matrix contains a column where rows 0, 2, 6, and 7 have an entry
of one, while all other entries are zero.

The second newly created file, ``./data/rocketfuel/AS2914samples.csv``, is an array of path delay samples.
Each column corresponds to a monitor path, and each row is a sample of the total delay along that path.

Infer the Routing Matrix
------------------------

Next, we will try to use the data in ``./data/rocketfuel/AS2914samples.csv`` to try and reconstruct the routing
matrix represented in ``./data/rocketfuel/AS2914links.json``.  From the ``high-order-tomography`` directory, run
the following:

.. code-block:: console

    $ hot infer \
        ./data/generated/AS2914samples.csv \
        ./output/AS2914/ \
        --order 3 --alphas 1e-40 1e-30 --powers 0.95 --thresholds 0.85 \
        --max-size 4 --l1-weight 1.0 --l1-exponent 0.5 --n-groups 50

Out::

    Estimating bounding topology...
    Estimating common cumulants...: 100% 60/60 [00:29<00:00,  2.03it/s]

The first argument is the synthetic dataset we created in the previous section, and the second argument is
the directory where the routing matrix estimate and auxiliary files will be written. The remaining settings
are tunable parameters that are described in the command line documentation.

The directory ``high-order-tomography/output/AS2914/`` contains 5 new files. For the purposes of this tutorial,
we are only interested in ``predicted-links.json``. The file contains several attributes recording the parameters
to reproduce this particular estimate, as well as the ``predicted-links`` attribute itself.
The value of the ``predicted-links`` attribute is a list-of-lists encoding a routing matrix in the same format
as ``./data/rocketfuel/AS2914links.json``: each list is a link, containing the set of monitor paths that utilize
the link.

Evaluating the Prediction
-------------------------

How accurate was the prediction? We can find out using the ``evaluate`` command:

.. code-block:: console

    $ hot evaluate ./output/AS2914/predicted-links.json ./data/generated/AS2914links.json

The first line of the output evaluates the bounding topology, i.e., estimate for which common cumulants are nonzero.
The precision and recall indicate that the bounding topology estimate was 100% accurate; i.e., the algorithm
was able to determine exactly which path sets correspond to nonzero common cumulants::

    Bounding topology precision: 1.0000, recall: 1.0000

The next lines indicate the true positives. These are all of the columns of the routing matrix (represented by the
list of paths corresponding to nonzero entries) that the algorithm correctly identified::

    Found 17 true positives:
        {0, 2, 6, 7}
        {2}
        {4, 6}
        {0, 1, 3, 8}
        {9}
        {0, 6, 7}
        {2, 4, 5}
        {2, 3, 4, 5}
        {8, 7}
        {3}
        {0, 1, 8}
        {1}
        {1, 4, 6, 9}
        {8, 9, 5, 7}
        {5}
        {4}
        {0, 8}

The next lines indicate the false positives, i.e., columns of the routing matrix that the algorithm falsely detected::

    Found 4 false positives:
        {7}
        {8}
        {0}
        {6}

The next lines are the columns of the routing matrix that were missed::

    Missed 1 false negatives:
        {2, 4}

Finally, the last line reports the precision, recall, and F1 score of the routing matrix estimate::

    Routing topology precision: 0.8095, recall: 0.9444, f1: 0.8718
