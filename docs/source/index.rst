
High-Order Tomography
=====================

*Network tomography* is a family of techniques for inferring information about the structure
and internal state of a network from end-to-end measurements.
Classically, tomography has been used to estimate properties of links, such as their average delay
or packet loss rate, assuming that the routing matrix is known.
But routing matrices are becoming increasingly difficult to identify. The traditional approach of
using traceroutes is becoming less effective, as
`more and more routers are configured to ignore traceroute probes <https://link.springer.com/chapter/10.1007/978-3-642-00975-4_3>`_.
Without access to the routing matrix, most applications of network tomography become impossible.

Fortunately, network tomography has come to its own rescue in recent work, as techniques have been
developed to infer the routing matrix itself from end-to-end measurements.
Tomography does not require internal routers to cooperate with traceroutes, since the probes can
be disguised as normal network traffic: only the statistics of delays, loss rates, etc.
are relevant. In particular, second-order statistics known as *path sharing metrics*
(such as covariances in delays) can be used to reconstruct multicast trees.

High-Order Tomography (HOT) aims to infer general routing matrices from end-to-end data,
without making any assumptions about the routing behavior.
In order to accomplish this, we use high-order statistics of end-to-end data,
going beyond second-order path sharing metrics. The mathematics are detailed in our paper
`Topology Inference with Multivariate Cumulants: The Möbius Inference Algorithm <https://arxiv.org/abs/2005.07880>`_,
which is to appear in *IEEE/ACM Transactions on Networking*.

Finally, a word of warning: since this project is a prototype for research purposes, this code should
not be used in a production setting. We make no guarantees regarding its accuracy.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting-started
   commands
   copyright
