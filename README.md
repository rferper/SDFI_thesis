# SDFI_thesis

This repository contains the Python code related to the experimental results of the PhD thesis "New characterizations of some families of fuzzy implication functions and their intersections. Applications to subgroup discovery" related to the subgroup discovery data mining technique. In particular, we provide an implementation of four different algorithms:

- SDFIOE: A Top-k subgroup discovery algorithm based on fuzzy implication functions with an optimistic estimate pruning and an exhaustive search.
- GSDFIW: A Top-k subgroup discovery algorithm based on fuzzy implication functions with a greedy search and a weighted covering algorithm.
- WCSDFI: A weighted covering post-processing technique for subgroup discovery algorithms based on fuzzy implication functions.
- STFI: An algorithm for searching sharp transitions based on fuzzy implication functions.

In all algorithms, the IF-THEN rules of the corresponding subgroups/transitions are modeled as fuzzy rules using a t-norm and a fuzzy implication function.

The algorithms allow a custom selection of the fuzzy partitions and the fuzzy operators. Nonetheless, these specifications should be selected manually by the user. Thus, for an easier use of this code, there are available six different pairs of fuzzy implication function I and t-norm T, and three automatic fuzzy partitioning methods (uniform, cmeans and fcm).

Finally, in the folder phd_results all the code linked to the results presented in the thesis monograph can be found.
