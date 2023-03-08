This folder contains the code required for the preprint "The non-deterministic genotype-phenotype map of RNA secondary structure" (Paula García-Galindo, Sebastian E. Ahnert, and Nora S. Martin, bioRxiv 2023)

This code uses ViennaRNA (2.4.14) and Python 3.7, and various standard Python packages (matplotlib, pandas etc.).

References:
- The code uses our ViennaRNA-based implementation of the RNAshapes concept because it is faster for short sequences (N. S. Martin, S. E. Ahnert, J. R. Soc. Interface. 18, 20210380, 2021).
- ViennaRNA manual: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf

Further details on the methods used and the underlying references can be found in the preprint.

Generic functions (for example for analysing the RNA genotype-phenotype map and creating plots) are adapted from our previous work on the RNA genotype-phenotype map.

The following three scripts are included:
analysis_new_metrics_genotypes.py generates the robustness and evolvability data for RNAshapes30.
analysis_new_metrics_genotypes_frequencies.py generates the phenotypic frequency data for RNAshapes30.
analysis_new_metrics_genotypes_fullstructures_full.py uses RNA12 (where we have exhaustive data) to test the sampling approach used for the ND phenotypic robustness calculation.

