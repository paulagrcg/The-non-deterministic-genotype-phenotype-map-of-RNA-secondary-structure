This folder contains the code required for the preprint "The non-deterministic genotype-phenotype map of RNA secondary structure" (Paula García-Galindo, Sebastian E. Ahnert, and Nora S. Martin, bioRxiv 2023) for the RNA length 12 analysis.

This code uses ViennaRNA (2.5.1) and Python 3.9.7, and various standard Python packages (matplotlib etc.).

Further details on the methods used can be found in the jupyter notebook code.

The data included is in pickle format and are dictionaries:

- dictgpmapMFE : complete MFE GP map of RNA12

- dictRNA12tot: complete ND GP map of RNA12 for energy gap 15kbT. (Heavy file)

- MFEquantities: D quantities on the MFE GP map.
		rhogMFE: genotypic robustness
		rhopMFE: phenotypic robustness		
		evgMFE: genotypic evolvability
		evpMFE: phenotypic evolvability
		neutralsetsMFE: phenotype set size

- averagequant500: average of D quantities from 500 deterministic GP map realisations. Genotypes/phenotypes are keys and values are quantity averages.

		avrhog500: average of genotypic robustness
		avrhop500: average of phenotypic robustness
		avevp500: average of phenotypic evolvability 
		avevg500: average of genotypic evolvability
		avrsets500: average of phenotype set sizes

- NDquant: Genotypes/phenotypes are keys and values are ND quantities.

		rhogND: genotypic robustness
		rhopND: phenotypic robustness		
		evgND: genotypic evolvability
		evpND: phenotypic evolvability
		NDsetsize: phenotype ND set size
	   folddict: {phenotype: [genotype,probability]} (as explained in code)

- dictpmfe: {genotype: mfe probability} as explained in code and preprint for plastogenetic congruence (Supplementary Information)

