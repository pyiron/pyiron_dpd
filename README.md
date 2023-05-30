# pyiron_dpd - Tools for Defect Phase Diagrams

This repository contains tools and workflows to automatically calculate Defect Phase Diagrams (DPDs) for arbitrary crystalline defects in binary systems. 

DPDs show which ordered phase is the most stable segregation pattern at a defect given a certain thermodynamic state variables.  This implementation supports DPDs in $\Delta\mu$ and $\Delta\mu-T$ space. Here we understand $\Delta\mu = \mu_2 - \mu_1$ as the difference in the chemical potential of both binary species, i.e. we work in the semi-grandcanonical ensemble.

The two major workflows are `SegregationFlow` and `IterativeSegregation`.  Examples are given in the `notebooks` folder.  The former exhaustively enumerates all possible segregation patterns up to a configurable number of solutes.  The latter has added methods to screen a large number of segregation patterns with an on-the-fly fitted proxy model before deciding which patterns are worth explicitely calculating.  Both approaches reduce the number of calculations necessary by taking symmetry into account.

# License

The software is provided here for reference only with all rights reserved.  You may **not** copy it without permission.  If you are interested in using this code or expanding it, please contact us first.  We will likely open source this in the future.

# Example DPDs

Below are the DPDs generated by the example notebooks for a grain boundary and a vacancy in fcc Cu decorated with Ag.

## Cu/Ag $\Sigma 5$ [001] GB

![CuAg_GB_DPD](https://github.com/pyiron/pyiron_dpd/assets/2719909/4610f3f5-812d-47bb-9b98-f15244d7765d)
![CuAg_GB_DPD_T](https://github.com/pyiron/pyiron_dpd/assets/2719909/bf945093-a33b-4732-be2a-01441e9e4c0e)

## Cu/Ag Vacancy in Cu host

![CuAg_V_DPD](https://github.com/pyiron/pyiron_dpd/assets/2719909/03fea5d4-9518-4a66-ad51-cec6a42c33ad)
![CuAg_V_DPD_T](https://github.com/pyiron/pyiron_dpd/assets/2719909/0ef3210e-cc6e-4b87-ae85-fd213b87a3e9)

# Caveat

This repository also contains prototypes of various workflow ideas and utility functions.  Do not rely on them in derived work.  They will change, be removed or eventually be subsumed into other pyiron packages.
