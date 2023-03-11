# multicomponent-slurry-quantification
Supporting GitHub information for Multicomponent Slurry Monitoring at Hanford

This repository contains the code, spectral data, and concentration data used in “Quantifying Dense Multicomponent Slurries with In-line ATR-FTIR and Raman Spectroscopy: A Hanford case study.”

## Contents:

1. `ATR-FTIR Concentrations.csv` and `Raman Concentrations.csv` contain gravimetric concentration measurements, where each row corresponds to an experiment and each column corresponds to a chemical species. `ATR-FTIR Concentrations.csv` reports molalities for all species, while `Raman Concentrations.csv` reports $\frac{g \enspace compound} { kg \enspace water}$.
2. `ATR-FTIR Spectra.csv` and `Raman Spectra.csv` contain the spectra corresponding gravimetric measurements in the Concentration files. The first row of spectra corresponds to the first Experiment, and so on.
3. `ATR-FTIR Wavenumbers.csv` and `Raman Wavenumbers.csv` contain the wavenumbers of their corresponding spectra (ATR-FTIR and Raman).
4. `FTIR_Quantification.py` and `Raman_Quantification.py` contain the code for analyzing and plotting the Concentration, Spectra, and Wavenumber .csv files.
