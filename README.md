# TDDFT Absorption Spectrum Visualization

This repository contains a Python script for visualizing Time-Dependent Density Functional Theory (TDDFT) output from Orca software. The script processes TDDFT output files to generate electronic absorption spectra and compare them with experimental data.

## Features

- **Spectra Visualization**: Generates electronic absorption spectra from TDDFT output files, supporting both Gaussian and Lorentzian curves.
- **Experimental Data Integration**: Compares theoretical spectra with experimental data (with the name Experiment.txt) if provided. If not, a mock dataset is used.
- **Output Summary**: Generates a summary of the TDDFT data in a DOCX file, tabulating the energy in cm-1, nm and eV.

## Requirements

Ensure you have Python installed on your system. The script requires several Python packages, which are listed in `requirements.txt`.

## Installation

To install the required packages, run:

```
pip install -r requirements.txt
```

## Usage

1. Place your TDDFT output file (e.g., `TDDFT.out`, this can be changed from the orca_tddft_analyzer.py) in the same directory as the script.
2. If available, also place your experimental data file (e.g., `Experiment.txt`) in the same directory.
3. Run the script:

```
python orca_tddft_analyzer.py
```

4. The script will generate electronic absorption spectra and a DOCX file with the analysis summary.

## Configuration

You can configure the script by editing the following variables:

- `orca_outputfile`: Path to the TDDFT output file.
- `x_start`, `x_end`, `x_resolution`: Parameters for the x-axis range and FWHM values.
- `mode`: Curve type for the spectra ('g' for Gaussian, 'l' for Lorentzian).

## Output

- A plot of the electronic absorption spectra.
- A DOCX file (`tddft_data.docx`) containing a summary of the TDDFT data.
- An absorption input file in .txt format that was extracted from the 'TDDFT.out' 
- An output Absorption_Input_Output.csv file that used previously created Absorption_Input.txt 
- An absorption_spectrum.csv file that is used to tabulate DOCX file

## Contributing

Feel free to fork this repository and contribute by submitting pull requests.





