# REFOC-and-HASH

This code uses HASH (Hardebeck and Shearer, 2002; 2003) as the basis to calculate the initial earthquake focal mechanisms using first-motion polarities and P-/S-wave amplitude ratios and use inter-event P-/P-wave and S-/S-wave ampltide ratios to future refine the focal mechanism (Cheng et al., 2023).

I wrote this tool using python and utilized GPU to accelerate the calculation.


Hardebeck, J. L., & Shearer, P. M. (2002). A new method for determining first-motion focal mechanisms. Bulletin of the Seismological Society of America, 92(6), 2264-2276.
Hardebeck, J. L., & Shearer, P. M. (2003). Using S/P amplitude ratios to constrain the focal mechanisms of small earthquakes. Bulletin of the Seismological Society of America, 93(6), 2434-2444.
Cheng, Y., Allen, R. M., & Taira, T. A. (2023). A New Focal Mechanism Calculation Algorithm (REFOC) Using Inter‐Event Relative Radiation Patterns: Application to the Earthquakes in the Parkfield Area. Journal of Geophysical Research: Solid Earth, 128(3), e2022JB025006.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Input File Format](#input-file-format)
- [Examples](#examples)
- [Calculate your own focal mechanisms](#calculate-your-own-focal-mechanisms)
- [License](#license)





## Installation

To install REFOC, first clone this library onto your computer with `git clone` and the address of the git repository (see the green Code button). The easiest way to gather all the dependencies is to create a new conda environment with `conda env create -f env_gpu.yml` in the directory where you've cloned the repository. 
Then, run python setup.py install from the directory where you've cloned the repository.

Note that, the preparation stage does not require GPU. If you perform the data preparation and focal mechanism calculation separately, you can use `env_cpu.yml` for data preparation and `env_gpu.yml` for focal mechanism calculation.

## Usage

The code consists of four steps and each of them has a main code.

Step0: Modify the `Input_parameters.py` file based your own paths and preferred parameters.

Step1: prepare take-off angle table, seismic radiation table, and focal mechanism rotation table
`python step1_prep_tables.py`

Step2: prepare input hashphase file by reading phase file and measuring P- and S-wave amplitude.
`python step2_prep_all_hashphase.py`

Step3: Calculate the initial focal mechanism using polarities and P/S amplitude ratios (HASH).
`python step3_calc_focmec_init.py`

Step4: Further refine the focal mechanisms inter-event P/P and S/S amplitude ratios (REFOC).
`python step4_calc_focmec_refoc.py`

## Input File Format

All required input files can be found from `Input_parameters.py`

Input event file: `evfile`
Detailed earthquake location file: `locfile`
station information file: `stafile`
station polarity reversal file: `plfile`
waveform file: in waveform directory `wf_dir`
list of velocity models: `vm_list`
velocity model file: in velocity model directory `wf_dir`
travel time table file: in table directory `table_dir`

## Examples

an example with 50 earthquakes in the Parkfield area is provided in the `Input_parameters.py`
Simply run the following 4 codes will output the focal mechanism solutions based on HASH (in `$dir_output/iter0/`) and REFOC (in `$dir_output/iter1/`) respectively.

`python step1_prep_tables.py`
`python step2_prep_all_hashphase.py`
`python step3_calc_focmec_init.py`
`python step4_calc_focmec_refoc.py`


## Calculate your own focal mechanisms

1. create input file based on the **Input File Format**
2. change the directories in the `Input_parameters.py` based on your own data
3. run codes from `step1_*.py` to `step4_*.py`
4. check the HASH focal mechanism (in `$dir_output/iter0/`) and REFOC focal mechanism (in `$dir_output/iter1/`)

## License

MIT License

Copyright (c) 2023 Yifang Cheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.







