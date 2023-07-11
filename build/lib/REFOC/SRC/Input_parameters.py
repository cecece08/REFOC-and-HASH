#*************************************************************************#
#                                                                         #
#  script INPUT_PARAMETERS                                                #
#                                                                         #
#  list of input parameters needed for the REFOC calculation              #
#                                                                         #
#*************************************************************************#


import numpy as np
import os

### NOTE: do not remove r before strings (r'filename'), to safely use
#         backslashes in filenames
#--------------------------------------------------------------------------
# input files for focal mechanism calculation
#--------------------------------------------------------------------------
# input evlist with waveform path
evfile = r'../INPUT/evlist.txt'
# location file for all earthquakes 
locfile = r'../INPUT/ncsn_relocation.txt'
# station file 
stafile = r'../INPUT/station_merge.txt'
# polarity reversal file
plfile = r'../INPUT/hash/merged.reverse_7char/'
# directory for waveform files
wf_dir = r'../INPUT/WAVEFORM/'
# directory for phase files
dir_phase = r'../INPUT/PHASE/'
# directory for velocity model files
vmdir = r'/INPUT/VMODEL/'
# directory for saving tables for focal mechanism calculation 
table_dir = r'../INPUT/tables'
# list of input velocity model files
vm_list = ["vz.socal", "vz.north", "vz.lab1", "vz.sgm1", "vz.vb1", "vz.emojave.smof_v2",  \
                "vz.epenin.smof_v2", "vz.wpenin.smof_v2", "vz.anza.smof_v2"]
#--------------------------------------------------------------------------
# Output files with temporary and final results
#--------------------------------------------------------------------------

# directory for saving hashphase file (input for REFOC)
dir_hashphase = r'../OUTPUT/hashphase'
# directory for saving output file (output for REFOC)
dir_output = r'../OUTPUT/'

#--------------------------------------------------------------------------
# parameters controling input quality
#--------------------------------------------------------------------------
# minimum number of polarities
npolmin = 8
# maximum azimuth gap
max_agap = 90
# maximum take-off angle gap
max_pgap = 60
# maximum station-event distance
delmax = 120
# signal to noise ratio of the input amplitude ratios
ratmin = 3
# minimum/maximum frequency for amplitude measurement
freqmin=1
freqmax=10
#--------------------------------------------------------------------------
# parameters controling inversion process
#--------------------------------------------------------------------------
# step (in degree) for searching all possible focal mechanism solutions
dang = 1
# number of perturbations in depth-take-off-angle combination
nmc = 30
# maximum number of relative amplitude ratios used in inversion
nrela = 5000
# maximum number of polarity used in inversion
npick0 = 1000
# number for controling the misfit calculation
ntab  = 180
#--------------------------------------------------------------------------
# parameters controling output properties
#--------------------------------------------------------------------------
# maximum number of output focal mechanisms
maxout = 500
# maximum acceptable polarity misfit 
# final results with be solutions with misfit < max(badfrac,min(misfit)+0.5*badfrac)
badfrac = 0.05
# maximum acceptable log10(s/P amplitude ratio) misfit 
# final results with be solutions with qmisfit < max(qbadfrac,min(qmisfit)+0.5*qbadfrac)
qbadfrac = 0.3
# angle used for calculating focal mechanism uncertainty (all focal mechanisms within cangle from the best-fitting one)
cangle = 45
# minimum probability required for possible output solutions
prob_max = 0.1
#
