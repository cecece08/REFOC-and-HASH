from time import time
from sub_focmec_amptable import *
from sub_sta_vmodel import *
import pickle
import Input_parameters as ip
import os

if not os.path.isdir(ip.table_dir):
    os.mkdir(ip.table_dir)
table,table_tp,deptab,delttab = Mk_tpTable(ip.vmdir,ip.vm_list)
table,table_ts,deptab,delttab = Mk_tsTable(ip.vmdir,ip.vm_list)

# take-off angle table
fout = open(ip.table_dir+'/table.pickle','wb')
pickle.dump(table,fout)
# ttp table
mttp = np.median(table_tp,axis = 2)
fout = open(ip.table_dir+'/mttp.pickle','wb')
pickle.dump(mttp,fout)
# tts table
mtts = np.median(table_ts,axis = 2)
fout = open(ip.table_dir+'/mtts.pickle','wb')
pickle.dump(mtts,fout)
# depth table
fout = open(ip.table_dir+'/deptab.pickle','wb')
pickle.dump(deptab,fout)
# distance table
fout = open(ip.table_dir+'/delttab.pickle','wb')
pickle.dump(delttab,fout)
print('finish tt table')

# amplitude radiation pattern at all directions relative to the nodal planes
ntab = 180
thetable,phitable,amptable = MK_AmpTable(ntab)
# strike difference table
fout = open(ip.table_dir+'/thetable.pickle','wb')
pickle.dump(thetable,fout)
# dip difference table
fout = open(ip.table_dir+'/phitable.pickle','wb')
pickle.dump(phitable,fout)
# amplitude table
fout = open(ip.table_dir+'/amptable.pickle','wb')
pickle.dump(amptable,fout)
print('finish amp table')

# rotation from the original strike, dip, rake to other orientations
dang = ip.dang
b1,b2,b3,strike_table,dip_table,rake_table,t_table,p_table,b_table = MK_RotTable(dang)
fout = open(ip.table_dir+'/b1.pickle','wb')
pickle.dump(b1,fout)
fout = open(ip.table_dir+'/b2.pickle','wb')
pickle.dump(b2,fout)
fout = open(ip.table_dir+'/b3.pickle','wb')
pickle.dump(b3,fout)
fout = open(ip.table_dir+'/strike_table.pickle','wb')
pickle.dump(strike_table,fout)
fout = open(ip.table_dir+'/dip_table.pickle','wb')
pickle.dump(dip_table,fout)
fout = open(ip.table_dir+'/rake_table.pickle','wb')
pickle.dump(rake_table,fout)
fout = open(ip.table_dir+'/t_table.pickle','wb')
pickle.dump(t_table,fout)
fout = open(ip.table_dir+'/p_table.pickle','wb')
pickle.dump(p_table,fout)
fout = open(ip.table_dir+'/b_table.pickle','wb')
pickle.dump(b_table,fout)
print('finish rot table')
