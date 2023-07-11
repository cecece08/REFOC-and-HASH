from time import time
from sub_focmec_amptable import *
from sub_sta_vmodel import *
import pickle
import Input_parameters as ip

table,deptab,delttab = Mk_Table(ip.vmdir,ip.vm_list)
# take-off angle table
fout = open(ip.table_dir+'/table.pickle','wb')
pickle.dump(table,fout)
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
b1,b2,b3,strike_table,dip_table,rake_table = MK_RotTable_1(dang)
fout = open(ip.table_dir+'/b1.pickle','wb')
pickle.dump(b1,fout)
fout = open(ip.table_dir+'/b2.pickle','wb')
pickle.dump(b2,fout)
fout = open(ip.table_dir+'/b3.pickle','wb')
pickle.dump(b3,fout)
print('finish rot table')

