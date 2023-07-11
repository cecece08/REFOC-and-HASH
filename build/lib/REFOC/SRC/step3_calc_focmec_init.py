import numpy as np
from numba import int32
import Input_parameters as ip
from sub_calc_focmec import *
import pickle
import os
from time import time
from random import sample

global thetable
global phitable
global amptable
global b1
global b2
global b3

fid = open(ip.table_dir+'/thetable.pickle','rb')
thetable = pickle.load(fid).astype(np.float32)
fid = open(ip.table_dir+'/phitable.pickle','rb')
phitable = pickle.load(fid).astype(np.float32)
fid = open(ip.table_dir+'/amptable.pickle','rb')
amptable = pickle.load(fid).astype(np.float32)
fid = open(ip.table_dir+'/b1.pickle','rb')
b1 = pickle.load(fid).astype(np.float32)
fid = open(ip.table_dir+'/b2.pickle','rb')
b2 = pickle.load(fid).astype(np.float32)
fid = open(ip.table_dir+'/b3.pickle','rb')
b3 = pickle.load(fid).astype(np.float32)
global nrot
nrot = int32(b3.shape[1])

if __name__ == "__main__":
    evlist =  open(ip.evfile)
    for line in evlist:
        info = line.split()
        inputline = info[0]+'/'+info[0]+info[1]+'/'+info[2]
        #inputline = '2021/202102/39805008'
        fpfile = ip.dir_output+'hashphase/'+inputline+'_hashphase.pkl'
        outfile = ip.dir_output+'iter0/'+inputline+'.focmec'
        if not os.path.isdir(ip.dir_output+'iter0/'):
            os.mkdir(ip.dir_output+'iter0')
        if not os.path.isdir(ip.dir_output+'iter0/'+info[0]):
            os.mkdir(ip.dir_output+'iter0/'+info[0])
        if not os.path.isdir(ip.dir_output+'iter0/'+info[0]+'/'+info[0]+info[1]):
            os.mkdir(ip.dir_output+'iter0/'+info[0]+'/'+info[0]+info[1])
        fout = open(outfile,'w')
        if not os.path.isfile(fpfile):
            print(fpfile+' not exists')
            continue
        print(fpfile+' start')
        t0 = time()
        fin = open(fpfile,'rb')
        ev_hashphase = pickle.load(fin)
        fin.close()
        print('finish read event')
        evid = ev_hashphase['evid']
        latitude =  ev_hashphase['latitude']
        longitude = ev_hashphase['longitude']
        depth = ev_hashphase['depth']
        magnitude = ev_hashphase['magnitude']
        nppl = ev_hashphase['nppl']
        nspr = ev_hashphase['nspr']
        evtime = ev_hashphase['time_orig']
        p_azi_mc = ev_hashphase['p_azi_mc'].astype(np.float32)
        p_the_mc = ev_hashphase['p_the_mc'].astype(np.float32)
        sp_ratio = ev_hashphase['sp_ratio'].astype(np.float32)
        p_pol = ev_hashphase['p_pol'].astype(np.float32)
        ntot = int32(ev_hashphase['ntot'])
        nmc = int32(ip.nmc)
        dang = ip.dang
        namp = len(ev_hashphase['indi_amp'])
        print(str(evid)+' ntot:'+str(ntot)+' nppl:'+str(nppl)+' nspr:'+str(nspr)+' namp:'+str(namp))
        nextra = max(int(nppl*ip.badfrac*0.5),2)
        nmismax = max(int(nppl*ip.badfrac),2)

        qextra = max(nspr*ip.qbadfrac*0.5,2)
        qmismax = max(nspr*ip.qbadfrac, 2)

#### start to calculate misfit
        griddim=(8192,1,1)
        blockdim=(512,1,1)
        nmis = np.zeros(nrot).astype(np.float32)
        qmis = np.zeros(nrot).astype(np.float32)
        CU_FocalAmp_Misfit[griddim,blockdim](p_azi_mc,p_the_mc,sp_ratio,p_pol,ntot,1,nrot,\
                            thetable,phitable,amptable,b1,b2,b3,qmis,nmis)
        np.savetxt('./qmis0_'+str(evid)+'.txt',qmis)
        np.savetxt('./nmis0_'+str(evid)+'.txt',nmis)
        index = np.where((nmis<max(nmismax,nmis.min()+1*nextra))&(qmis<max(qmismax,qmis.min()+1*qextra)))[0]
        nf = len(index)
        #print(nf)
        if nf > ip.maxout:
            index = sample(list(index),ip.maxout)
            nf = ip.maxout
        if ntot==0:
            index = []
            nf = 0
        norm1 = b3[:,index]
        norm2 = b1[:,index]
        norm1_avg,norm2_avg = Mech_Avg(nf,norm1,norm2)

        nmult, str_avg, dip_avg, rak_avg, prob, var_est = Mech_Prob(nf,norm1,norm2,ip.cangle,ip.prob_max)
#        print(var_est)
        print('finish obtain best-fitting focmec')
        var_avg = np.zeros(nmult)
        for imult in np.arange(0,nmult):
            var_avg[imult] = (var_est[0,imult]+var_est[1,imult])/2
        i = np.argmax(prob[:nmult])

        ev_hashphase['str_avg'] = str_avg[i]
        ev_hashphase['dip_avg'] = dip_avg[i]
        ev_hashphase['rak_avg'] = rak_avg[i]
        ev_hashphase['var_est1'] = var_est[0,i]
        ev_hashphase['var_est2'] = var_est[1,i]
        ev_hashphase['prob'] = prob[i]
        mfrac,mavg,stdr = Get_Misf_Amp(ntot,p_azi_mc[:,5],p_the_mc[:,5],sp_ratio,p_pol,str_avg[i],dip_avg[i],rak_avg[i])
#        print(str(time()-t0)+' second')

        ev_hashphase['mfrac'] = mfrac
        ev_hashphase['mavg'] = mavg
        ev_hashphase['stdr'] = stdr

        magap,mpgap = Get_Gap(ntot,p_azi_mc[:,5],p_the_mc[:,5])
        ev_hashphase['magap'] = magap
        ev_hashphase['mpgap'] = mpgap
        if prob[i]>0.8 and var_avg[i]<=25 and mfrac<=0.15 and stdr>=0.5:
            qual = 'A'
        elif prob[i]>0.6 and var_avg[i]<=35 and mfrac<=0.2 and stdr>=0.4:
            qual = 'B'
        elif prob[i]>0.5 and var_avg[i]<=45 and mfrac<=0.3 and stdr>=0.3:
            qual = 'C'
        elif magap<=90 and mpgap<=60:
            qual = 'D'
        elif magap>90 or mpgap>60:
            qual = 'E'
        else:
            qual = 'F'
        string = '{:>4} {:>2} {:>2} {:>2} {:>2} {:6.3f} '.format(evtime.year,evtime.month,evtime.day,evtime.hour,evtime.minute,evtime.second+evtime.microsecond/1000000)+\
             '{:>16} {:9.5f} {:10.5f} {:7.3f}  {:5.3f} '.format(str(evid),latitude,longitude,depth,magnitude)+\
             '{:>4} {:>3} {:>4} {:>3} {:>3} {:>4}  {:4.2f} '.format(int(str_avg[i]),int(dip_avg[i]),int(rak_avg[i]),int(var_est[0,i]),int(var_est[1,i]),nppl,mfrac)+\
             '{:>4}  {:4.2f} {:1} {:4.2f} {:>3} {:>3} {:4.2f} {:4.2f}'.format(nspr,mavg,qual,prob[i],int(magap),int(mpgap),mfrac,stdr)
        fout.write(string+'\n')
        print(string)
        print(str(time()-t0)+' second')
        fout.close()

        ev_focmec = {}
        syn_amp = {}

        for ista in ev_hashphase['station_angle']:
            ista_angle = ev_hashphase['station_angle'][ista]
            p_amp,s_amp = Get_Syn_Amp(ista_angle[0],ista_angle[1],str_avg[i],dip_avg[i],rak_avg[i])
            syn_amp[ista] = (p_amp,s_amp)
        ev_focmec['station_angle'] = ev_hashphase['station_angle']
        ev_focmec['indi_amp'] = ev_hashphase['indi_amp']
        ev_focmec['syn_amp'] = syn_amp
        ev_focmec['solution'] = (str_avg[i],dip_avg[i],rak_avg[i])
        ev_focmec['quality'] = qual
        ev_focmec['uncertainty'] = (var_est[0,i],var_est[1,i])
        ev_focmec['evid'] = evid
        ev_focmec['location'] = (latitude,longitude,depth)
        ev_focmec['magnitude'] = magnitude
        ev_focmec['misfit'] = (prob,mfrac,mavg,stdr)
        ev_focmec['input No.'] = (nppl,nspr,namp,magap,mpgap)
        fout = open(ip.dir_output+ 'iter0/'+inputline+'_focmec.pkl','wb')
        pickle.dump(ev_focmec,fout)
        fout.close()

