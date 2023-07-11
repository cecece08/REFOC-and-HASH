from numba import int32
import numpy as np
from time import time
import pickle
import Input_parameters as ip
import os
from sub_calc_focmec import *
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

    evdata = np.loadtxt(ip.evfile,usecols=(0,1,2,3,4,5,6))
    year_all = evdata[:,0].astype('int')
    mon_all = evdata[:,1].astype('int')
    evid_all = evdata[:,2].astype('int')
    lat_all = evdata[:,3]
    lon_all = evdata[:,4]
    dep_all = evdata[:,5]
    mag_all = evdata[:,6]

    citer = '1'
    liter = '0'
    for line in evlist:
        info = line.split()
        inputline = info[0]+'/'+info[0]+info[1]+'/'+info[2]
        fpfile = ip.dir_hashphase+inputline+'_hashphase.pkl'
        outdir = ip.dir_output+'iter'+citer
        focdir = ip.dir_output+'iter'+liter
        outfile = outdir+'/'+inputline+'.focmec'
        magfile = outdir+'/'+inputline+'.mag_amp'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        if not os.path.isdir(outdir+'/'+info[0]):
            os.mkdir(outdir+'/'+info[0])
        if not os.path.isdir(outdir+'/'+info[0]+'/'+info[0]+info[1]):
            os.mkdir(outdir+'/'+info[0]+'/'+info[0]+info[1])
        
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
        indi_amp = ev_hashphase['indi_amp']
        sta_ang = ev_hashphase['station_angle']
        namp = len(ev_hashphase['indi_amp'])
        print(str(evid)+' ntot:'+str(ntot)+' nppl:'+str(nppl)+' nspr:'+str(nspr)+' namp:'+str(namp))
        nextra = max(int(nppl*ip.badfrac*0.5),2)
        nmismax = max(int(nppl*ip.badfrac),2)

        qextra = max(nspr*ip.qbadfrac*0.5,2)
        qmismax = max(nspr*ip.qbadfrac, 2)
        # find neighboring events
        dist = hypo_dist(lat_all,lon_all,dep_all,latitude,longitude,depth)
        index = np.where(np.sort(dist)<3)[0]
        if len(index) > 100:
            index = np.argsort(dist)[:100]
        rela_amp = np.zeros((ip.nrela,2,ip.nmc)).astype(np.float32)-999
        rela_azi = np.zeros(ip.nrela).astype(np.float32)
        rela_the = np.zeros((ip.nrela,ip.nmc)).astype(np.float32)
        rela_weight = np.zeros(ip.nrela).astype(np.float32)
        amp_diff = np.zeros((ip.nrela,2)) - 999
        mag_diff = np.zeros(ip.nrela)
        namp = int32(0)
        nump = 0
        nums = 0
        pamp_tot = 0
        samp_tot = 0
        for inei in index:
            focfile = focdir + '/' + str(year_all[inei])+'/'+ \
                        str(year_all[inei])+str(mon_all[inei]).zfill(2)+'/'+\
                        str(evid_all[inei])+'_focmec.pkl'
            try:
                fin = open(focfile,'rb')
            except:
                continue
            ev_focmec = pickle.load(fin)
            fin.close()
            nqual = ev_focmec['quality']
            if nqual == 'A':
                w = 1
            elif nqual == 'B':
                w = 0.7
            elif nqual == 'C':
                w = 0.4
            else:
                continue
            nei_syn_amp = ev_focmec['syn_amp']
            nei_mag = ev_focmec['magnitude']
            nei_indi_amp = ev_focmec['indi_amp']
            for ista_chan in indi_amp:
                if not ista_chan in nei_indi_amp:
                    continue 
                info = ista_chan.split()
                net = info[0]
                sta = info[1]
                if not net+' '+sta in nei_syn_amp:
                    continue
                if not net+' '+sta in sta_ang:
                    continue
                mag_ratio = magnitude - nei_mag
                mag_diff[namp] = mag_ratio
                if indi_amp[ista_chan][0]!=-999 and nei_indi_amp[ista_chan][0]!=-999:
                    real_ratio_p = np.log10(indi_amp[ista_chan][0]/nei_indi_amp[ista_chan][0])
                    amp_diff[namp,0] = real_ratio_p
                    rela_amp[namp,0,:] = real_ratio_p - mag_ratio + np.log10(nei_syn_amp[net+' '+sta][0])
                    nump = nump + 1
                    pamp_tot = pamp_tot + w
                else:
                    rela_amp[namp,0,:] = np.zeros(ip.nmc)-999
                    
                if indi_amp[ista_chan][1]!=-999 and nei_indi_amp[ista_chan][1]!=-999:
                    real_ratio_s = np.log10(indi_amp[ista_chan][1]/nei_indi_amp[ista_chan][1])
                    amp_diff[namp,1] = real_ratio_s
                    rela_amp[namp,1,:] = real_ratio_s - mag_ratio + np.log10(nei_syn_amp[net+' '+sta][1])
                    nums = nums + 1
                    samp_tot = samp_tot + w
                else:
                    rela_amp[namp,1,:] = np.zeros(ip.nmc)-999
                if amp_diff[namp,0]==-999 and amp_diff[namp,1]==-999:
                    continue
                else:
                    rela_azi[namp] = sta_ang[net+' '+sta][0]
                    rela_the[namp,:] = sta_ang[net+' '+sta][1]
                    rela_weight[namp] = w
                    namp = namp + 1
                
                if namp >= ip.nrela:
                    break
            if namp >= ip.nrela:
                break
        np.savetxt(magfile,np.column_stack((mag_diff,amp_diff)))
        pextra = max(int(pamp_tot*ip.badfrac*0.25),2)
        pmismax = max(int(pamp_tot*ip.badfrac),2)
        sextra = max(int(samp_tot*ip.badfrac*0.25),2)
        smismax = max(int(samp_tot*ip.badfrac),2)
        print('namp: '+str(namp)+' ptot: '+str(pamp_tot)+'  stot: '+str(samp_tot))
        print('number of relative amp input:'+str(rela_azi.shape))
        np.savetxt('./rela_azi_'+str(evid)+'.txt',rela_azi)
        np.savetxt('./rela_the_'+str(evid)+'.txt',rela_the)
        np.savetxt('./amp_diff_'+str(evid)+'.txt',amp_diff)
        griddim=(8192,1,1)
        blockdim=(512,1,1)
        nmis = np.zeros(nrot).astype(np.float32)
        qmis = np.zeros(nrot).astype(np.float32)
        pampmis = np.zeros(nrot).astype(np.float32)
        sampmis = np.zeros(nrot).astype(np.float32)
        CU_FocalAmp_Misfit_Rela[griddim,blockdim](p_azi_mc,p_the_mc,sp_ratio,p_pol,ntot,1,nrot,\
                            rela_azi,rela_the,rela_weight,rela_amp,namp,\
                            thetable,phitable,amptable,b1,b2,b3,qmis,nmis,pampmis,sampmis)

        np.savetxt('./nmis_'+str(evid)+'.txt',nmis)
        np.savetxt('./qmis_'+str(evid)+'.txt',qmis)
        np.savetxt('./pampmis_'+str(evid)+'.txt',pampmis)
        np.savetxt('./sampmis_'+str(evid)+'.txt',sampmis)
        index1 = np.where((nmis<max(nmismax,nmis.min()+1*nextra))&(qmis<max(qmismax,qmis.min()+1*qextra)))[0]
        if len(index1)<=5:
            index1 = np.where((nmis<max(nmismax,nmis.min()+1.5*nextra))&(qmis<max(qmismax,qmis.min()+1.5*qextra)))[0]
        if len(index1)<=5:
            index1 = np.where((nmis<max(nmismax,nmis.min()+2*nextra))&(qmis<max(qmismax,qmis.min()+2*qextra)))[0]
        if len(index1)<=5:
            index = index1
        else:
            pampmis1 = pampmis[index1]
            sampmis1 = sampmis[index1]

            index2 = np.where((sampmis1<sampmis1.min()+samp_tot*ip.qbadfrac*0.5)& \
                              (pampmis1<pampmis1.min()+pamp_tot*ip.qbadfrac*0.5))[0]
            print('N solution before rela:'+str(len(index1))+' after rela:'+str(len(index2)))
            if len(index2)<=0:
                index = index1
            else:
                index = index1[index2]
        nf = len(index)
        print('nf:'+str(nf))
        if nf > ip.maxout:
            index = sample(list(index),ip.maxout)
            nf = ip.maxout
        norm1 = b3[:,index]
        norm2 = b1[:,index]
        norm1_avg,norm2_avg = Mech_Avg(nf,norm1,norm2)
        nmult, str_avg, dip_avg, rak_avg, prob, var_est = Mech_Prob(nf,norm1,norm2,ip.cangle,ip.prob_max)
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
        mfrac,mavg,stdr,pavg,savg = Get_Misf_Amp_Rela(ntot,p_azi_mc[:,0],p_the_mc[:,0],sp_ratio,p_pol, \
                                                    rela_azi,rela_the[:,0],rela_amp[:,:,0],rela_weight, \
                                                    str_avg[i],dip_avg[i],rak_avg[i])
        
        ev_hashphase['mfrac'] = mfrac
        ev_hashphase['mavg'] = mavg
        ev_hashphase['stdr'] = stdr
        
        nsrela = len(np.where(amp_diff[:,1]!=-999)[0])
        nprela = len(np.where(amp_diff[:,0]!=-999)[0])
        magap,mpgap = Get_Gap_Rela(ntot,p_azi_mc[:,0],p_the_mc[:,0],rela_azi,rela_the[:,0])
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
        string = '{:>4} {:>2} {:>2} {:>2} {:>2} {:6.3f} '.format(evtime.year,evtime.month,evtime.day,evtime.hour,evtime.minute,evtime.second+evtime.microsecond/1000000)+\
             '{:>16} {:9.5f} {:10.5f} {:7.3f}  {:5.3f} '.format(str(evid),latitude,longitude,depth,magnitude)+\
             '{:>4} {:>3} {:>4} {:>3} {:>3} {:>4}  {:4.2f} '.format(int(str_avg[i]),int(dip_avg[i]),int(rak_avg[i]),int(var_est[0,i]),int(var_est[1,i]),nppl,mfrac)+\
             '{:>4}  {:4.2f} {:1} {:4.2f} {:>3} {:>3} {:4.2f} {:4.2f} {:>4} {:4.2f} {:>4} {:4.2f}'.format(nspr,mavg,qual,prob[i],int(magap),int(mpgap),mfrac,stdr,nprela,pavg,nsrela,savg)
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
        ev_focmec['syn_amp'] = syn_amp
        ev_focmec['indi_amp'] = ev_hashphase['indi_amp']
        ev_focmec['station_angle'] = ev_hashphase['station_angle']
        ev_focmec['solution'] = (str_avg[i],dip_avg[i],rak_avg[i])
        ev_focmec['quality'] = qual
        ev_focmec['uncertainty'] = (var_est[0,i],var_est[1,i])
        ev_focmec['evid'] = evid
        ev_focmec['location'] = (latitude,longitude,depth)
        ev_focmec['magnitude'] = magnitude
        ev_focmec['misfit'] = (prob,mfrac,mavg,stdr)
        ev_focmec['input No.'] = (nppl,nspr,namp,magap,mpgap)
        fout = open(outdir+ '/'+inputline+'_focmec.pkl','wb')
        pickle.dump(ev_focmec,fout)
        fout.close()