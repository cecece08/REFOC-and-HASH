from obspy.core.utcdatetime import UTCDateTime
from obspy import read
import numpy as np
from time import time
import os
import pickle
from sub_sta_vmodel import *
from sub_hashphase import *
import Input_parameters as ip

global table
global deptab
global delttab
fid = open(ip.table_dir+'/table.pickle','rb')
table = pickle.load(fid)
fid = open(ip.table_dir+'/deptab.pickle','rb')
deptab = pickle.load(fid)
fid = open(ip.table_dir+'/delttab.pickle','rb')
delttab = pickle.load(fid)

global staloc
staloc = {}
fid = open(ip.stafile)
for line in fid:
    net = line[:2]
    sta = line[2:9]
    lat = float(line[51:59])
    lon = float(line[60:70])
    elev = float(line[72:76])
    staloc[net.split()[0]+' '+sta.split()[0]] = (lat,lon,elev)

if __name__ == "__main__":
    fev = ip.evfile
    phase_dir = ip.dir_phase
    out_dir = ip.dir_output
    hashphase_dir = ip.dir_hashphase
    waveform_dir = ip.wf_dir
    freloc = ip.locfile
    vm_dir = ip.vmdir
    fTTp = ip.table_dir+'/ttp_median.txt'
    fTTs= ip.table_dir+'/tts_median.txt'
    TTp = np.loadtxt(fTTp)
    TTs = np.loadtxt(fTTs)
    fid=open(fev)
    for wfline in fid:
        no_wf = 0
        tt1=time()
        if len(wfline)<3:
            print('no enough waveform info')
            continue
        lineinfo = wfline.split()
        evid=lineinfo[2]
        dir0=lineinfo[0]
        dir1=lineinfo[0]+lineinfo[1]
        if not os.path.isfile(phase_dir+'/'+dir0+'/'+dir1+'/'+evid+'.phase'):
            print(phase_dir+'/'+dir0+'/'+dir1+'/'+evid+'.phase')
            print('prep_hash '+evid+' no phase file')
            continue
        pfile=open(phase_dir+'/'+dir0+'/'+dir1+'/'+evid+'.phase')
        if not os.path.isdir(hashphase_dir):
            os.mkdir(hashphase_dir)
        if not os.path.isdir(hashphase_dir+dir0):
            os.mkdir(hashphase_dir+dir0)
        if not os.path.isdir(hashphase_dir+dir0+'/'+dir1):
            os.mkdir(hashphase_dir+dir0+'/'+dir1)
        print(evid)
        evline=pfile.readline()
        if len(evline.split())<=7:
            print('no enough event info')
            continue
        qlat0,qlon0,qdep0,qmag0,qtime0,flag0=get_evinfo(freloc,evid,ftype='reloc')
        qlon0 = -qlon0
        qlat,qlon,qdep,qmag,qtime,flag=get_evinfo(evline,evid,ftype='phase')

        if flag0==0:
            qlat0=qlat
            qlon0=qlon
            qdep0=qdep
            qtime0=qtime
            qmag0=qmag

        if not os.path.isfile(waveform_dir+'/'+wfline.split()[-1]):
            print('prep_hash '+evid+' no waveform')
            no_wf  = 1
        try:
            st_all=read(waveform_dir+'/'+wfline.split()[-1])
        except:
            print('prep_hash '+evid+' no waveform')
            st_all = []
            no_wf = 1

        ev_hashphase = {}
        ev_hashphase['time_orig'] = qtime0
        iyr = qtime0.year
        imon = qtime0.month
        idy = qtime0.day
        ev_hashphase['latitude'] = qlat0
        ev_hashphase['longitude'] = qlon0
        ev_hashphase['depth'] = qdep0
        ev_hashphase['magnitude'] = qmag0
        ev_hashphase['evid'] = evid
        
        aspect = np.cos(qlat0/180*np.pi)
        k = 0
        nspr = 0
        nppl = 0
        npol = 0
        nramp = 0
        
        phase_all = {}
        station_all = {}
        qdep2 = np.zeros(ip.nmc)
        index = np.zeros(ip.nmc).astype('int')
        qdep2[0] = qdep0
        index[0] = 1
        for nm in np.arange(1,ip.nmc):
            val = np.random.normal(0,1,1)
            qdep2[nm] = qdep + 0.2*val
            #index[nm] = int(np.mod(nm,len(tvel_list)))
            index[nm] = int(np.mod(nm,len(ip.vm_list)))
        p_pol = np.zeros(ip.npick0)
        p_azi_mc = np.zeros((ip.npick0,ip.nmc))
        p_the_mc = np.zeros((ip.npick0,ip.nmc))
        sp_ratio = np.zeros(ip.npick0)-999
        for pline in pfile:
            #print(pline)
            net,sta,chan,slat,slon,phase,polar,onset,dist,tt=get_phaseinfo(pline)
            phase_all[net+' '+sta+' '+phase] = (slat,slon,dist,tt)
            if ((phase!='P')):
                continue
            #### get azimuth
            #print('qlon:'+str(qlon0)+' qlat:'+str(qlat0)+' slon:'+str(slon)+' slat:'+str(slat))
            dx = (slon-qlon0)*111.2*aspect
            dy = (slat-qlat0)*111.2
            dist = np.sqrt(dx**2+dy**2)
            #print('dist: '+str(dist))
            qazi = 90 - np.arctan2(dy,dx)/np.pi*180
            if qazi<0:
                qazi = qazi + 360

            depth=min(26,qdep0)
            depth=max(1,qdep0)

            ttp,junk=calc_tt(dist,depth,TTp,TTs,phase='P')
            tts,junk=calc_tt(dist,depth,TTp,TTs,phase='S')
            qdeg,qazi0=calc_deg_azi(slat,slon,qlat0,qlon0)
            cang=np.cos(qazi0/180*np.pi)
            sang=np.sin(qazi0/180*np.pi)
            #### get polarity
            if polar=='U' or polar=='u' or polar=='+':
                p_pol[k] = 1
            elif polar=='D' or polar=='d' or polar=='-':
                p_pol[k] = -1
            else:
                p_pol[k] = 0
            ispol = Check_Pol(ip.plfile,sta,iyr,imon,idy)
            p_pol[k] = p_pol[k]*ispol
            #### find corresponding waveforms
            ixist_spr = 0
            while(1):
                if no_wf ==1:
                    break
                tstart=qtime+tt-10
                tend=qtime+tts+10
                if tend<=tstart+10:
                    print('short window: '+net+' '+sta+' '+chan)
                    break
                st1=st_all.select(channel=chan,station=sta)
                st2=st_all.select(station=sta,channel=chan[:-1]+'N')+st_all.select(station=sta,channel=chan[:-1]+'1')
                st3=st_all.select(station=sta,channel=chan[:-1]+'E')+st_all.select(station=sta,channel=chan[:-1]+'2')
                if st1.count()*st2.count()*st3.count()==0:
                    print('no enough 3-component data 1:'+net+' '+sta+' '+chan)
                    break
                st1=st1.trim(starttime=tstart,endtime=tend)
                st2=st2.trim(starttime=tstart,endtime=tend)
                st3=st3.trim(starttime=tstart,endtime=tend)
                if st1.count()*st2.count()*st3.count()==0:
                    print('no enough 3-component data 2:'+net+' '+sta+' '+chan)
                    break
                st=st1+st2[0]+st3[0]
                if st[0].stats.delta<0.01:
                    st.filter(type='lowpass',freq=40.0)
                try:
                    st.filter('bandpass',freqmin=ip.freqmin,freqmax=ip.freqmax)
                    st.interpolate(100.0)
                except:
                    print('error when filtering the data: '+net+' '+sta+' '+chan)
                    break
                tstart=qtime+tt-5
                tend=qtime+tts+5
                latest_start = np.max([x.stats.starttime for x in st])
                earliest_stop = np.min([x.stats.endtime for x in st])
                if latest_start>tstart:
                    tstart=latest_start
                if earliest_stop<tend:
                    tend=earliest_stop
                if tstart<tend:
                    dt=st[0].stats.delta
                    wflen=int((tend-tstart)/dt)
                    data=np.zeros((5,wflen))
                    for itr in range(3):
                        istart=int((tstart-st[itr].stats['starttime'])/dt)
                        data[itr,:]=st[itr].data[istart:istart+wflen]
                    data[3,:]=-data[1,:]*cang-data[2,:]*sang
                    data[4,:]=data[1,:]*sang-data[2,:]*cang
                    data_p=np.sign(data[0,:]+data[3,:])*np.sqrt(data[0,:]**2+data[3,:]**2)
                    data_s=np.sign(data[0,:]+data[3,:]+data[4,:])* \
                           np.sqrt(data[0,:]**2+data[3,:]**2+data[4,:]**2)
                    if st[1].stats['channel'][-1]=='1':
                        data_p=data_s
                    tbo=qtime-tstart
                    P_noise,S_noise,P_amp,S_amp,data_p_return,data_s_return=calc_ps(data_p,data_s,dt,ttp,tts,tt,tbo)
                else:
                    break

                if abs(P_noise)>0.1 and abs(S_noise)>0.1 and abs(P_amp)>0.1 and abs(S_amp)>0.1:
                    s2n1 = abs(P_amp/P_noise)
                    s2n2 = abs(S_amp/S_noise)
                    spin = abs(S_amp/P_amp)
                    if s2n1 < ip.ratmin:
                        break
                    sp_ratio[k] = np.log10(spin)
                    ixist_spr = 1
                break
            if p_pol[k]==0 and ixist_spr==0:
                continue
            for nm in np.arange(0,ip.nmc):
                p_azi_mc[k,nm] = qazi
                #p_the_mc[k,nm] = 180 - get_tka(index[nm],qdeg,qdep2[nm],tka_table)
                iflag, p_the_mc[k,nm] = Get_TTS(index[nm],dist,qdep2[nm],table,deptab,delttab)
                #print('dist:'+str(dist)+' dep:'+str(qdep2[nm])+' tka:'+str(p_the_mc[k,nm]))
            if p_the_mc[k,0] == 999:
                print('no tka')
                continue

            if p_pol[k]!=0:
                nppl = nppl + 1
            if ixist_spr!=0:
                nspr = nspr+1
            station_all[net+' '+sta] = (qazi,p_the_mc[k,:])
            k = k+1
        npol = nppl + nspr
        ntot = npol
        for i in range(0,k):
            if p_pol[i]!=0 and sp_ratio[i]!=-999:
                ntot = ntot - 1