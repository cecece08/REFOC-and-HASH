from obspy.core.utcdatetime import UTCDateTime
import numpy as np
def get_evinfo(line,evid,ftype='phase'):
    if ftype=='phase':
        info=line.split()
        qlat=float(line[42:51])
        qlon=float(line[52:63])
        qdep=float(line[63:70])
        qmag=float(line[70:75])
        qtime=UTCDateTime(line[17:41])
        flag=1
    elif ftype=='reloc':
        flag=0
        qlat=0
        qlon=0
        qdep=0
        qmag=0
        qtime=0
        fid=open(line)
        for line in fid:
            info=line.split()
            if info[6]==evid:
                flag=1
                qlat=float(info[7])
                qlon=float(info[8])
                qdep=float(info[9])
                qmag=float(info[10])
                hr=int(info[3])
                minu=int(info[4])
                sec=float(info[5])
                if sec<0:
                    sec=sec+60
                    minu=minu-1
                if sec>=60:
                    sec=sec-60
                    minu=minu+1
                if minu<0:
                    hr=hr-1
                    minu=minu+60
                if minu>=60:
                    hr=hr+1
                    minu=minu-60
                qtime=UTCDateTime(int(info[0]),int(info[1]),
                                  int(info[2]),hr,minu,sec)
                break
        fid.close()
    return qlat,qlon,qdep,qmag,qtime,flag


def get_phaseinfo(line):
    info=line.split()
    net=info[0]
    sta=info[1]
    chan=info[2]

    slat=float(info[4])
    slon=float(info[5])
    phase=info[7]
    dist=float(info[-2])
    tt=float(info[-1])
    polar=info[8][0]
    onset='E'

    if polar=='c':
        polar='U'
        onset='I'
    if polar=='d':
        onset='I'
        polar='D'
    if polar=='.':
        onset='E'
    return net,sta,chan,slat,slon,phase,polar,onset,dist,tt

#*************## Functions for station event travel path
def calc_tt(dist,evdep,ttp,tts,phase='P'):
    #epidist = np.arange(0,500,0.5)
    #depth = np.arange(0,100,0.5)
    arrival_std = 0
    arrival = 0
    index_epi = int(dist*2)
    if index_epi >= 1000:
        return 0,0
    index_dep = int(evdep*2)
    if index_dep >= 200:
        return 0,0
    if phase=='P':
        arrival=ttp[index_epi,index_dep]
        if dist < 25:
            arrival_std = 1
        elif dist<50:
            arrival_std = 2
        elif dist<75:
            arrival_std = 3
        elif dist<125:
            arrival_std = 4
        else:
            arrival_std = 0
    if phase=='S':
        arrival=tts[index_epi,index_dep]
        if dist < 50:
            arrival_std = 1
        elif dist<125:
            arrival_std = 2
        else:
            arrival=0
            arrival_std = 0
    return arrival, arrival_std

def calc_deg_azi(lat1,lon1,lat2,lon2):
    if (((lat1==lat2)&(lon1==lon2))|((lat1==90)&(lat2==90))|((lat1==-90)&(lat2==-90))):
        angle=0
        azi=0
        return -1,-1
    raddeg=np.pi/180
    theta1=(90-lat1)*raddeg
    theta2=(90-lat2)*raddeg
    phi1=lon1*raddeg
    phi2=lon2*raddeg
    stheta1=np.sin(theta1)
    stheta2=np.sin(theta2)
    ctheta1=np.cos(theta1)
    ctheta2=np.cos(theta2)
    cang=stheta1*stheta2*np.cos(phi2-phi1)+ctheta1*ctheta2
    ang=np.arccos(cang)
    angle=ang/raddeg
    sang=np.sqrt(1-cang*cang)
    caz=(ctheta2-ctheta1*cang)/(sang*stheta1)
    saz=-stheta2*np.sin(phi1-phi2)/sang
    az=np.arctan2(saz,caz)
    azi=az/raddeg
    if azi<0:
        azi=azi+360
    return angle,azi

def Check_Pol(plfile,isname,iyr,imon,idy):
    fid = open(plfile)
    for line in fid:
        sta = line[:7]
        begintime = int(line[8:16])
        endtime = int(line[17:25])
        
        evtime = iyr*10000+imon*100+idy
        if sta.split()[0] == isname.split()[0]:
            if evtime>begintime and evtime<endtime:
                return -1
    return 1

#**************** Function for P/S ratio *****************
#Functions for P/S ratio
def calc_ps(data_p,data_s,dt,ttp,tts,tt,tbo):
# tbo: second before original time
    if tts-ttp<=1.0:
        return 0,0,0,0,0,0
# find P arrival
    i1=int((ttp-1.5+tbo)/dt)-50
    i2=int((ttp+1.5+tbo)/dt)+50

    if i2/dt-tbo>tts-0.5:
        i2=int((tts-0.5+tbo)/dt)
    if len(data_p[i1:i2])==0:
        index_p=-1
    else:
        index_p,err_p=piki(data_p[i1:i2])
    if index_p>0:
        p_jlh=i1+index_p
        ppick=p_jlh
    elif tt>0:
        ppick=int((tt+tbo)/dt)
    else:
        ppick=int((ttp+tbo)/dt)
# find S arrival
    i1=int((tts-1.5+tbo)/dt)-50
    i2=int((tts+1.5+tbo)/dt)+50
    i1=max(i1,ppick)
    i2=min(i2,len(data_p[:]))
    if len(data_s[i1:i2])==0:
        index_s=-1
    else:
        index_s,err_s=piki(data_s[i1:i2])

    if index_s>=0:
        s_jlh=index_s+i1
        spick=s_jlh
    else:
        spick=int((tts+tbo)/dt)
# get amplitude info
    if spick-ppick-1/dt<0:
        return 0,0,0,0,0,0
    Ttime=2.0
    Tp=int(min(Ttime/dt,spick-ppick-0.5/dt))
    Ts=int(min(Ttime/dt,spick-ppick))
    data_pnoise=data_p[ppick-int((2+Ttime)/dt):ppick-int(2.0/dt)]
    data_snoise=data_s[spick-int((2+Ttime)/dt):spick-Ts]
    data_pamp=data_p[ppick-int(1.0/dt):ppick+Tp]
    data_samp=data_s[spick-int(1.0/dt):spick+int(Ttime/dt)]
    data_p_return=data_p[ppick-int(2.0/dt):ppick+Tp+int(1.0/dt)]
    data_s_return=data_s[spick-int(2.0/dt):spick+int(Ttime/dt)+int(1.0/dt)]
    if len(data_pnoise)*len(data_snoise)*len(data_pamp)*len(data_samp)==0:
#        print('no data in selected window')
        return 0,0,0,0,0,0
    P_noise=get_max_amp(data_pnoise)
    S_noise=get_max_amp(data_snoise)
    P_amp=get_max_amp(data_pamp)
    S_amp=get_max_amp(data_samp)
    return P_noise,S_noise,P_amp,S_amp,data_p_return,data_s_return

def get_max_amp(data):
    return data.max()-data.min()

def piki(data):
    pik_thres=2
    xmax=(np.absolute(data)).max()
    qstep=xmax/30
    s=np.absolute(data/qstep)
    s[np.where(s==0)]=0.1

    c=fbcurv(s)
    ipick=-8
    epik=0
    if ((np.argmax(c)!=0)&(c.max()>=pik_thres)):
        ipick=np.argmax(c)
        temp=((220/c.max()**2)+1)*88
        if temp<1:
            epik=1
        elif temp>50:
            epik=50
        else:
            epik=int(temp)
    if c.max()==0:
        ipick=-8
        epik=0
    elif c.max()<pik_thres:
        ipick=-9
        epik=0
    return ipick,epik

def fbcurv(x):
    wlen=50
    nb=wlen
    nf=wlen
    wf=1-np.arange(wlen).astype('float')/nf
    wb=(np.arange(wlen).astype('float')+1)/nb
    c=np.zeros(len(x))
    nx=len(x)
    c[0:min(nx-1,nb-2)]=1
    c[max(0,nx-nf+1):nx-1]=1
    for i in np.arange(nb,nx-nf):
        back=np.sum(wb*x[i-nb:i])
        fore=np.sum(wf*x[i:i+nf])
        if back!=0:
            c[i]=fore/back
        else:
            c[i]=1
    return c
