import Input_parameters as ip
import numpy as np

def Cross(v1,v2):
# cross product of v1 and v2
    v3 = np.zeros(3)
    v3[0]=v1[1]*v2[2]-v1[2]*v2[1]
    v3[1]=v1[2]*v2[0]-v1[0]*v2[2]
    v3[2]=v1[0]*v2[1]-v1[1]*v2[0]
    return v3

def tpb(strike,dip,rake,ctype='xyz'):
	strike=strike/180*np.pi
	dip=dip/180*np.pi
	rake=rake/180*np.pi
	from numpy import sin,cos,sqrt,pi
	t1=(-sin(strike)*sin(dip)+cos(strike)*cos(rake)+sin(strike)*cos(dip)*sin(rake))/sqrt(2);
	t2=(cos(strike)*sin(dip)+sin(strike)*cos(rake)-cos(strike)*cos(dip)*sin(rake))/sqrt(2);
	t3=(-cos(dip)-sin(dip)*sin(rake))/sqrt(2);

	t_theta=np.arctan(t2/t1)/np.pi*180
	t_phi=np.arctan(np.sqrt(t1*t1+t2*t2)/t3)/np.pi*180
	t_r=np.sqrt(t1*t1+t2*t2+t3*t3)
	if t1<0:
		t_theta=t_theta+180
	if t3<0:
		t_phi=t_phi+180

	p1=(-sin(strike)*sin(dip)-cos(strike)*cos(rake)-sin(strike)*cos(dip)*sin(rake))/sqrt(2);
	p2=(cos(strike)*sin(dip)-sin(strike)*cos(rake)+cos(strike)*cos(dip)*sin(rake))/sqrt(2);
	p3=(-cos(dip)+sin(dip)*sin(rake))/sqrt(2);

	p_theta=np.arctan(p2/p1)/np.pi*180
	p_phi=np.arctan(np.sqrt(p1*p1+p2*p2)/p3)/np.pi*180
	p_r=np.sqrt(p1*p1+p2*p2+p3*p3)
	if p1<0:
		p_theta=p_theta+180
	if p3<0:
		p_phi=p_phi+180
	b1=cos(strike)*sin(rake)-sin(strike)*cos(dip)*cos(rake);
	b2=sin(strike)*sin(rake)+cos(strike)*cos(dip)*cos(rake);
	b3=sin(dip)*cos(rake);
	b_theta=np.arctan(b2/b1)/np.pi*180
	b_phi=np.arctan(np.sqrt(b1*b1+b2*b2)/b3)/np.pi*180
	b_r=np.sqrt(b1*b1+b2*b2+b3*b3)
	if b1<0:
		b_theta=b_theta+180
	if b3<0:
		b_phi=b_phi+180

	if ctype=='xyz':
		t=np.asarray([t1,t2,t3])
		p=np.asarray([p1,p2,p3])
		b=np.asarray([b1,b2,b3])
	elif ctype=='rtf':
		t=np.asarray([t_r,t_theta,t_phi])
		p=np.asarray([p_r,p_theta,p_phi])
		b=np.asarray([b_r,b_theta,b_phi])
	return t,p,b

def FPCOOR(input1,input2,input3,intype='sdr'):
# get fault normal vector, and slip vector from strike-sip rake
    if intype=='sdr':
        strike = input1
        dip = input2
        rake = input3
        
        phi = strike/180*np.pi
        delt = dip/180*np.pi
        lam = rake/180*np.pi
        fnorm = np.zeros(3)
        slip = np.zeros(3)
        fnorm[0] = -np.sin(delt)*np.sin(phi)
        fnorm[1] = np.sin(delt)*np.cos(phi)
        fnorm[2] = -np.cos(delt)
        slip[0] = np.cos(lam)*np.cos(phi)+np.cos(delt)*np.sin(lam)*np.sin(phi)
        slip[1] = np.cos(lam)*np.sin(phi)-np.cos(delt)*np.sin(lam)*np.cos(phi)
        slip[2] = -np.sin(lam)*np.sin(delt)
        
        output1 = fnorm
        output2 = slip
        output3 = 999
        
    elif intype =='fsr':
        fnorm = input1
        slip = input2
        if (1-abs(fnorm[2])<=np.power(10.0,-7)):
            #print('warning: fnorm, strike undefined')
            delt = 0
            phi = np.arctan2(-slip[0],slip[1])
            clam = np.cos(phi)*slip[0]+np.sin(phi)*slip[1]
            slam = np.sin(phi)*slip[0]-np.cos(phi)*slip[1]
            lam = np.arctan2(slam,clam)
        else:
            phi = np.arctan2(-fnorm[0],fnorm[1])
            a = np.sqrt(fnorm[0]*fnorm[0]+fnorm[1]*fnorm[1])
            delt = np.arctan2(a,-fnorm[2])
            clam = np.cos(phi)*slip[0]+np.sin(phi)*slip[1]
            slam = -slip[2]/np.sin(delt)
            lam = np.arctan2(slam,clam)
            if (delt > 0.5*np.pi):
                delt = np.pi - delt
                phi = phi + np.pi
                lam = -lam
        strike = phi/np.pi*180
        if strike < 0:
            strike = strike+360
        dip = delt/np.pi*180
        rake = lam/np.pi*180
        if rake <= -180: 
            rake = rake + 360
        if rake > 180:
            rake = rake - 360
        output1 = strike
        output2 = dip
        output3 = rake
    return output1, output2, output3


def MK_AmpTable(ntab):

    ntab = ip.ntab
    thetable = np.zeros(2*ntab + 1)
    phitable = np.zeros((2*ntab+1,2*ntab+1))
    amptable = np.zeros((2,ntab,2*ntab))
    print('prepare amplitude table')

    astep = float(1.0/ntab)
    for i in np.arange(0,2*ntab+1): # -1 to 1 cos(azimuth)
        bbb3 = -1 + i*astep
        thetable[i] = np.arccos(bbb3)
        for j in np.arange(0,2*ntab+1): #-1 to 1 tan(dip)
            bbb1 = -1 + j*astep
            phitable[i,j] = np.arctan2(bbb3,bbb1)
            if phitable[i,j]<0:
                phitable[i,j] = phitable[i,j]+2*np.pi

    for i in np.arange(0,2*ntab): # 0-2pi azimuth
        phi = i*np.pi*astep
        for j in np.arange(0,ntab): # 0-pi dip
            theta = j*np.pi*astep
            amptable[0,j,i] = abs(np.sin(2*theta)*np.cos(phi)) # P amplitude rable
            s1 = np.cos(2*theta)*np.cos(phi)
            s2 = -np.cos(theta)*np.sin(phi)
            amptable[1,j,i] = np.sqrt(s1*s1+s2*s2) # S amplitude rable
    return thetable,phitable,amptable

def MK_RotTable(dang):

    print('prepare rotation table')
    if dang ==10:
        ncoor = 4032
    elif dang == 7:
        ncoor = 10426
    elif dang == 5:
        ncoor = 29820
    elif dang == 4:
        ncoor = 56100
    elif dang == 3:
        ncoor = 137942
    elif dang == 2:
        ncoor = 464936
    elif dang == 1:
        ncoor = 3716398
    b1 = np.zeros((3,ncoor))
    b2 = np.zeros((3,ncoor))
    b3 = np.zeros((3,ncoor))

    strike = np.zeros((ncoor))
    dip = np.zeros((ncoor))
    rake = np.zeros((ncoor))
    the_all = np.zeros((ncoor))
    phi_all = np.zeros((ncoor))
    zeta_all = np.zeros((ncoor))
    taxis = np.zeros((3,ncoor))
    baxis = np.zeros((3,ncoor))
    paxis = np.zeros((3,ncoor))
    bb3 = np.zeros(3)
    bb1 = np.zeros(3)
    irot = 0
    for ithe in np.arange(0,int(90.1/dang)+1): # take-off angle from 0 to =90
        the = ithe*dang
        rthe = the/180*np.pi
        costhe = np.cos(rthe)
        sinthe = np.sin(rthe)
        fnumang = 360/dang
        numphi = int(fnumang*sinthe)
        if numphi != 0:
            dphi = float(360.0/numphi)
        else:
            dphi = 10000
        for iphi in np.arange(0,int(359.9)/dphi): #azimuth from 0 to <360
            phi = iphi*dphi
            rphi = phi/180.0*np.pi
            for izeta in np.arange(-int(90.1/dang),int(89.9/dang)): #rake from -90 to <90
                zeta = izeta*dang
                rzeta = zeta/180.0*np.pi
                coszeta = np.cos(rzeta)
                sinzeta = np.sin(rzeta)

                b3[:,irot],b1[:,irot],junk = FPCOOR(phi,the,zeta,intype='sdr')
                b2[:,irot] = Cross(b3[:,irot],b1[:,irot])

                the_all[irot] = the
                phi_all[irot] = phi
                zeta_all[irot] = zeta
                strike[irot],dip[irot],rake[irot] = FPCOOR(b3[:,irot],b1[:,irot],2,intype = 'fsr')
                taxis[:,irot],paxis[:,irot],baxis[:,irot] = tpb(strike[irot],dip[irot],rake[irot],ctype='xyz')
                irot = irot + 1
    nrot = irot
    return b1[:,:nrot],b2[:,:nrot],b3[:,:nrot],strike[:nrot],dip[:nrot],rake[:nrot],taxis[:nrot],paxis[:nrot],baxis[:nrot]

