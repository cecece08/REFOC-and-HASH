import Input_parameters as ip
import numpy as np

def Cross(v1,v2):
# cross product of v1 and v2
    v3 = np.zeros(3)
    v3[0]=v1[1]*v2[2]-v1[2]*v2[1]
    v3[1]=v1[2]*v2[0]-v1[0]*v2[2]
    v3[2]=v1[0]*v2[1]-v1[1]*v2[0]
    return v3

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

def MK_RotTable_1(dang):
    print('prepare rotation table')
    if dang ==10:
        ncoor = 4032*2
    elif dang == 7:
        ncoor = 10426*2
    elif dang == 5:
        ncoor = 31032*2
    elif dang == 3:
        ncoor = 141180*2
    elif dang == 2:
        ncoor = 472410*2
    elif dang == 1:
        ncoor = 3716398*2
    b1 = np.zeros((3,ncoor))
    b2 = np.zeros((3,ncoor))
    b3 = np.zeros((3,ncoor))

    strike = np.zeros((ncoor))
    dip = np.zeros((ncoor))
    rake = np.zeros((ncoor))
    the_all = np.zeros((ncoor))
    phi_all = np.zeros((ncoor))
    zeta_all = np.zeros((ncoor))
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
            cosphi=np.cos(rphi)
            sinphi=np.sin(rphi)

            bb3[2] = -costhe
            bb3[0] = -sinthe*sinphi
            bb3[1] = sinthe*cosphi

            for izeta in np.arange(0,int(359.9/dang)): #rake from 0 to <180
                zeta = izeta*dang
                rzeta = zeta/180.0*np.pi
                coszeta = np.cos(rzeta)
                sinzeta = np.sin(rzeta)

                b3[:,irot] = bb3

                b1[0,irot] = coszeta*cosphi+costhe*sinzeta*sinphi
                b1[1,irot] = coszeta*sinphi-costhe*sinzeta*cosphi
                b1[2,irot] = -sinzeta*sinthe
                b3[:,irot],b1[:,irot],junk = FPCOOR(phi,the,zeta,intype='sdr')
                b2[:,irot] = Cross(b3[:,irot],b1[:,irot])

                the_all[irot] = the
                phi_all[irot] = phi
                zeta_all[irot] = zeta
                strike[irot],dip[irot],rake[irot] = FPCOOR(b3[:,irot],b1[:,irot],2,intype = 'fsr')
                irot = irot + 1
    nrot = irot
    return b1[:,:nrot],b2[:,:nrot],b3[:,:nrot],strike[:nrot],dip[:nrot],rake[:nrot]