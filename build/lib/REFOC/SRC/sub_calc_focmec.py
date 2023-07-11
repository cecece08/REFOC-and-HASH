from numba import cuda
import math
from numpy import np

############################################
# Functions for finding neighboring events #
############################################

def geo_dist(lat,lon, lat0, lon0):
# get distance between 
# a list of points (lat,lon) to
# the reference point (lat0,lon0)
# Output: dist (unit: km)
    R=6373.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)	
    dlon = lon_r - lon0_r
    dlat = lat_r - lat0_r
	
    a = np.sin(dlat/2)*np.sin(dlat/2)+np.cos(lat_r)*np.cos(lat0_r)*np.sin(dlon/2)*np.sin(dlon/2)
    c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
	
    dist = R*c
    return dist

def hypo_dist(lat,lon,dep,lat0,lon0,dep0):
    epi_dist = geo_dist(lat,lon,lat0,lon0)
    dep_diff = abs(dep-dep0)
    dist = np.sqrt(epi_dist**2 + dep_diff**2)
    return dist
#########################
# functions for calculating all misfit
#########################
@cuda.jit('void(float32[:,:],float32[:,:],float32[:],float32[:],int32,int32,int32,float32[:],float32[:,:],float32[:,:,:],float32[:,:],float32[:,:],float32[:,:],float32[:],float32[:])')
def CU_FocalAmp_Misfit(p_azi_mc,p_the_mc,sp_amp,p_pol,npsta,nmc,nrot,thetable,phitable,amptable,b1,b2,b3,qmis,nmis):
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    tx = cuda.threadIdx.x
    idx = tx + bx*bw
    pi_val = math.pi
    ntab = 180.0
#    astep = float(1.0/ntab)

    if (idx < nrot):
        irot = idx
        nmis[irot] = 0
        qmis[irot] = 0
        im = 0
        while(im<nmc):
            inmis = 0
            iqmis = 0
            ista = 0
            while(ista<npsta):
                p_a_0 = math.sin(p_the_mc[ista,im]/180*pi_val)*math.cos(p_azi_mc[ista,im]/180*pi_val)
                p_a_1 = math.sin(p_the_mc[ista,im]/180*pi_val)*math.sin(p_azi_mc[ista,im]/180*pi_val)
                p_a_2 = -math.cos(p_the_mc[ista,im]/180*pi_val)

                p_b1 = b1[0,irot]*p_a_0+b1[1,irot]*p_a_1+b1[2,irot]*p_a_2
                p_b3 = b3[0,irot]*p_a_0+b3[1,irot]*p_a_1+b3[2,irot]*p_a_2

                if sp_amp[ista]!= -999:
                    p_proj_0 = p_a_0-p_b3*b3[0,irot]
                    p_proj_1 = p_a_1-p_b3*b3[1,irot]
                    p_proj_2 = p_a_2-p_b3*b3[2,irot]

                    plen = math.sqrt(p_proj_0*p_proj_0+p_proj_1*p_proj_1+p_proj_2*p_proj_2)
                    p_proj_0 = p_proj_0/plen
                    p_proj_1 = p_proj_1/plen
                    p_proj_2 = p_proj_2/plen

                    pp_b1 = b1[0,irot]*p_proj_0+b1[1,irot]*p_proj_1+b1[2,irot]*p_proj_2
                    pp_b2 = b2[0,irot]*p_proj_0+b2[1,irot]*p_proj_1+b2[2,irot]*p_proj_2
                    i = int((p_b3+1)*ntab)
                    if i>2*ntab:
                        i = 0
                    theta = thetable[i]
                    i = int((pp_b2+1)*ntab)
                    j = int((pp_b1+1)*ntab)
                    phi = phitable[i,j]
                    i = int(phi/pi_val*ntab)
                    if i>=2*ntab:
                        i = 0
                    j = int(theta/pi_val*ntab)
                    if j>=ntab:
                        j = 0
                    p_amp = amptable[0,j,i]
                    s_amp = amptable[1,j,i]
                    if p_amp ==0:
                        sp_ratio = 4.0
                    elif s_amp == 0:
                        sp_ratio = -2.0
                    else:
                        sp_ratio = math.log10(4.9*s_amp/p_amp)
                    iqmis = iqmis+abs(sp_amp[ista]-sp_ratio)
                if p_pol[ista]!=0.0:
                    prod = p_b1*p_b3
                    ipol = -1.0
                    if prod>0:
                        ipol = 1.0
                    if ipol != p_pol[ista]:
                        inmis = inmis+1.0
                ista = ista + 1
            if im == 0:
                nmis[irot] = inmis
                qmis[irot] = iqmis
            else:
                if inmis < nmis[irot]:
                    nmis[irot] = inmis
                if iqmis < qmis[irot]:
                    qmis[irot] = iqmis
            im = im + 1

@cuda.jit('void(float32[:,:],float32[:,:],float32[:],float32[:],int32,int32,int32,float32[:],float32[:,:],float32[:],float32[:,:,:],int32,float32[:],float32[:,:],float32[:,:,:],float32[:,:],float32[:,:],float32[:,:],float32[:],float32[:],float32[:],float32[:])')
def CU_FocalAmp_Misfit_Rela(p_azi_mc,p_the_mc,sp_amp,p_pol,npsta,nmc,nrot,rela_azi,rela_the,rela_weight,rela_amp,namp,thetable,phitable,amptable,b1,b2,b3,qmis,nmis,pmis,smis):
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    tx = cuda.threadIdx.x
    idx = tx + bx*bw
    pi_val = math.pi
    ntab = 180.0
#    astep = float(1.0/ntab)

    if (idx < nrot):
        irot = idx
        nmis[irot] = 0
        qmis[irot] = 0
        pmis[irot] = 0
        smis[irot] = 0
        im = 0
        while(im<nmc):
            inmis = 0
            iqmis = 0
            ipmis = 0
            ismis = 0
            ista = 0
            while(ista<npsta):
                p_a_0 = math.sin(p_the_mc[ista,im]/180*pi_val)*math.cos(p_azi_mc[ista,im]/180*pi_val)
                p_a_1 = math.sin(p_the_mc[ista,im]/180*pi_val)*math.sin(p_azi_mc[ista,im]/180*pi_val)
                p_a_2 = -math.cos(p_the_mc[ista,im]/180*pi_val)

                p_b1 = b1[0,irot]*p_a_0+b1[1,irot]*p_a_1+b1[2,irot]*p_a_2
                p_b3 = b3[0,irot]*p_a_0+b3[1,irot]*p_a_1+b3[2,irot]*p_a_2

                if sp_amp[ista]!= -999:
                    p_proj_0 = p_a_0-p_b3*b3[0,irot]
                    p_proj_1 = p_a_1-p_b3*b3[1,irot]
                    p_proj_2 = p_a_2-p_b3*b3[2,irot]

                    plen = math.sqrt(p_proj_0*p_proj_0+p_proj_1*p_proj_1+p_proj_2*p_proj_2)
                    p_proj_0 = p_proj_0/plen
                    p_proj_1 = p_proj_1/plen
                    p_proj_2 = p_proj_2/plen

                    pp_b1 = b1[0,irot]*p_proj_0+b1[1,irot]*p_proj_1+b1[2,irot]*p_proj_2
                    pp_b2 = b2[0,irot]*p_proj_0+b2[1,irot]*p_proj_1+b2[2,irot]*p_proj_2
                    i = int((p_b3+1)*ntab)
                    if i>2*ntab:
                        i = 0
                    theta = thetable[i]
                    i = int((pp_b2+1)*ntab)
                    j = int((pp_b1+1)*ntab)
                    phi = phitable[i,j]
                    i = int(phi/pi_val*ntab)
                    if i>=2*ntab:
                        i = 0
                    j = int(theta/pi_val*ntab)
                    if j>=ntab:
                        j = 0
                    p_amp = amptable[0,j,i]
                    s_amp = amptable[1,j,i]
                    if p_amp ==0:
                        sp_ratio = 4.0
                    elif s_amp == 0:
                        sp_ratio = -2.0
                    else:
                        sp_ratio = math.log10(4.9*s_amp/p_amp)
                    iqmis = iqmis+abs(sp_amp[ista]-sp_ratio)
                if p_pol[ista]!=0.0:
                    prod = p_b1*p_b3
                    ipol = -1.0
                    if prod>0:
                        ipol = 1.0
                    if ipol != p_pol[ista]:
                        inmis = inmis+1.0
                ista = ista + 1

            ista = 0
            while(ista<namp):
                p_a_0 = math.sin(rela_the[ista,im]/180*pi_val)*math.cos(rela_azi[ista]/180*pi_val)
                p_a_1 = math.sin(rela_the[ista,im]/180*pi_val)*math.sin(rela_azi[ista]/180*pi_val)
                p_a_2 = -math.cos(rela_the[ista,im]/180*pi_val)

                p_b1 = b1[0,irot]*p_a_0+b1[1,irot]*p_a_1+b1[2,irot]*p_a_2
                p_b3 = b3[0,irot]*p_a_0+b3[1,irot]*p_a_1+b3[2,irot]*p_a_2
                
                p_proj_0 = p_a_0-p_b3*b3[0,irot]
                p_proj_1 = p_a_1-p_b3*b3[1,irot]
                p_proj_2 = p_a_2-p_b3*b3[2,irot]

                plen = math.sqrt(p_proj_0*p_proj_0+p_proj_1*p_proj_1+p_proj_2*p_proj_2)
                p_proj_0 = p_proj_0/plen
                p_proj_1 = p_proj_1/plen
                p_proj_2 = p_proj_2/plen

                pp_b1 = b1[0,irot]*p_proj_0+b1[1,irot]*p_proj_1+b1[2,irot]*p_proj_2
                pp_b2 = b2[0,irot]*p_proj_0+b2[1,irot]*p_proj_1+b2[2,irot]*p_proj_2
                i = int((p_b3+1)*ntab)
                if i>2*ntab:
                    i = 0
                theta = thetable[i]
                i = int((pp_b2+1)*ntab)
                j = int((pp_b1+1)*ntab)
                phi = phitable[i,j]
                i = int(phi/pi_val*ntab)
                if i>=2*ntab:
                    i = 0
                j = int(theta/pi_val*ntab)
                if j>=ntab:
                    j = 0
                if rela_amp[ista,0,im]!=-999:
                    p_amp = amptable[0,j,i]
                    if p_amp == 0:
                        ipmis = ipmis + 4.0*rela_weight[ista]
                    else:
                        ipmis = ipmis + abs(rela_amp[ista,0,im]-math.log10(p_amp))*rela_weight[ista]
                if rela_amp[ista,1,im]!=-999:
                    s_amp = amptable[1,j,i]
                    if s_amp ==0:
                        ismis = ismis + 4.0*rela_weight[ista]
                    else:
                        ismis = ismis + abs(rela_amp[ista,1,im]-math.log10(s_amp))*rela_weight[ista]
                ista = ista + 1

            if im == 0:
                nmis[irot] = inmis
                qmis[irot] = iqmis
                pmis[irot] = ipmis
                smis[irot] = ismis
            else:
                if inmis < nmis[irot]:
                    nmis[irot] = inmis
                if iqmis < qmis[irot]:
                    qmis[irot] = iqmis
                if ipmis < pmis[irot]:
                    pmis[irot] = ipmis
                if ismis < smis[irot]:
                    smis[irot] = ismis
            im = im + 1
#####################
#     functions for final solution and uncertainty determination
#########################
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

def Mech_Rot(norm1,norm2,slip1,slip2):
    from numpy import arccos,sort,pi,dot
    B1 = Cross(norm1,slip1)
    B2 = Cross(norm2,slip2)
    ip1 = abs(dot(norm1,norm2))
    ip2 = abs(dot(slip1,slip2))
    ip3 = abs(dot(B1,B2))
    if ip1>1:
        ip1 = 1
    if ip2>1:
        ip2 = 1
    if ip3>1:
        ip3=1
    phi=arccos((ip1+ip2+ip3-1)/2)/pi*180
    #print(str(ip1)+' '+str(ip2)+' '+str(ip3)+' '+str(phi))
    if phi > 90:
        ip=sort(np.asarray([ip1,ip2,ip3]))
        phi=arccos((-ip[0]+ip[1]+ip[2]-1)/2)/pi*180
    return phi  

def Mech_Avg(nf,norm1,norm2):

# if only on return it
    if nf == 0:
        return 0,0
    if nf <= 1:
        return norm1[:,0],norm2[:,0]
# find the average normal vector for each plane
# determine which nodal plane of each event corresponding to which
    norm1_avg = norm1[:,0]
    norm2_avg = norm2[:,0]
    ref1 = norm1[:,0]
    ref2 = norm2[:,0]

    for i in np.arange(1,nf):
        temp1 = norm1[:,i]
        temp2 = norm2[:,i]

        rota = Mech_Rot(ref1,temp1,ref2,temp2)
        norm1_avg = norm1_avg + temp1
        norm2_avg = norm2_avg + temp2
    ln_norm1 = np.sqrt(np.sum(norm1_avg**2))
    ln_norm2 = np.sqrt(np.sum(norm2_avg**2))
    if ln_norm1 ==0:
        norm1_avg = norm1_avg
    else:
        norm1_avg = norm1_avg/ln_norm1
    if ln_norm2 == 0:
        norm2_avg = norm2_avg
    else:
        norm2_avg = norm2_avg/ln_norm2
# determine the RMS observed angular difference
# between the average normal vectors and the normal vectors of each mechanism
    avang1 = 0
    avang2 = 0
    for i in np.arange(nf):
        temp1 = norm1[:,i]
        temp2 = norm2[:,i]

#        rota = Mech_Rot(norm1_avg,temp1,norm2_avg,temp2)
        d11 = np.sum(temp1*norm1_avg)
        d22 = np.sum(temp2*norm2_avg)
        d11 = min(1,d11)
        d11 = max(-1,d11)
        d22 = min(1,d22)
        d22 = max(-1,d22)
        a11 = np.arccos(d11)
        a22 = np.arccos(d22)
        avang1 = avang1 + a11**2
        avang2 = avang2 + a22**2
    avang1 = np.sqrt(avang1/nf)
    avang2 = np.sqrt(avang2/nf)
# the average normal vectors may nor be exactly orthoganal
# find the misfit from orthogonal and adjust the vectos to make them orthogonal
# adjust the more poorly constrained plane
    if avang1+avang2 < 0.0001:
        return norm1_avg,norm2_avg
    maxmisf = 0.01
    fract1 = avang1/(avang1+avang2)
    for icount in np.arange(100):
        dot1 = np.sum(norm1_avg*norm2_avg)
        misf = 90 - np.arccos(dot1)/np.pi*180
        if (abs(misf)<=maxmisf):
            return norm1_avg,norm2_avg
        theta1 = misf*fract1/180*np.pi
        theta2 = misf*(1-fract1)/180*np.pi

        temp = norm1_avg
        norm1_avg = norm1_avg - norm2_avg*np.sin(theta1)
        norm2_avg = norm2_avg - temp*np.sin(theta2)
        ln_norm1 = np.sqrt(np.sum(norm1_avg**2))
        ln_norm2 = np.sqrt(np.sum(norm2_avg**2))
        norm1_avg = norm1_avg/ln_norm1
        norm2_avg = norm2_avg/ln_norm2

    return norm1_avg,norm2_avg

def Mech_Prob(nf,norm1in,norm2in,cangle,prob_max):
# if there is only on mechanism, return that mechanism
    norm1 = np.zeros(norm1in.shape)
    norm2 = np.zeros(norm2in.shape)
    rota = np.zeros(nf)
    prob = np.zeros(5)
    #prob[0] = 1
    rms_diff = np.zeros((2,5))
    str_avg = np.zeros(5)
    dip_avg = np.zeros(5)
    rak_avg = np.zeros(5)
    rota = np.zeros(nf)
    if nf <= 1:
        prob[0] = 1
        rms_diff[0,0] = 0
        rms_diff[1,0] = 0
        nsltn = 1
        str_avg[0] = 999
        dip_avg[0] = 999
        rak_avg[0] = 999
        print('mech prob: no enought input focmec')
        return nsltn,str_avg,dip_avg,rak_avg,prob,rms_diff
# find the prefered mechanism and any multiples
    for j in np.arange(nf):
        norm1[:,j] = norm1in[:,j]
        norm2[:,j] = norm2in[:,j]

    nfault = nf
    nc = nf
    for imult in np.arange(5):
        if nc < 1:
            nsltn = imult
            #print('mech prob: nsltn')
            return nsltn,str_avg,dip_avg,rak_avg,prob,rms_diff

# find average mechanism by repeatedly
# 1. throwing out the mechanism with largest angular differen from the average.
# 2. finding the average of the remaining mechanism
# 3. stop when all mechanisms are within cangle of average
        #print('imult:'+str(imult)+' nc:'+str(nc))
        norm1_avg,norm2_avg = Mech_Avg(nc,norm1,norm2)
        print('imult:'+str(imult)+' '+str(nc))#+' '+str(norm1_avg.shape))
       # print(norm1_avg)
       # print(norm2_avg)
        rota = np.zeros(nc)
        for i in np.arange(nc):
            temp1 = norm1[:,i]
            temp2 = norm2[:,i]
            rota[i] = abs(Mech_Rot(norm1_avg,temp1,norm2_avg,temp2))
        index_in = np.where(rota<=cangle)[0]
        index_out = np.where(rota>cangle)[0]
        a = float(len(index_in))
        b = float(nfault)
        prob[imult] = float(a/b)
        #print('imult:'+str(imult)+'  prob:'+str(prob[imult]))
        if imult>1 and prob[imult]<prob_max:
            nsltn = imult
            return nsltn,str_avg,dip_avg,rak_avg,prob,rms_diff

        norm1 = norm1[:,index_out]
        norm2 = norm2[:,index_out]
        nc = len(index_out)
        nf = len(index_out)
# determine the RMS observed angular difference between
# the average normal vectors and normal vectors of each mechanism
        rms_diff[0,imult] = 0
        rms_diff[1,imult] = 0
        for i in np.arange(nfault):
            temp1 = norm1in[:,i]
            temp2 = norm2in[:,i]
            #print(temp1)
            #print(temp2)
            #rota = Mech_Rot(norm1_avg,temp1,norm2_avg,temp2)
            d11 = np.sum(temp1*norm1_avg)
            d22 = np.sum(temp2*norm2_avg)
            d11 = min(d11,1)
            d11 = max(d11,-1)
            d22 = min(d22,1)
            d22 = max(d22,-1)

            a11 = np.arccos(d11)
            a22 = np.arccos(d22)

            rms_diff[0,imult] = rms_diff[0,imult] + a11*a11
            rms_diff[1,imult] = rms_diff[1,imult] + a22*a22

        rms_diff[0,imult] = np.sqrt(rms_diff[0,imult]/nfault)/np.pi*180
        rms_diff[1,imult] = np.sqrt(rms_diff[1,imult]/nfault)/np.pi*180
        str_avg[imult],dip_avg[imult],rak_avg[imult] = FPCOOR(norm1_avg,norm2_avg,2,intype='fsr')

    nsltn = imult - 1

    return nsltn,str_avg,dip_avg,rak_avg,prob,rms_diff

###########################
#  functions for quantifying quality of final solution
#################################
def To_Car(the,phi,r):
# transforms spherical coodinates to cartesian coordinates
    z = -r*np.cos(the/180*np.pi)
    x = r*np.sin(the/180*np.pi)*np.cos(phi/180*np.pi)
    y = r*np.sin(the/180*np.pi)*np.sin(phi/180*np.pi)
    return np.asarray([x,y,z])

def Get_Misf_Amp(npol,p_azi_mc,p_the_mc,sp_ratio,p_pol,str_avg,dip_avg,rak_avg):
    M = np.zeros((3,3))

    strike = str_avg/180*np.pi
    dip = dip_avg/180*np.pi
    rake = rak_avg/180*np.pi
    a = np.zeros(3)
    b = np.zeros(3)
    M[0,0] = -np.sin(dip)*np.cos(rake)*np.sin(2*strike)-np.sin(2*dip)*np.sin(rake)*np.sin(strike)*np.sin(strike)
    M[1,1] = np.sin(dip)*np.cos(rake)*np.sin(2*strike)-np.sin(2*dip)*np.sin(rake)*np.cos(strike)*np.cos(strike)
    M[2,2] = np.sin(2*dip)*np.sin(rake)

    M[0,1] = np.sin(dip)*np.cos(rake)*np.cos(2*strike) + 0.5*np.sin(2*dip)*np.sin(rake)*np.sin(2*strike)
    M[1,0] = M[0,1]

    M[0,2] = -np.cos(dip)*np.cos(rake)*np.cos(strike)-np.cos(2*dip)*np.sin(rake)*np.sin(strike)
    M[2,0] = M[0,2]

    M[1,2] = -np.cos(dip)*np.cos(rake)*np.sin(strike)+np.cos(2*dip)*np.sin(rake)*np.cos(strike)
    M[2,1] = M[1,2]

    bb3,bb1,junk = FPCOOR(strike,dip,rake,intype='sdr')
    bb2 = Cross(bb3,bb1)


    mfrac = 0
    qcount = 0
    stdr = 0
    scount = 0
    mavg = 0
    acount = 0

    for k in np.arange(0,npol):
        p_a = To_Car(p_the_mc[k],p_azi_mc[k],1)
        p_b1 = np.sum(bb1*p_a)
        p_b3 = np.sum(bb3*p_a)

        p_proj = p_a - p_b3*bb3
        plen=np.sqrt(np.sum(p_proj**2))
        p_proj = p_proj/plen

        pp_b1 = np.sum(bb1*p_proj)
        pp_b2 = np.sum(bb2*p_proj)

        phi = np.arctan2(pp_b2,pp_b1)
        theta = np.arccos(p_b3)
        p_amp = abs(np.sin(2*theta)*np.cos(phi))
        wt = np.sqrt(p_amp)

        if (p_pol[k]!=0):
            azi = p_azi_mc[k]/180*np.pi
            toff = p_the_mc[k]/180*np.pi
            a[0] = np.sin(toff)*np.cos(azi)
            a[1] = np.sin(toff)*np.sin(azi)
            a[2] = -np.cos(toff)
            for index in np.arange(3):
                b[index] = np.sum(M[index,:]*a)
            if np.sum(a*b)<0:
                pol = -1
            else:
                pol = 1
            if pol*p_pol[k]<0:
                mfrac = mfrac+wt
            qcount = qcount + wt
            stdr = stdr + wt
            scount = scount + 1
        if (sp_ratio[k]!=-999):
            s1 = np.cos(2*theta)*np.cos(phi)
            s2 = -np.cos(theta)*np.sin(phi)
            s_amp = np.sqrt(s1*s1+s2*s2)
            sp_rat = np.log10(4.9*s_amp/p_amp)
            mavg = mavg+abs(sp_ratio[k]-sp_rat)
            acount = acount + 1
            stdr = stdr + wt
            scount = scount + 1
    #print(qcount)
    if qcount == 0:
        mfrac = 0
    else:
        mfrac = mfrac/qcount
    if acount ==0:
        mavg = 0
    else:
        mavg = mavg/acount
    if scount ==0:
        stdr = 0
    else:
        stdr = stdr/scount
    return mfrac,mavg,stdr

def Get_Misf_Amp_Rela(npol,p_azi_mc,p_the_mc,sp_ratio,p_pol, \
                    rela_azi,rela_the,rela_amp,rela_weight, \
                    str_avg,dip_avg,rak_avg):
    M = np.zeros((3,3))

    strike = str_avg/180*np.pi
    dip = dip_avg/180*np.pi
    rake = rak_avg/180*np.pi
    a = np.zeros(3)
    b = np.zeros(3)
    M[0,0] = -np.sin(dip)*np.cos(rake)*np.sin(2*strike)-np.sin(2*dip)*np.sin(rake)*np.sin(strike)*np.sin(strike)
    M[1,1] = np.sin(dip)*np.cos(rake)*np.sin(2*strike)-np.sin(2*dip)*np.sin(rake)*np.cos(strike)*np.cos(strike)
    M[2,2] = np.sin(2*dip)*np.sin(rake)

    M[0,1] = np.sin(dip)*np.cos(rake)*np.cos(2*strike) + 0.5*np.sin(2*dip)*np.sin(rake)*np.sin(2*strike)
    M[1,0] = M[0,1]

    M[0,2] = -np.cos(dip)*np.cos(rake)*np.cos(strike)-np.cos(2*dip)*np.sin(rake)*np.sin(strike)
    M[2,0] = M[0,2]

    M[1,2] = -np.cos(dip)*np.cos(rake)*np.sin(strike)+np.cos(2*dip)*np.sin(rake)*np.cos(strike)
    M[2,1] = M[1,2]

    bb3,bb1,junk = FPCOOR(strike,dip,rake,intype='sdr')
    bb2 = Cross(bb3,bb1)
    
    mfrac = 0
    qcount = 0
    
    stdr = 0
    stcount = 0

    mavg = 0
    acount = 0

    pavg = 0
    pcount = 0

    savg = 0
    scount = 0

    for k in np.arange(0,npol):
        p_a = To_Car(p_the_mc[k],p_azi_mc[k],1)
        p_b1 = np.sum(bb1*p_a)
        p_b3 = np.sum(bb3*p_a)

        p_proj = p_a - p_b3*bb3
        plen=np.sqrt(np.sum(p_proj**2))
        p_proj = p_proj/plen

        pp_b1 = np.sum(bb1*p_proj)
        pp_b2 = np.sum(bb2*p_proj)

        phi = np.arctan2(pp_b2,pp_b1)
        theta = np.arccos(p_b3)
        p_amp = abs(np.sin(2*theta)*np.cos(phi))
        wt = np.sqrt(p_amp)

        if (p_pol[k]!=0):
            azi = p_azi_mc[k]/180*np.pi
            toff = p_the_mc[k]/180*np.pi
            a[0] = np.sin(toff)*np.cos(azi)
            a[1] = np.sin(toff)*np.sin(azi)
            a[2] = -np.cos(toff)
            for index in np.arange(3):
                b[index] = np.sum(M[index,:]*a)
            if np.sum(a*b)<0:
                pol = -1
            else:
                pol = 1
            if pol*p_pol[k]<0:
                mfrac = mfrac+wt
            qcount = qcount + wt
            stdr = stdr + wt
            stcount = stcount + 1
        if (sp_ratio[k]!=-999):
            s1 = np.cos(2*theta)*np.cos(phi)
            s2 = -np.cos(theta)*np.sin(phi)
            s_amp = np.sqrt(s1*s1+s2*s2)
            sp_rat = np.log10(4.9*s_amp/p_amp)
            mavg = mavg+abs(sp_ratio[k]-sp_rat)
            acount = acount + 1
            stdr = stdr + wt
            stcount = stcount + 1
    for k in np.arange(0,len(rela_azi)):
        p_a = To_Car(rela_the[k],rela_azi[k],1)
        p_b1 = np.sum(bb1*p_a)
        p_b3 = np.sum(bb3*p_a)

        p_proj = p_a - p_b3*bb3
        plen=np.sqrt(np.sum(p_proj**2))
        p_proj = p_proj/plen

        pp_b1 = np.sum(bb1*p_proj)
        pp_b2 = np.sum(bb2*p_proj)

        phi = np.arctan2(pp_b2,pp_b1)
        theta = np.arccos(p_b3)
        p_amp = abs(np.sin(2*theta)*np.cos(phi))
        s1 = np.cos(2*theta)*np.cos(phi)
        s2 = -np.cos(theta)*np.sin(phi)
        s_amp = np.sqrt(s1*s1+s2*s2)
        wtp = np.sqrt(p_amp)
        wts = np.sqrt(s_amp)
        if rela_amp[k,0]!=-999:
            pavg = pavg + abs(rela_amp[k,0]-math.log10(p_amp))*rela_weight[k]
            pcount = pcount + rela_weight[k]
            stdr = stdr + wtp
            stcount = stcount + 1

        if rela_amp[k,1]!=-999:
            savg = savg + abs(rela_amp[k,1]-math.log10(s_amp))*rela_weight[k]
            scount = scount + rela_weight[k]
            stdr = stdr + wts
            stcount = stcount + 1
    #print(qcount)
    if qcount == 0:
        mfrac = 0
    else:
        mfrac = mfrac/qcount
    if acount ==0:
        mavg = 0
    else:
        mavg = mavg/acount
    if stcount ==0:
        stdr = 0
    else:
        stdr = stdr/stcount
    if pcount == 0:
        pavg = 0
    else:
        pavg = pavg/pcount
    if scount == 0:
        savg = 0
    else:
        savg = savg/scount
    return mfrac,mavg,stdr,pavg,savg

def Get_Gap(ntot, p_azi_mc, p_the_mc):
    p2_the = p_the_mc.copy()
    p2_azi = p_azi_mc.copy()

    index = np.where(p2_the>90)[0]
    p2_the[index] = 180-p2_the[index]
    #p2_azi[index] = p_azi_mc[index]-180

    index = np.where(p2_azi < 0)[0]
    p2_azi[index] = p2_azi[index] + 360

    if ntot <= 1:
        magap = 360
        mpgap = 90
        return magap, mpgap
    p2_azi = np.sort(p2_azi[:ntot])
    p2_the = np.sort(p2_the[:ntot])

    magap = -999
    mpgap = -999
    for k in np.arange(1,len(p2_the)):
        if p2_azi[k]-p2_azi[k-1]>magap:
            magap = p2_azi[k]-p2_azi[k-1]
        if p2_the[k]-p2_the[k-1]>mpgap:
            mpgap = p2_the[k]-p2_the[k-1]
    if p2_azi[0]-p2_azi[-1]+360 > magap:
        magap = p2_azi[0]-p2_azi[-1]+360
    if (90-p2_the[-1])>mpgap:
        mpgap = 90-p2_the[-1]
    if p2_the[0]>mpgap:
        mpgap = p2_the[0]
            
    return magap, mpgap

def Get_Gap_Rela(ntot, p_azi_mc, p_the_mc,rela_azi,rela_the):
    p2_the = np.append(p_the_mc[:ntot],rela_the)
    p2_azi = np.append(p_azi_mc[:ntot],rela_azi)

    index = np.where(p2_the>90)[0]
    p2_the[index] = 180-p2_the[index]
    #p2_azi[index] = p_azi_mc[index]-180

    index = np.where(p2_azi < 0)[0]
    p2_azi[index] = p2_azi[index] + 360

    if len(p2_the) <= 1:
        magap = 360
        mpgap = 90
        return magap, mpgap
    p2_azi = np.sort(p2_azi)
    p2_the = np.sort(p2_the)

    magap = -999
    mpgap = -999
    for k in np.arange(1,len(p2_the)):
        if p2_azi[k]-p2_azi[k-1]>magap:
            magap = p2_azi[k]-p2_azi[k-1]
        if p2_the[k]-p2_the[k-1]>mpgap:
            mpgap = p2_the[k]-p2_the[k-1]
    if p2_azi[0]-p2_azi[-1]+360 > magap:
        magap = p2_azi[0]-p2_azi[-1]+360
    if (90-p2_the[-1])>mpgap:
        mpgap = 90-p2_the[-1]
    if p2_the[0]>mpgap:
        mpgap = p2_the[0]

    return magap, mpgap

###########################
#  functions for synthetic amplitude
#################################
def Get_Syn_Amp(p_azi_mc,p_the_mc,str_avg,dip_avg,rak_avg):
    s_amp = np.zeros(len(p_the_mc))
    p_amp = np.zeros(len(p_the_mc))
    
    strike = str_avg/180*np.pi
    dip = dip_avg/180*np.pi
    rake = rak_avg/180*np.pi
    
    bb3,bb1,junk = FPCOOR(strike,dip,rake,intype='sdr')
    bb2 = Cross(bb3,bb1)
    for k in np.arange(0,len(p_the_mc)):
        p_a = To_Car(p_the_mc[k],p_azi_mc,1)
        p_b1 = np.sum(bb1*p_a)
        p_b3 = np.sum(bb3*p_a)
        
        p_proj = p_a - p_b3*bb3
        plen=np.sqrt(np.sum(p_proj**2))
        p_proj = p_proj/plen
        
        pp_b1 = np.sum(bb1*p_proj)
        pp_b2 = np.sum(bb2*p_proj)

        phi = np.arctan2(pp_b2,pp_b1)
        theta = np.arccos(p_b3)
        p_amp[k] = abs(np.sin(2*theta)*np.cos(phi))
        
        s1 = np.cos(2*theta)*np.cos(phi)
        s2 = -np.cos(theta)*np.sin(phi)
        s_amp[k] = np.sqrt(s1*s1+s2*s2)
    return p_amp,s_amp


