import numpy as np
import Input_parameters as ip

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

#######################################
#      Functions for takeoff angle   #
#######################################
def read_tka_table(vm_dir,tvel_list):
    tka_table=[]
    for tvel in tvel_list:
        tka_table.append(np.loadtxt(ip.vmdir+'/'+tvel+'.tka'))
    return tka_table
#######################################
#      Functions for velocity model   #
#######################################
global para_vel
para_vel = {'nx0':101,# maximum source station distance bins for look-up. table
            'nd0':14, # maximum source depth bins for look-up table
            'nindex':10, #maximum number of verlocity model lookup
            'dep1': 0, #minimum source depth
            'dep2': 39, #maximum source depth
            'dep3': 3, #interval
            'del1': 0, #minimum source-station distance
            'del2': 200, # maximum source-station distance
            'del3':2, #interval
            'pmin':0,#minimum ray parameter for ray tracing
            'nump':9000 #number of rays traced
           }

def LayerTrace(p,h,utop,ubot,imth):

    if h==0:
        dx1=0
        dt1=0
        irtr=-1
        return dx1,dt1,irtr
    u = utop
    y = u-p
    if (y<=0):
        dx1 = 0
        dt1 = 0
        irtr = 0
        return dx1,dt1,irtr
    q = y*(u+p)
    qs = np.sqrt(q)
    #special function needed for integral at top of layer
    if imth == 2:
        y = u+qs
        if p!=0:
            y = y/p
        qr = np.log(y)
    elif imth == 3:
        qr = np.arctan2(qs,p)
    if imth ==1:
        b = -(utop**2-ubot**2)/(2*h)
    elif imth == 2:
        vtop = 1/utop
        vbot = 1/ubot
        b = -(vtop-vbot)/h
    else:
        b = -np.log(ubot/utop)/h

    if b == 0:
        b = 1/h
        etau = qs
        ex = p/qs
        irtr = 1
        dx1 = ex/b
        dtau = etau/b
        dt1 = dtau+p*dx1
        return dx1,dt1,irtr

    if imth ==1:
        etau = -q*qs/3
        ex = -qs*p
    elif imth ==2:
        ex = qs/u
        etau = qr-ex
        if p!=0:
            ex = ex/p
    else:
        etau = qs-p*qr
        ex = qr

    u = ubot
    if u<=p:
        irtr = 2
        dx1 = ex/b
        dtau = etau/b
        dt1 = dtau+p*dx1
        return dx1,dt1,irtr
    irtr = 1
    q = (u-p)*(u+p)
    qs = np.sqrt(q)

    if (imth==1):
        etau = etau+q*qs/3
        ex = ex+qs*p
    elif imth ==2:
        y = u+qs
        z = qs/u
        etau = etau+z
        if p!=0:
            y = y/p
            z = z/p
        qr = np.log(y)
        etau = etau-qr
        ex = ex-z
    else:
        qr = np.arctan2(qs,p)
        etau = etau-qs+p*qr
        ex = ex-qr
    dx1 = ex/b
    dtau = etau/b
    dt1 = dtau+p*dx1

    return dx1,dt1,irtr

def Mk_Table(vmdir,vm_list):

    #set up table
    nray0 = 10001
    deptab = np.zeros(para_vel['nd0'])
    ptab = np.zeros(nray0)
    deltab = np.zeros(nray0)
    delttab = np.zeros(para_vel['nx0'])
    tttab = np.zeros(nray0)
    table = np.zeros((para_vel['nx0'],para_vel['nd0'],para_vel['nindex']))
    z = np.zeros(1000)
    alpha = np.zeros(1000)
    slow = np.zeros(1000)
    depxcor = np.zeros((nray0,para_vel['nd0']))
    depucor = np.zeros((nray0,para_vel['nd0']))
    deptcor = np.zeros((nray0,para_vel['nd0']))
    tt = np.zeros((nray0,para_vel['nd0']))
    xsave = np.zeros(20000)
    tsave = np.zeros(20000)
    psave = np.zeros(20000)
    usave = np.zeros(20000)

    for itab in np.arange(0,len(vm_list)):
        print(vm_list[itab])
        qtempdep2 = para_vel['dep2'] + para_vel['dep3']/20
        ndep = int((qtempdep2-para_vel['dep1'])/para_vel['dep3'])+1
        for idep in np.arange(0,ndep):
            dep = para_vel['dep1'] + para_vel['dep3']*(idep)
            deptab[idep] = dep
        #read velocity model

        fid = open(vmdir+'/'+vm_list[itab])

        num = 0
        for line in fid:
            info = line.split()
            z[num]=(float(info[0]))
            alpha[num] = (float(info[1]))
            num = num+1
        z[num] = z[num-1]
        alpha[num]=alpha[num-1]
        npts = num
        npts_old = npts
        for i in np.arange(npts_old-1,0,-1):
            for idep in np.arange(ndep-1,-1,-1):
                if z[i-1]<=deptab[idep]-0.1 and z[i]>=deptab[idep]+0.1:
                    npts = npts + 1
                    for j in np.arange(npts-1,i,-1):
                        z[j] = z[j-1]
                        alpha[j] = alpha[j-1]
                    z[i] = deptab[idep]
                    frac = (z[i]-z[i-1])/(z[i+1]-z[i-1])
                    alpha[i] = alpha[i-1] + frac*(alpha[i+1]-alpha[i-1])
        slow = 1/alpha
        pmax = slow[0]
        plongcut = slow[npts]
        pstep = (pmax-para_vel['pmin'])/float(para_vel['nump'])

        # do P-wave ray tracing
        npmax = int((pmax+pstep/2-para_vel['pmin'])/pstep)+1
        for np1 in np.arange(0,npmax):
            p = para_vel['pmin'] + pstep*np1
            ptab[np1] = p
            x=0
            t=0
            imth=3
            for idep in np.arange(0,ndep):
                if deptab[idep]==0:
                    depxcor[np1,idep] = 0
                    deptcor[np1,idep] = 0
                    depucor[np1,idep] = slow[0]
                else:
                    depxcor[np1,idep] = -999
                    deptcor[np1,idep] = -999
                    depucor[np1,idep] = -999
            for i in np.arange(0,npts-1):
                if (z[i]>=9999):
                    x = -999/2
                    t = -999/2
                    break
                h = z[i+1] - z[i]
                if h==0:
                    continue
                dx,dt,irtr = LayerTrace(p,h,slow[i],slow[i+1],imth)
                x = x+dx
                t = t+dt
                if irtr==0 or irtr==2:

                    break
                xdeg = x
                tmin = t
                for idep in np.arange(0,ndep):
                    if (abs(z[i+1]-deptab[idep])<0.1):
                        depxcor[np1,idep] = xdeg
                        deptcor[np1,idep] = tmin
                        depucor[np1,idep] = slow[i+1]
            xdeg = 2*x
            tmin = 2*t
            deltab[np1] = xdeg
            tttab[np1] = tmin
        # creat table
        for idep in range(0,ndep):
            icount = 0
            xold = -999
            if (deptab[idep]==0):
                i2 = np
            for i in range(0,np1):
                x2=depxcor[i,idep]
                if x2==-999:
                    continue
                if x2<=xold:
                    continue
                t2=deptcor[i,idep]
                icount = icount+1
                xsave[icount] = x2
                tsave[icount] = t2
                psave[icount] = -ptab[i]
                usave[icount] = depucor[i,idep]
                xold = x2
            i2 = i-1
            for i in np.arange(i2-1,-1,-1):
                if (depxcor[i,idep]==-999):
                    continue
                if deltab[i]==-999:
                    continue
                x2 = deltab[i] - depxcor[i,idep]
                t2 = tttab[i] - deptcor[i,idep]
                icount = icount + 1
                xsave[icount] = x2
                tsave[icount] = t2
                psave[icount] = ptab[i]
                usave[icount] = depucor[i,idep]
                xold = x2
            ncount = icount
            ndel = int((para_vel['del2']-para_vel['del1'])/para_vel['del3'])+1
            for idel in np.arange(0,ndel):
                dell = para_vel['del1']+para_vel['del3']*idel
                delttab[idel] = dell
                tt[idel,idep] = 999
                for i in np.arange(1,ncount):
                    x1 = xsave[i-1]
                    x2 = xsave[i]
                    if x1>dell or x2<dell:
                        continue
                    if psave[i]>0 and psave[i]<plongcut:
                        continue
                    frac = (dell-x1)/(x2-x1)
                    t1 = tsave[i-1]+frac*(tsave[i]-tsave[i-1])
                    if t1<tt[idel,idep]:
                        tt[idel,idep] = t1
                        scr1 = psave[i]/usave[i]
                        angle = np.arcsin(scr1)/np.pi*180
                        if angle<0:
                            angle = -angle
                        else:
                            angle = 180-angle
                        table[idel,idep,itab] = angle
        if delttab[0] == 0:
            for idep in np.arange(0, ndep):
                table[0,idep,itab] = 0

    return table,deptab,delttab

################################################
#      Functions for calculating travel time   #
################################################
def Get_TTS(index,dist,qdep,table,deptab,delttab):
    if qdep<deptab[0]:
        qdep=deptab[0]
    if qdep>deptab[-1]:
        qdep = deptab[-1]
    id1_all = np.where(deptab>qdep)[0]
    if len(id1_all)==0:
        id1 = len(deptab)-2
        id2 = len(deptab)-1
    elif id1_all[0]==0:
        id1=0
        id2=1
    else:
        id1=id1_all[0]-1
        id2=id1_all[0]
    ix_all = np.where(delttab>dist)[0]
    if len(ix_all)==0:
        ix1 = len(delttab)-2
        ix2 = len(delttab)-1
    elif ix_all[0]==0:
        ix1=0
        ix2=1
    else:
        ix1=ix_all[0]-1
        ix2=ix_all[0]
    #print(ix1,ix2,id1,id2,index)
    if table[ix1,id1,index]==0 or table[ix1,id2,index]==0 or \
        table[ix2,id2,index]==0 or table[ix2,id1,index]==0 or delttab[ix2]<dist:
    # Extrapolate to get
        iflag = 1
        xoffmin1 = 999
        xoffmin2 = 999
        ixbest1 = 999
        ixbest2 = 999
        for ix in np.arange(1,len(delttab)):
            if table[ix-1,id1,index]==0 or table[ix,id1,index]==0:
                if table[ix-1,id2,index]==0 or table[ix,id2,index]==0:
                    continue
                xoff=abs((delttab[ix-1]+delttab[ix])/2 - dist)
                if xoff < xoffmin2:
                    xoffmin2 = xoff
                    ixbest2 = ix
        if ixbest1==999 or ixbest2==999:
            iflag = -1
            tt = 999
            return iflag,tt
        xfrac1 = (dist-delttab[ixbest1-1])/(delttab[ixbest1]-delttab[ixbest1-1])
        t1 = table[ixbest1-1,id1,index]
        t2 = table[ixbest1,id1,index]
        tt1 = t1 + xfrac1*(t2-t1)
        xfrac2 = (dist-delttab[ixbest2-1])/(delttab[ixbest2]-delttab[ixbest2-1])
        t1 = table[ixbest2-1,id2,index]
        t2 = table[ixbest2,id2,index]
        tt2 = t1+xfrac2*(t2-t1)
        dfrac = (qdep-deptab[id1])/(deptab[id2]-deptab[id1])
        tt = tt1 + dfrac*(tt2-tt1)
    else:
        iflag = 0
        xfrac = (dist-delttab[ix1])/(delttab[ix2]-delttab[ix1])
        t1 = table[ix1,id1,index]+xfrac*(table[ix2,id1,index]-table[ix1,id1,index])
        t2 = table[ix1,id2,index]+xfrac*(table[ix2,id2,index]-table[ix1,id2,index])
        dfrac = (qdep-deptab[id1])/(deptab[id2]-deptab[id1])
        tt = t1 + dfrac*(t2-t1)
    return iflag,tt