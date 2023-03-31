# Calculate variance figure 8 paper
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from xmitgcm import open_mdsdataset

outdir = '/data/SO2/sio-kramosmusalem/exp06_512x612x100_ORL_SVB/01_SVB_febTS_output/'
outdir2 = '/data/SO2/sio-kramosmusalem/exp06_512x612x100_ORL/01_noSVB_febTS/'

outdirB = '/data/SO2/sio-kramosmusalem/exp06_512x612x100_ORL_SVB/02_SVB_barotropic_output/'
outdir2B = '/data/SO2/sio-kramosmusalem/exp06_512x612x100_ORL/02_noSVB_barotropic/'

outdirA = '/data/SO2/sio-kramosmusalem/exp06_512x612x100_ORL_SVB/04_SVB_augTS_output/'
outdir2A = '/data/SO2/sio-kramosmusalem/exp06_512x612x100_ORL/04_noSVB_augTS/'

levels = [1,   2,  3,  4,  5,  6,  7,  8,  9, 10, 
          11, 12, 13, 14, 15, 16, 17,
          18, 19, 20, 21, 22, 23, 24, 25,
          26, 27, 28, 29, 30, 31,
          32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
          45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,              
          58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 
          74, 79, 84, 89, 94, 99,]
ds = open_mdsdataset(outdir, prefix=['eta','dynVars'], levels=levels)
ds2 = open_mdsdataset(outdir2, prefix=['eta','dynVars'], levels=levels)

dsB = open_mdsdataset(outdirB, prefix=['eta','dynVars'], levels=levels)
ds2B = open_mdsdataset(outdir2B, prefix=['eta','dynVars'], levels=levels)

dsA = open_mdsdataset(outdirA, prefix=['eta','dynVars'], levels=levels)
ds2A = open_mdsdataset(outdir2A, prefix=['eta','dynVars'], levels=levels)

nx = 512
ny = 612
dt = 600
z1 = 25 # Zl[25] = -99 m
z2 = 37 # Zl[37] = -197.2 m
z3 = 45 # Zl[45] = -292.2 m
z4 = 55 # Zl[55] = -508.2 m
levs = [z1,z2,z3,z4]
days = [144-1,288-1,(144*3)-1,(144*4)-1,(144*5)-1] # time index day 1, 2, ...
nlevs = len(levs) # calc variance at 4 depth levels
ndays = len(days) # calc variance after day 1, day 2, ... day 5
time = np.arange(0,720)*600

# centers mask
hFacC = ds2['hFacC'][:]
hfac = np.ma.masked_values(hFacC, 0)
mask = np.ma.getmask(hfac)

# centers mask bathy with bay
hFacCSVB = ds['hFacC'][:]
hfacSVB = np.ma.masked_values(hFacCSVB, 0)
maskSVB = np.ma.getmask(hfacSVB)

def get_var(ds, ds2, tini, tend, mask, zz, time, dt=600):
    ''' Calculate the vertical velocity variance through a time slice at a certain depth level.
    INPUT
    ds: ds from bay run
    ds2: ds from no bay run
    tini: initial time index of slice
    tend: final time index of slice
    mask: 3D mask at centers
    zz: depth level
    time: time array
    dt: time step between time records (default is 600 seconds)

    RETURNS
    var1: 2D variance for bay run at depth level zz
    var1_nob: 2D variance for no bay run at depth level zz
    var1_diff: 2D varince difference between bay and no bay runs at depth level zz
    '''
    mask_ext = np.expand_dims(mask[zz,:,:],0)
    mask_ext = mask_ext + np.zeros_like(ds.variables['WVEL'][tini:tend,zz,...])
    Wmean = np.nanmean(np.ma.masked_array(ds.variables['WVEL'][tini:tend,zz,...],
                                          mask=mask_ext),axis=0)
    Wmean_ext = np.expand_dims(Wmean,0)
    Wmean_ext = Wmean_ext + np.zeros_like(ds.variables['WVEL'][tini:tend,zz,...])

    masked_W = np.ma.masked_array(ds.variables['WVEL'][tini:tend,zz,...],mask=mask_ext)
    var1 = dt*np.nansum((masked_W-Wmean_ext)**2,axis=0)/(time[tend]-time[tini])

    Wmean_nob = np.nanmean(np.ma.masked_array(ds2.variables['WVEL'][tini:tend,zz,...],
                                          mask=mask_ext),axis=0)
    Wmean_nob_ext = np.expand_dims(Wmean_nob,0)
    Wmean_nob_ext = Wmean_nob_ext + np.zeros_like(ds2.variables['WVEL'][tini:tend,zz,...])

    masked_W_nob = np.ma.masked_array(ds2.variables['WVEL'][tini:tend,zz,...],mask=mask_ext)
    var1_nob = dt*np.nansum((masked_W_nob-Wmean_nob_ext)**2,axis=0)/(time[tend]-time[tini])

    Wdif = (masked_W-Wmean_ext)-(masked_W_nob-Wmean_nob_ext)
    var1_diff = dt*np.nansum((Wdif)**2,axis=0)/(time[tend]-time[tini])

    return(var1, var1_nob, var1_diff)

## W variance at different depth levels

tini = 0
# base (febTS)
var_SVB = np.empty((ndays,nlevs,ny,nx))
var_NB = np.empty((ndays,nlevs,ny,nx))
var_dif = np.empty((ndays,nlevs,ny,nx))

# baro
var_SVB_B = np.empty((ndays,nlevs,ny,nx))
var_NB_B = np.empty((ndays,nlevs,ny,nx))
var_dif_B = np.empty((ndays,nlevs,ny,nx))

# augTS
var_SVB_A = np.empty((ndays,nlevs,ny,nx))
var_NB_A = np.empty((ndays,nlevs,ny,nx))
var_dif_A = np.empty((ndays,nlevs,ny,nx))

for tt,ii in zip(days,range(ndays)):
    for zz,kk in zip(levs,range(nlevs)):
        # febTS
        var_SVB[ii,kk,...],var_NB[ii,kk,...],var_dif[ii,kk,...] = get_var(ds,ds2,tini,tt,mask,zz,time,dt=600)
        print(ii,kk)

np.savez('wvar_2d_febTS_', var_SVB=var_SVB, var_NB=var_NB, var_dif=var_dif)

for tt,ii in zip(days,range(ndays)):
    for zz,kk in zip(levs,range(nlevs)):
        # baro
        var_SVB_B[ii,kk,...],var_NB_B[ii,kk,...],var_dif_B[ii,kk,...] = get_var(dsB,ds2B,tini,tt,mask,zz,time,dt=600)
        print(ii,kk)

np.savez('wvar_2d_baro_', var_SVB_B=var_SVB_B, var_NB_B=var_NB_B, var_dif_B=var_dif_B)

for tt,ii in zip(days,range(ndays)):
    for zz,kk in zip(levs,range(nlevs)):
        # augTS
        var_SVB_A[ii,kk,...],var_NB_A[ii,kk,...],var_dif_A[ii,kk,...] = get_var(dsA,ds2A,tini,tt,mask,zz,time,dt=600)
        print(ii,kk)

np.savez('wvar_2d_augTS_', var_SVB_A=var_SVB_A, var_NB_A=var_NB_A, var_dif_A=var_dif_A)

## Area-weighted w variance

mean_var_dif = np.empty((ndays,nlevs))
mean_var_dif_B = np.empty((ndays,nlevs))
mean_var_dif_A = np.empty((ndays,nlevs))

for ii in range(ndays):
    for zz,kk in zip(levs,range(nlevs)):
        masked_area = np.ma.masked_array(ds2.rA,mask=mask[zz,:,:])
        mean_var_dif[ii,kk] = np.nansum(var_dif[ii,kk,...]*masked_area)/np.nansum(masked_area)
        mean_var_dif_B[ii,kk] = np.nansum(var_dif_B[ii,kk,...]*masked_area)/np.nansum(masked_area)
        mean_var_dif_A[ii,kk] = np.nansum(var_dif_A[ii,kk,...]*masked_area)/np.nansum(masked_area)
        print(ii,kk)
np.savez('mean_wvar_dif',mean_var_dif=mean_var_dif,mean_var_dif_B=mean_var_dif_B,mean_var_dif_A=mean_var_dif_A)


