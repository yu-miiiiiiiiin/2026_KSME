"""Finite inter-layer mixing sensitivity using measured inlet capture kernels.

Flux-density diffusion with reflecting side boundaries is a reduced model, not
full-array CFD. No wall removal is counted. Cross-flow electric drift outside
the single-column CFD window and droplet-position changes are not represented.
"""
import argparse,json
import numpy as np
from scipy.ndimage import gaussian_filter
import params_layers as P

def evaluate(kernel_path,sigma_mm=3.95,vertical=True,shifts=None):
    d=np.load(kernel_path);h=.00025
    halfwidth=round(float(max(abs(d['y'].min()),abs(d['y'].max())))/h)*h
    ye=np.arange(-halfwidth,halfwidth+h/2,h);ze=np.arange(-.016,.016+h/2,h)
    samples=np.histogram2d(d['y'],d['z'],bins=(ye,ze))[0]
    hit=np.histogram2d(d['y'][d['hits']],d['z'][d['hits']],bins=(ye,ze))[0]
    kernel=np.divide(hit,samples,out=np.zeros_like(hit),where=samples>0)
    z=(ze[1:]+ze[:-1])/2
    flux=np.broadcast_to(P.inlet_profile(z,1.),(840,128)).copy()
    inlet=flux.sum();rows=[]
    for L,n in enumerate((3,2)*4):
        centers=(np.arange(n)-(n-1)/2)*.07
        if shifts is not None:centers+=shifts[L]*.001
        before=flux.sum()
        for center in centers:
            start=round((center-halfwidth+.105)/h)
            if start<0 or start+len(kernel)>len(flux):raise ValueError('Kernel outside duct')
            flux[start:start+len(kernel)]*=1-kernel
        after=flux.sum();rows.append(float((before-after)/inlet))
        if L<7 and sigma_mm:
            flux=gaussian_filter(flux,(sigma_mm*.001/h,sigma_mm*.001/h if vertical else 0),mode='reflect')
    return dict(removal=float(1-flux.sum()/inlet),layer_removal=rows,sigma_mm=sigma_mm,vertical=vertical,empty_kernel_bins=int(np.sum(samples==0)),shifts_mm=shifts)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kernel');a=p.parse_args()
    out=[evaluate(a.kernel,s,v) for s,v in [(0,False),(2,True),(3.95,False),(3.95,True),(6,True),(10,True)]]
    print(json.dumps(out,indent=2))
