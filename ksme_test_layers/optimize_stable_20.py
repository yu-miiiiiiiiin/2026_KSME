"""Conditional 20-unit optimization using saved CFD and recomputed acoustics.

No claim of experimental safety or long-term stability: acoustic trap forces are
local/separable, droplets rigid, and stage mixing is idealized. See generated report.
Run `python3 optimize_stable_20.py --help`.
"""
from pathlib import Path
import argparse, base64, json, math, time, zlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay
from scipy.stats import qmc
import params_layers as P

OUT = Path('results')
Z = np.array([-11.7295,-7.0168,-2.3248,2.3248,7.0168,11.7295])*1e-3
CENTERS = np.column_stack((np.zeros(6),np.zeros(6),Z))
KE = 8.9875517923e9
QP0 = 1.6689750807590623e-15

def decode_vtu(path):
    result = {}
    for item in ET.parse(path).iter('DataArray'):
        s = ''.join(item.text.split())
        n = int(np.frombuffer(base64.b64decode(s[:16]), dtype='<u4')[0])
        nch = 4*((4*(3+n)+2)//3)
        header = np.frombuffer(base64.b64decode(s[:nch]), dtype='<u4')
        packed = base64.b64decode(s[nch:])
        chunks, start = [], 0
        for size in header[3:]:
            chunks.append(zlib.decompress(packed[start:start+int(size)]))
            start += int(size)
        dtype = {'Float64':'<f8','Int64':'<i8','UInt8':'u1'}[item.attrib['type']]
        data = np.frombuffer(b''.join(chunks),dtype=dtype)
        nc = int(item.attrib.get('NumberOfComponents',1))
        result[item.attrib['Name']] = data.reshape(-1,nc) if nc>1 else data
    return result

class SavedFlow:
    """P1 interpolation of exported CFD, sampled on an 80um near-drop grid.

    Delaunay reconstruction can change tetrahedral connectivity; this is a
    postprocessing approximation, explicitly benchmarked against original runs.
    """
    def __init__(self,U,hnear=8e-5):
        self.U=U
        cache=OUT/f'flow_probe_U{U:g}_h{hnear:g}.npz'
        if cache.exists():
            data=np.load(cache); self.axes=[tuple(data[f'g{j}{k}'] for k in range(3)) for j in range(2)]
            self.itp=[RegularGridInterpolator(ax,data[f'v{j}'],bounds_error=False,fill_value=0) for j,ax in enumerate(self.axes)]
            return
        data=decode_vtu(OUT/f'sol_E1_U{U:g}_r0.vtu')
        pts=data['Points']; vel=data['u']
        print('CFD interpolation',U,pts.shape,flush=True)
        interp=LinearNDInterpolator(Delaunay(pts),vel,fill_value=0)
        boxes=[(-.008,.024,0,.007,0,.016,2.5e-4),(-.003,.003,0,.003,0,.014,hnear)]
        self.axes=[]; self.itp=[]; saved={}
        for j,(x0,x1,y0,y1,z0,z1,h) in enumerate(boxes):
            ax=tuple(np.linspace(a,b,int(np.ceil((b-a)/h))+1) for a,b in [(x0,x1),(y0,y1),(z0,z1)])
            xyz=np.stack(np.meshgrid(*ax,indexing='ij'),axis=-1).reshape(-1,3)
            vals=np.concatenate([interp(xyz[k:k+100000]) for k in range(0,len(xyz),100000)])
            inside=np.linalg.norm(xyz[:,None,:]-CENTERS[None,3:,:],axis=2).min(axis=1)<.001
            vals[inside]=0
            vals=vals.reshape(*(len(a) for a in ax),3)
            self.axes.append(ax); self.itp.append(RegularGridInterpolator(ax,vals,bounds_error=False,fill_value=0))
            saved.update({f'g{j}{k}':a for k,a in enumerate(ax)}); saved[f'v{j}']=vals
        np.savez_compressed(cache,**saved)

    def __call__(self,x):
        q=x.copy(); q[:,1:]=abs(q[:,1:])
        v=np.zeros_like(x); v[:,0]=P.inlet_profile(np.clip(x[:,2],-.016,.016),self.U)
        for ax,itp in zip(self.axes,self.itp):
            mask=np.all((q>=np.array([a[0] for a in ax]))&(q<=np.array([a[-1] for a in ax])),axis=1)
            if mask.any(): v[mask]=itp(q[mask])
        v[:,1:]*=np.where(x[:,1:]<0,-1,1)
        return v

def capture(flow,Q,qp,n=2048,seed=1,dt_scale=1.,halfwidth=.006,xin=-.0075,kernel_path=None):
    """Uniform nonoverlapping inlet rectangle; counts all six collectors once.

    Common scrambled Sobol points reduce noise in comparisons. Returned area is
    flux weighted, rather than incorrectly treating nonuniform inflow as uniform.
    Particle acoustic force omitted (known slip <0.005m/s), reported limitation.
    """
    U=flow.U
    pts=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(math.log2(n)))
    x=np.column_stack((np.full(n,xin),(2*pts[:,0]-1)*halfwidth,(2*pts[:,1]-1)*.016))
    weights=P.inlet_profile(x[:,2],U)/U
    v=flow(x); active=np.ones(n,bool); hits=np.zeros(n,bool)
    dt=min(.2*P.TAU_PART,.05*.002/U)*dt_scale
    limit=int(np.ceil(2.6*(.024-xin)/U/dt)); radius=.001005
    def acceleration(x,v):
        rel=flow(x)-v
        re=P.RHO_A*np.linalg.norm(rel,axis=1)*P.D_PART/P.MU_A
        acc=rel/P.TAU_PART*(1+.15*re**.687)[:,None]
        acc[:,2]-=P.G*(1-P.RHO_A/P.RHO_PART)
        if qp:
            rv=x[:,None,:]-CENTERS[None,:,:]
            r=np.maximum(np.linalg.norm(rv,axis=2),radius)
            force=-KE*abs(qp)*Q*np.sum(rv/r[:,:,None]**3,axis=1)
            a=.001
            mag=-KE*qp**2*a**3*(2*r*r-a*a)/(r**3*(r*r-a*a)**2)
            force+=np.sum(mag[:,:,None]*rv/r[:,:,None],axis=1)
            acc+=force/P.M_PART
        return acc
    for step in range(limit):
        ids=np.flatnonzero(active)
        if not len(ids): break
        xa=x[ids]; va=v[ids]
        k1v=acceleration(xa,va); k1x=va
        k2x=va+dt/2*k1v; k2v=acceleration(xa+dt/2*k1x,k2x)
        k3x=va+dt/2*k2v; k3v=acceleration(xa+dt/2*k2x,k3x)
        k4x=va+dt*k3v; k4v=acceleration(xa+dt*k3x,k4x)
        xn=xa+dt/6*(k1x+2*k2x+2*k3x+k4x)
        vn=va+dt/6*(k1v+2*k2v+2*k3v+k4v)
        # Segment-sphere intersection prevents stepping through absorbing surfaces.
        d=xn-xa; rel=CENTERS[None,:,:]-xa[:,None,:]
        t=np.clip(np.sum(rel*d[:,None,:],axis=2)/np.maximum(np.sum(d*d,axis=1)[:,None],1e-30),0,1)
        close=xa[:,None,:]+t[:,:,None]*d[:,None,:]-CENTERS[None,:,:]
        hit=np.min(np.linalg.norm(close,axis=2),axis=1)<=radius
        hits[ids[hit]]=True
        escaped=(xn[:,0]>.024)|(xn[:,0]<xin-.001)|(abs(xn[:,1])>max(.007,halfwidth+.001))|(abs(xn[:,2])>=.016)
        active[ids[hit|escaped]]=False; x[ids]=xn; v[ids]=vn
    area=2*halfwidth*.032
    if kernel_path:
        np.savez_compressed(kernel_path,y=(2*pts[:,0]-1)*halfwidth,z=(2*pts[:,1]-1)*.016,hits=hits,weights=weights)
    A=area*np.mean(hits*weights)
    E=1-np.prod([1-nc*A/P.A_DUCT for nc in (3,2)*4])
    return dict(U=U,Q_pC=Q*1e12,qp_fC=abs(qp)*1e15,n=n,seed=seed,A_col_m2=float(A),removal=float(E),CADR_m3h=float(E*U*P.A_DUCT*3600),hits=int(hits.sum()),unresolved=int(active.sum()),edge_hits=int(np.sum(hits&(abs((2*pts[:,0]-1)*halfwidth)>.9*halfwidth))),halfwidth=halfwidth,dt=dt)

def stability_table(path=OUT/'acoustic_20_ref.npz',reserve=.2):
    """Local harmonic coupled electrostatic model, all 120 charged droplets.

    Requires force utilization <=1-reserve, and generalized stiffness eigenvalue
    >=reserve, plus displacement <=(1-reserve) measured escape displacement.
    Positive stiffness alone is only local stability, not nonlinear certification.
    """
    data=np.load(path); tab={k:data['table'][:,i] for i,k in enumerate(data['keys'])}
    xyz=np.column_stack([tab['xc'],tab['yc'],tab['z_eq']]); n=len(xyz)
    Kdiag=np.stack([tab['k_lat_x'],tab['k_lat_y'],tab['k_ax']],axis=1).reshape(-1)
    assert np.all(Kdiag>0), 'Uncharged nominal array has negative local stiffness'
    H=np.zeros((3*n,3*n)); F=np.zeros((n,3))
    for i in range(n):
        for j in range(i):
            r=xyz[i]-xyz[j]; d=np.linalg.norm(r); u=r/d
            # q expressed in pC for numerical conditioning.
            b=KE*1e-24/d**3*(np.eye(3)-3*np.outer(u,u))
            si=slice(3*i,3*i+3); sj=slice(3*j,3*j+3)
            H[si,si]+=b;H[sj,sj]+=b;H[si,sj]-=b;H[sj,si]-=b
            f=KE*1e-24*r/d**3; F[i]+=f;F[j]-=f
    normalized=H/np.sqrt(Kdiag[:,None]*Kdiag[None,:])
    lam=np.linalg.eigvalsh(normalized).max()
    Qstiff=math.sqrt((1-reserve)/lam)
    out=[]
    for U in [.5,1.,1.5,2.]:
        c=json.load(open(OUT/f'E1_U{U:g}_r0.json'))
        drag=np.array([r['F_drag_N'] for r in c['drag']])
        fdrag=np.zeros_like(F);fdrag[:,0]=np.tile(drag,n//6)
        def check(q,details=False):
            f=F*q*q
            axial=(P.W_DROP+abs(f[:,2]))/tab['F_cap']
            lateral=(fdrag[:,0]+abs(f[:,0]))/tab['F_esc_x']
            side=abs(f[:,1])/tab['F_esc_y']
            stiff=1-q*q*lam
            ok=(max(axial.max(),lateral.max(),side.max())<=1-reserve and stiff>=reserve)
            return dict(axial_utilization=float(axial.max()),lateral_utilization=float(lateral.max()),side_utilization=float(side.max()),stiffness_fraction=float(stiff),pass_screen=bool(ok))
        low,high=0.,min(Qstiff,635.)
        if not check(0)['pass_screen']:
            out.append(dict(U=U,Q_max_pC=0.,**check(0)));continue
        for _ in range(45):
            mid=(low+high)/2
            if check(mid)['pass_screen']:low=mid
            else:high=mid
        # Coupled linear equilibrium shift; check against measured escape radius.
        q=low
        for _ in range(100):
            displacement=np.linalg.solve(np.diag(Kdiag)-q*q*H,(fdrag+q*q*F).reshape(-1)).reshape(-1,3)
            ratio=max(np.max(abs(displacement[:,0])/tab['s_esc_x']),np.max(abs(displacement[:,1])/tab['s_esc_y']))
            if ratio<=1-reserve:break
            q*=.98
        out.append(dict(U=U,Q_max_pC=float(q),displacement_escape_fraction=float(ratio),max_displacement_mm=float(np.linalg.norm(displacement,axis=1).max()*1000),**check(q)))
    result=dict(reserve=reserve,n_droplets=n,Q_stiffness_limit_pC=Qstiff,cases=out)
    (OUT/'stable_20_limits.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['flow','stability','capture']);ap.add_argument('--U',type=float,default=1.5);ap.add_argument('--Q',type=float,default=31.494);ap.add_argument('--qp',type=float,default=1.668975);ap.add_argument('--n',type=int,default=2048);ap.add_argument('--seed',type=int,default=1);ap.add_argument('--dt-scale',type=float,default=1.);ap.add_argument('--h',type=float,default=8e-5);ap.add_argument('--tag',default='trial');ap.add_argument('--halfwidth',type=float,default=.006);ap.add_argument('--kernel',action='store_true');ap.add_argument('--xin',type=float,default=-.0075)
    a=ap.parse_args();t=time.time()
    if a.mode=='stability': print(json.dumps(stability_table(),indent=2));return
    flow=SavedFlow(a.U,a.h)
    if a.mode=='flow':return
    r=capture(flow,a.Q*1e-12,a.qp*1e-15,a.n,a.seed,a.dt_scale,a.halfwidth,xin=a.xin,kernel_path=OUT/f'opt20_{a.tag}_kernel.npz' if a.kernel else None);r['seconds']=time.time()-t;r['hnear']=a.h;r['xin']=a.xin
    (OUT/f'opt20_{a.tag}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

if __name__=='__main__':main()
