"""Nonlinear 3D equilibrium and full coupled stiffness verification.

Rigid point-scatterer Gor'kov model, prescribed saved-CFD drag, pair Coulomb
forces. Does not include acoustic rescattering, droplet deformation or turbulence.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import acoustics_layers as AC
import params_layers as P

KE=8.9875517923e9

class Field:
    def __init__(self,path='results/acoustic_20_ref.npz'):
        d=np.load(path);self.d=d
        self.t={k:d['table'][:,i] for i,k in enumerate(d['keys'])}
        self.origin=np.stack([self.t['xc'],self.t['yc'],self.t['z_eq']],axis=1)
    def potential(self,x,h=2e-5):
        offsets=np.concatenate([np.zeros((1,3)),h*np.eye(3),-h*np.eye(3)])
        pts=(x[:,None,:]+offsets[None,:,:]).reshape(-1,3)
        p=AC.pressure(pts,self.d['pos'],self.d['nrm'],self.d['phase'],float(self.d['P0']),chunk=256).reshape(len(x),7)
        grad=(p[:,1:4]-p[:,4:7])/(2*h)
        kap,f1,f2=AC.contrast(P.RHO_W,P.C_W)
        return P.V_DROP*(f1*.25*kap*abs(p[:,0])**2-f2*.375*P.RHO_A*np.sum(abs(grad)**2,axis=1)/(P.OMEGA*P.RHO_A)**2)
    def force_stiffness(self,x,h=3e-5):
        off=[np.zeros(3)];off.extend(h*np.eye(3));off.extend(-h*np.eye(3))
        for a,b in [(0,1),(0,2),(1,2)]:
            for sa,sb in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                o=np.zeros(3);o[a]=sa*h;o[b]=sb*h;off.append(o)
        val=self.potential((x[:,None,:]+np.array(off)[None,:,:]).reshape(-1,3)).reshape(len(x),19)
        f=-(val[:,1:4]-val[:,4:7])/(2*h)
        blocks=np.zeros((len(x),3,3))
        for a in range(3):blocks[:,a,a]=(val[:,a+1]-2*val[:,0]+val[:,a+4])/h**2
        for k,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
            j=7+4*k;blocks[:,a,b]=blocks[:,b,a]=(val[:,j]-val[:,j+1]-val[:,j+2]+val[:,j+3])/(4*h*h)
        K=np.zeros((3*len(x),3*len(x)))
        for i in range(len(x)):K[3*i:3*i+3,3*i:3*i+3]=blocks[i]
        return f,K

def coulomb(x,Q):
    n=len(x);F=np.zeros_like(x);H=np.zeros((3*n,3*n))
    for i in range(n):
        for j in range(i):
            r=x[i]-x[j];d=np.linalg.norm(r);u=r/d
            f=KE*Q*Q*r/d**3;F[i]+=f;F[j]-=f
            b=KE*Q*Q/d**3*(np.eye(3)-3*np.outer(u,u))
            si=slice(3*i,3*i+3);sj=slice(3*j,3*j+3)
            H[si,si]+=b;H[sj,sj]+=b;H[si,sj]-=b;H[sj,si]-=b
    return F,H

def verify(U,Q_pC,load=1.,h=3e-5,tag='eq',maxiter=20,drive=1.,initial=None):
    field=Field();x=field.origin.copy() if initial is None else np.array(json.load(open(initial))['coordinates_m']);n=len(x)
    data=json.load(open(f'results/E1_U{U:g}_r0.json'))
    ext=np.zeros_like(x);ext[:,0]=np.tile([r['F_drag_N'] for r in data['drag']],n//6);ext[:,2]=-P.W_DROP
    history=[];converged=False
    for i in range(maxiter):
        fac,Kac=field.force_stiffness(x,h);fac*=drive**2;Kac*=drive**2;fel,H=coulomb(x,Q_pC*1e-12)
        F=fac+load*(ext+fel);K=Kac-load*H
        mineig=float(np.linalg.eigvalsh(K).min());res=float(np.max(np.linalg.norm(F,axis=1)))
        history.append(dict(iter=i,residual_uN=res*1e6,min_stiffness_N_m=mineig))
        print(tag,history[-1],flush=True)
        if res<1e-8 and mineig>0:converged=True;break
        shift=np.linalg.solve(K,F.reshape(-1)).reshape(-1,3)
        largest=np.linalg.norm(shift,axis=1).max()
        shift*=min(1.,.00025/max(largest,1e-30))
        if mineig<=0 and i>3:break
        x+=shift
        if np.max(np.linalg.norm(x-field.origin,axis=1))>.004:break
    result=dict(U=U,Q_pC=Q_pC,load_factor=load,drive=drive,h=h,converged=converged,history=history,max_shift_mm=float(np.linalg.norm(x-field.origin,axis=1).max()*1000),min_stiffness_N_m=mineig,coordinates_m=x.tolist())
    Path(f'results/opt20_{tag}.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--U',type=float,default=.5);p.add_argument('--Q',type=float,default=30.);p.add_argument('--load',type=float,default=1.);p.add_argument('--drive',type=float,default=1.);p.add_argument('--h',type=float,default=3e-5);p.add_argument('--tag',default='eq');p.add_argument('--initial');a=p.parse_args()
    verify(a.U,a.Q,a.load,a.h,a.tag,drive=a.drive,initial=a.initial)
