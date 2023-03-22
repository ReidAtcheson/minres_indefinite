import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm




seed=23998273
rng=np.random.default_rng(seed)


def make_random_banded(m,diag,bands,p,rng):
    for b in bands:
        assert(b>0)

    A = sp.diags([rng.uniform(-1,1,m) for b in bands],bands,shape=(m,m))
    A = 0.5*(A+A.T)
    d = abs(A)@np.ones(m)
    d = d + diag
    d = d * rng.choice([-1,1],m,p=[1-p,p])
    A = A + sp.diags(d,0,shape=(m,m))
    return A


def solve_with_minres(m,diag,bands,p,rng):
    A=make_random_banded(m,diag,bands,p,rng)
    b=rng.uniform(-1,1,size=m)

    resl=[]
    def callback(xk):
        nonlocal resl
        resl.append(la.norm(b-A@xk))

    neval=0
    def Amatvec(x):
        nonlocal neval
        neval+=1
        return A@x


    x,info=spla.minres(spla.LinearOperator((m,m),matvec=Amatvec),b,callback=callback,tol=1e-12,maxiter=1000)

    return np.log(resl[-1]/resl[0]),neval



m=512
diag=1e-3
bands=[1,2,3,50,100]

nsamples=1000
nats=[]
nevals=[]
ps=[]
for _ in tqdm.tqdm(range(nsamples)):
    p=rng.uniform(0,1)
    nat,neval=solve_with_minres(m,diag,bands,p,rng)
    nats.append(nat)
    nevals.append(neval)
    ps.append(p)



plt.scatter(ps,np.array(nats)/np.array(nevals))
#plt.hist(nats,bins=50)
plt.savefig('nat.png')
