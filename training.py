from numpy import conj,linspace,meshgrid,logical_and
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EventCallback,BaseCallback
from ising import IsingEnv

def rho(z):
    return z/(1+(1-z)**(1/2))**2

def llambda(z):
    return abs(rho(z))+abs(rho(1-z))
Nx = 25
Ny = 25
x = linspace(-1,2,Nx)
y = linspace(-2,2,Ny)
xv, yv = meshgrid(x,y)
zv = xv+1j*yv
lambda_c = 0.6
z = zv.flatten()
lz=llambda(z)
z = z[logical_and(logical_and(lz<lambda_c,z.real>=1/2),z.imag>=0)]
Nz = z.size
pts=[[p,conj(p)] for p in z]

ising= IsingEnv(pts)
model = SAC("MlpPolicy", ising, learning_rate=5e-4, buffer_size=100000, learning_starts=1000, batch_size=64, \
            tau=0.001, gamma= 0.99, verbose=1, device='cuda')
model.learn(total_timesteps=100000, log_interval=4)
model.save("Ising_baselines")