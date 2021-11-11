from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EventCallback,BaseCallback
from ising import IsingEnv, gen_pts



ising= IsingEnv(gen_pts())
model = SAC("MlpPolicy", ising, learning_rate=5e-4, buffer_size=100000, learning_starts=1000, batch_size=64, \
            tau=0.001, gamma= 0.99, verbose=1, device='cuda', ent_coef=200)
# model.learn(total_timesteps=1, log_interval=10)
# model.save("Ising_baselines")
for i in range(1000):
    print(i)
    print(ising.state)
    ising.render()
    ising.step(model.action_space.sample()) # take a random action
ising.close()