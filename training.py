from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EventCallback,BaseCallback
from ising import IsingEnv, gen_pts



ising= IsingEnv(gen_pts())
model = SAC("MlpPolicy", ising, learning_rate=5e-4, buffer_size=100000, learning_starts=1000, batch_size=64, \
            tau=0.001, gamma= 0.99, verbose=1, device='cuda', ent_coef=200)
model.learn(total_timesteps=100000, log_interval=10)
model.save("Ising_baselines")
obs = ising.reset('random')
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = ising.step(action) # take a random action
    if i % 10 == 0:
        print(i)
        print("State:", ising.state)
        print("Reward:", rewards)
        ising.render()
ising.close()