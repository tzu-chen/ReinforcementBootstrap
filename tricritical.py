from numpy import array, inf, vectorize, repeat, power, maximum, zeros_like
from numpy.random import rand
from numpy.linalg import norm
# from scipy.special import hyp2f1
from mpmath import hyp2f1, fp
from numpy import abs as np_abs
from numpy import sum as np_sum
from numpy import conj, linspace, meshgrid, logical_and
from functools import cache
import gym
from gym.envs.classic_control import rendering

spins = array([0, 0, 2, 2, 2, 4, 6])
ex_h = array([[7/16, 7/16] for i in range(4)])
accuracy_threshold = 10**-3


def rho(z):
    return z/(1+(1-z)**(1/2))**2


def llambda(z):
    return abs(rho(z))+abs(rho(1-z))


def gen_pts():
    Nx = 25
    Ny = 25
    x = linspace(-1, 2, Nx)
    y = linspace(-2, 2, Ny)
    xv, yv = meshgrid(x, y)
    zv = xv+1j*yv
    lambda_c = 0.6
    z = zv.flatten()
    lz = llambda(z)
    z = z[logical_and(logical_and(lz < lambda_c, z.real >= 1/2), z.imag >= 0)]
    # Nz = z.size
    pts = array([[_p, conj(_p)] for _p in z])
    return pts


def param_to_spec(param):
    # print(param)
    _A = param[::2]
    _B = param[1::2]
    output = array([(_A+spins)/2, (_A-spins)/2, _B]).transpose().flatten()
    # print(output)
    return output
    # return array([[(param[2*i]+spins[i])/2, (param[2*i]-spins[i])/2, param[2*i+1]]for i in range(5)])

@cache
def g(h, hb, z, zb):
    # h12 = ex_h[0][0] - ex_h[1][0]
    # h34 = ex_h[2][0] - ex_h[3][0]
    # hb12 = ex_h[0][1] - ex_h[1][1]
    # hb34 = ex_h[2][1] - ex_h[3][1]
    h12 = 0
    h34 = 0
    hb12 = 0
    hb34 = 0
    output = (1/2 if h == hb else 1)*(z**h*zb**hb*(hyp2f1(h-h12, h+h34, 2*h, z))*(hyp2f1(hb-hb12, hb+hb34, 2*hb, zb)) +
                                    zb**h*z**hb*(hyp2f1(h-h12, h+h34, 2*h, zb))*(hyp2f1(hb-hb12, hb+hb34, 2*hb, z)))
#     print('g=',output)
    return fp.mpc(output)
    # return output


def p(h, hb, c, z, zb):
    output = c*(power(((z-1)*(zb-1)),7/8)*g(h,hb,z,zb) - power(z,7/8)*power(zb,7/8)*g(h,hb,1-z,1-zb))
#     print('p=',output)
    return output

vec_p = vectorize(p, excluded=['z', 'zb'])


def e(spec, pts):
    _A = vec_p(spec[:,0], spec[:,1], spec[:,2], pts[:,0], pts[:,1])
    _B = ((pts[:,0]-1)*(pts[:,1]-1))**(7/8)-pts[:,0]**(7/8)*pts[:1]**(7/8)
    output = _A + repeat(_B, len(spec))
#     output= array([(vec_p(spec[:,0], spec[:,1], spec[:,2], z[0], z[1]) +
#               ((z[0]-1)*(z[1]-1))**(1/8)-z[0]**(1/8)*z[1]**(1/8)) for z in pts])
#     print('e=',output)
    return output


def e_abs(spec, pts):
    # _A = vec_p(spec[:, 0], spec[:, 1], spec[:, 2], pts[:, 0], pts[:, 1])
    # _B = ((pts[:, 0] - 1) * (pts[:, 1] - 1)) ** (1 / 8) - pts[:, 0] ** (1 / 8) * pts[:1] ** (1 / 8)
    # output = np_abs(np_sum(_A + repeat(_B, len(spec)), axis=1))
    # print(spec)
    output= np_abs(array([(np_sum(vec_p(spec[::3], spec[1::3], spec[2::3], z[0], z[1])) +
                  ((z[0]-1)*(z[1]-1))**(7/8)-z[0]**(7/8)*z[1]**(7/8)) for z in pts]))
#     print('e_abs=',output)
    return output

def A(spec,pts):
    output = norm(e(spec,pts))/e_abs(spec,pts)
#     print('A=',output)
    return output


class TriCriticalIsingEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    def __init__(self, pts):
        self.state=rand(10)
        self.action_space = gym.spaces.Box(array([-10 for i in range(14)]), array([10 for i in range(14)]))
        self.observation_space = gym.spaces.Box(array([-inf for i in range(30)]),array([inf for i in range(30)]))
        self.n = 7
        self.best_reward=-inf
        self.best_state = None
        self.pts = pts
        self.viewer = None
        # print("Best Possible Reward: ", -norm(e_abs(param_to_spec(array([4,2.44141*10**-4,1,0.25,2,0.015625,4,2.19727*10**-4,6,1.36239*10**-5])),self.pts)))
    def reset(self, mode='last_best'):
        if self.best_state is None or mode == 'random':
            self.state=rand(14)
        else:
            self.state = self.best_state
        return self.obs()

    def _next_observation(self):
        return self.obs()

    def step(self, action):
        self._take_action(action)
        done = False
        obs = self._next_observation()
#         spec = [[(obs[2*i]+spins[i])/2, (obs[2*i]-spins[i])/2, obs[2*i+1]]for i in range(self.n)]
        reward = -norm(obs)
        # print(reward)
        if reward>self.best_reward :
            self.best_reward = reward
            self.best_state = self.state
            print('best_reward=',self.best_reward)
            done=True
        info = {}
        return obs, reward, done, info

    def obs(self):
        return e_abs(param_to_spec(self.state), self.pts)

    def _take_action(self,action):
        self.state = action
        self.state = maximum(zeros_like(self.state), self.state)
        for i in range(self.n):
            if self.state[2 * i] < spins[i]:
                self.state[2 * i] = spins[i]
            if self.state[2 * i] > 6.5:
                self.state[2 * i] = 6.5

    def _get_accuracy(self):
        return A(param_to_spec(self.state),self.pts)

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2, 2, -2, 2)
            pt = rendering.make_circle(0.05)
            pt.set_color(1, 0, 0)
            self.trans = rendering.Transform()
            pt.add_attr(self.trans)
            self.viewer.add_geom(pt)
        self.trans.set_translation(self.state[0], self.state[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
