from numpy import array, inf
from numpy.random import rand
from numpy.linalg import norm
from mpmath import hyp2f1, fp
from numpy import abs as np_abs
from numpy import conj, linspace, meshgrid, logical_and
import gym
from gym.envs.classic_control import rendering

spins = array([0, 0, 2, 4, 6])
ex_h = [[3/80, 3/80] for i in range(4)]
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
    pts = [[_p, conj(_p)] for _p in z]
    return pts


def param_to_spec(param):
    return [[(param[2*i]+spins[i])/2, (param[2*i]-spins[i])/2, param[2*i+1]]for i in range(5)]


def g(h, hb, z, zb):
    h12 = ex_h[0][0] - ex_h[1][0]
    h34 = ex_h[2][0] - ex_h[3][0]
    hb12 = ex_h[0][1] - ex_h[1][1]
    hb34 = ex_h[2][1] - ex_h[3][1]
    output = (1/2 if h == hb else 1)*(z**h*zb**hb*(hyp2f1(h-h12, h+h34, 2*h, z))*(hyp2f1(hb-hb12, hb+hb34, 2*hb, zb)) +
                                    zb**h*z**hb*(hyp2f1(h-h12, h+h34, 2*h, zb))*(hyp2f1(hb-hb12, hb+hb34, 2*hb, z)))
#     print('g=',output)
    return fp.mpc(output)


def p(h, hb, c, z, zb):
    output = c*(((z-1)*(zb-1))**(1/8)*g(h,hb,z,zb)-z**(1/8)*zb**(1/8)*g(h,hb,1-z,1-zb))
#     print('p=',output)
    return output


def e(spec, pts):
    output= [(sum([p(n[0],n[1],n[2],z[0],z[1]) for n in spec if n[0]>=n[1]]) +
              ((z[0]-1)*(z[1]-1))**(1/8)-z[0]**(1/8)*z[1]**(1/8)) for z in pts]
#     print('e=',output)
    return output


def e_abs(spec, pts):
    output= [(sum(np_abs([p(n[0],n[1],n[2],z[0],z[1]) for n in spec if n[0]>=n[1]])) +
                  ((z[0]-1)*(z[1]-1))**(1/8)-z[0]**(1/8)*z[1]**(1/8)) for z in pts]
#     print('e_abs=',output)
    return output


def A(spec,pts):
    output = norm(e(spec,pts))/e_abs(spec,pts)
#     print('A=',output)
    return output


class IsingEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    def __init__(self, pts):
        self.state=rand(10)
        self.action_space = gym.spaces.Box(array([-.1 for i in range(10)]), array([.1 for i in range(10)]))
        self.observation_space = gym.spaces.Box(array([-inf for i in range(30)]),array([inf for i in range(30)]))
        self.n = 5
        self.best_reward=-inf
        self.pts = pts
        self.viewer = None

    def reset(self):
        self.state=rand(10)
        return self.obs()

    def _next_observation(self):
        return self.obs()

    def step(self, action):
        self._take_action(action)
        done = False
        obs = self._next_observation()
#         spec = [[(obs[2*i]+spins[i])/2, (obs[2*i]-spins[i])/2, obs[2*i+1]]for i in range(self.n)]
        reward = -norm(self.obs())
#         print(reward)
        if reward>self.best_reward :
            self.best_reward = reward
            # print('best_reward=',self.best_reward)
            done=True
        info = {}
        return obs, reward, done, info

    def obs(self):
        return e_abs(param_to_spec(self.state), self.pts)

    def _take_action(self,action):
        self.state+=action

    def _get_accuracy(self):
        return A(param_to_spec(self.state),self.pts)

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1, 1, -1, 1)
            pt = rendering.make_circle(0.01)
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
