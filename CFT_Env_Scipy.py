import numpy as np
from numpy import array, inf, vectorize, repeat, power
from numpy import abs as np_abs
from numpy import sum as np_sum
import mpmath as mp
from scipy.special import hyp2f1
from mpmath import fp#, hyp2f1
from numpy.linalg import norm
from numpy import array
from numpy import inf
from itertools import zip_longest



class CFT_Env:

    def __init__(self, s_spec: dict, t_spec: dict, Del_Ext: float, s_Id = 1, t_Id = 1, s_is_t = False):

        self.Del_Ext = Del_Ext
        self.s_Id = s_Id
        self.t_Id = t_Id
        self.s_is_t = s_is_t

        self.s_h_hb_c_spec_dict_list = []
        self.t_h_hb_c_spec_dict_list = []

        self.s_spin_spec_flat = []
        self.t_spin_spec_flat = []

        self.s_del_c_spec_flat = []
        self.t_del_c_spec_flat = []

        for spin_mag_s, spin_mag_t in zip_longest(s_spec, t_spec):

            if spin_mag_s in s_spec:

                self.s_h_hb_c_spec_dict_list += [ 
                                                { 
                                                'h': (s_spec[spin_mag_s][i]['Delta'] + spin_mag_s) / 2, 
                                                'hb':(s_spec[spin_mag_s][i]['Delta'] - spin_mag_s) / 2,
                                                'C': (s_spec[spin_mag_s][i]['C'])
                                                }
                                                for i in range(len(s_spec[spin_mag_s]))
                                                ]

                self.s_spin_spec_flat += ([spin_mag_s] * len(s_spec[spin_mag_s]))

                self.s_del_c_spec_flat += [
                                            [ 
                                            s_spec[spin_mag_s][i]['Delta'],
                                            s_spec[spin_mag_s][i]['C']
                                            ]
                                            for i in range(len(s_spec[spin_mag_s]))
                                            ]


            if spin_mag_t in t_spec:

                self.t_h_hb_c_spec_dict_list += [ 
                                                {
                                                'h': (t_spec[spin_mag_t][i]['Delta'] + spin_mag_t) / 2, 
                                                'hb':(t_spec[spin_mag_t][i]['Delta'] - spin_mag_t) / 2,
                                                'C': (t_spec[spin_mag_t][i]['C'])
                                                }
                                                for i in range(len(t_spec[spin_mag_t]))
                                                ]
                
                self.t_spin_spec_flat += ([spin_mag_t] * len(t_spec[spin_mag_t]))

                self.t_del_c_spec_flat += [
                                            [ 
                                            t_spec[spin_mag_t][i]['Delta'],
                                            t_spec[spin_mag_t][i]['C']
                                            ]
                                            for i in range(len(t_spec[spin_mag_t]))
                                            ]
        
        self.s_del_c_spec_flat = sum(self.s_del_c_spec_flat, [])
        self.t_del_c_spec_flat = sum(self.t_del_c_spec_flat, [])

        if s_is_t:
            self.full_del_c_spec_flat = array(self.s_del_c_spec_flat)
            self.full_spin_spec_flat = array(self.s_spin_spec_flat)
        else:
            self.full_del_c_spec_flat = array(self.s_del_c_spec_flat + self.t_del_c_spec_flat)
            self.full_spin_spec_flat = array(self.s_spin_spec_flat + self.t_spin_spec_flat)
        
        self.state_dim = len(self.full_del_c_spec_flat)
    

    def Full_Spec_List_to_Spec_Dict(self):

        s_spec_dict = {}
        t_spec_dict = {}

        for i, spin in enumerate(self.full_spin_spec_flat):
            
            if i < len(self.s_spin_spec_flat):
            
                if (spin in s_spec_dict) is False and spin is not None:
                    
                    s_spec_dict[spin] = [{'Delta': self.full_del_c_spec_flat[2*i], 'C': self.full_del_c_spec_flat[2*i + 1]}]
                
                elif spin in s_spec_dict and spin is not None:

                    s_spec_dict[spin] += [{'Delta': self.full_del_c_spec_flat[2*i], 'C': self.full_del_c_spec_flat[2*i + 1]}]
            
            else:
            
                if (spin in t_spec_dict) is False and spin is not None:

                    t_spec_dict[spin] = [{'Delta': self.full_del_c_spec_flat[2*i], 'C': self.full_del_c_spec_flat[2*i + 1]}]
                
                elif spin in t_spec_dict and spin is not None:

                    t_spec_dict[spin] += [{'Delta': self.full_del_c_spec_flat[2*i], 'C': self.full_del_c_spec_flat[2*i + 1]}]

        if len(t_spec_dict) == 0:
            t_spec_dict = s_spec_dict.copy()
        
        return s_spec_dict, t_spec_dict
    
    
    def Update_Spec(self, action):

        self.full_del_c_spec_flat += array(action)
        new_s_spec_dict, new_t_spec_dict = self.Full_Spec_List_to_Spec_Dict()

        self.__init__(s_spec = new_s_spec_dict, t_spec = new_t_spec_dict, Del_Ext = self.Del_Ext, s_Id = self.s_Id, t_Id = self.t_Id, s_is_t = self.s_is_t)
    
    def Update_Spec_with_new_spec(self, new_spec):

        self.full_del_c_spec_flat = array(new_spec)
        new_s_spec_dict, new_t_spec_dict = self.Full_Spec_List_to_Spec_Dict()

        self.__init__(s_spec = new_s_spec_dict, t_spec = new_t_spec_dict, Del_Ext = self.Del_Ext, s_Id = self.s_Id, t_Id = self.t_Id, s_is_t = self.s_is_t)
    

    @staticmethod
    def g(h, hb, z, zb):
        h12 = 0
        h34 = 0
        hb12 = 0
        hb34 = 0
        
        output = (1/2 if h == hb else 1) * (
            power(z, h) * power(zb, hb) * (hyp2f1(h - h12, h + h34, 2 * h, z)) * (hyp2f1(hb - hb12, hb + hb34, 2 * hb, zb)) +
            power(z, hb) * power(zb, h) * (hyp2f1(h - h12, h + h34, 2 * h, zb)) * (hyp2f1(hb - hb12, hb + hb34, 2 * hb, z))
                                    )
        
        return fp.mpc(output)


    def s_OPE(self, z, zb):

        res_s = np_sum(array([  power((1 - z)*(1 - zb), self.Del_Ext)  *  data['C'] * self.g(data['h'], data['hb'], z, zb) for data in self.s_h_hb_c_spec_dict_list]))
        res_s += self.s_Id * power((1 - z)*(1 - zb), self.Del_Ext)

        return res_s
    

    def t_OPE(self, z, zb):

        res_t = np_sum(array([  power( z * zb, self.Del_Ext)  *  data['C'] * self.g(data['h'], data['hb'], 1 - z, 1 - zb) for data in self.t_h_hb_c_spec_dict_list]))
        res_t +=  self.t_Id * power(z*zb, self.Del_Ext) 

        return res_t

    def E_Vec(self, pts):
        return array([self.s_OPE(points[0], points[1]) - self.t_OPE(points[0], points[1]) for points in pts])
    
    
    def s_OPE_abs(self, z, zb):

        res_s = np_sum(array([ np_abs( power((1 - z)*(1 - zb), self.Del_Ext) ) * np_abs( data['C'] * self.g(data['h'], data['hb'], z, zb) ) for data in self.s_h_hb_c_spec_dict_list]))
        res_s += np_abs(self.s_Id * power((1 - z)*(1 - zb), self.Del_Ext))

        return res_s
    

    def t_OPE_abs(self, z, zb):

        res_t = np_sum(array([ np_abs( power( z * zb, self.Del_Ext) ) * np_abs( data['C'] * self.g(data['h'], data['hb'], 1 - z, 1 - zb) ) for data in self.t_h_hb_c_spec_dict_list]))
        res_t += np_abs( self.t_Id * power(z*zb, self.Del_Ext) )

        return res_t

    def E_abs(self, pts):
        return np_sum(array([self.s_OPE_abs(points[0], points[1]) + self.t_OPE_abs(points[0], points[1]) for points in pts]))
    
    
    def Reward(self, pts):
        return - norm(self.E_Vec(pts))
    
    def Accuracy(self, pts):
        return -self.Reward(pts)/self.E_abs(pts)

