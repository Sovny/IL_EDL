# -*- coding:utf-8 -*-
import sys
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import pickle
import scipy.optimize as opt
import scipy.constants as constants
from scipy.integrate import odeint,simps
from scipy.interpolate import CubicSpline

class PBEsolver:
    def __init__(
            self, delta=10, gamma=0.5, V0=0.001, Vmax=100,
            xmax=130, xstep=0.1, evrange=[40,80],evstep=1,
            vstep=1.05, init_points=3,
            epsilon=15,T=300, conc=2400,
            __parameter_input=False, file_dir=None, __q_mode=False,q0=None):
        """
        evrange: the range in which conversion check will be executed.
        input should be percentile. For example, xmax=130 and [0.4,0.8] gives the
        evalation conditoin [520, 104]
        """

        """scale parameter setup"""
        self.epsilon=epsilon
        self.T=T
        self.conc=conc

        """calculation parameter serup"""
        self.delta = delta
        self.gamma = gamma
        self.V0  =V0
        self.Vmax = Vmax
        self.xmax=xmax
        self.xstep=xstep
        self.evrange=evrange
        self.evstep=evstep
        self.vstep = vstep
        self.__init_points=init_points
        """make calc points"""
        #tmp=int(opt.fsolve(lambda n: V0*(vstep)**n-Vmax,1))+1
        #self.cp_volt=[V0*(vstep)**i for i in range(1,tmp+1)]
        """make calc points for x"""
        self.set_cp_x()
        self.set_evrange()
        self.set_vstep()
        ##self.__erange_idx=[self.x[i] for i in cnv_er]
        #self.res_icd = np.array([np.zeros(4) for _ in self.cp_volt])
        self.spline_v1tov0=None #defined in :  def spline()
        self.spline_v1tov2=None
        self.scaled_x=None
        self.scaled_icd=None #defined in : scaler(),it is used to show results
        self.scaler()
    def set_cp_x(self):
        self.__ngrid=self.xmax/self.xstep
        self.cp_x=np.linspace(0,self.xmax,self.__ngrid+1)


    def set_vstep(self):
        tmp=int(opt.fsolve(lambda n: self.V0*(self.vstep)**n-self.Vmax,1))+1
        self.cp_volt=[self.V0*(self.vstep)**i for i in range(1,tmp+1)]
        self.res_icd = np.array([np.zeros(4) for _ in self.cp_volt])

    def set_evrange(self):
        self.__erange_idx=list(map(int,np.array(self.evrange)*self.__ngrid/self.xmax))

    def __set_icd(self):
        k1,k2,A,dc = sy.symbols('k1 k2 A dc')
        x=sy.symbols('x')
        V0 = self.V0
        if self.delta !=0:
            dc = self.delta
            k1 = sy.sqrt(2*dc+1)/(2*dc)
            k2 = sy.sqrt(2*dc-1)/(2*dc)
            A = -k1/k2*(dc-1)/(dc+1)
            v = V0*sy.exp(-k1*x)*(sy.cos(k2*x)+A*sy.sin(k2*x))
            v1_low = v.diff(x,1).subs(x,0).evalf()
            v2_low = v.diff(x,2).subs(x,0).evalf()
            v3_low = v.diff(x,3).subs(x,0).evalf()
            icdl =[V0] + list(map(float,[v1_low,v2_low,v3_low]))
        else:
            icdl =[V0,-V0*0.5,0,0]


        self.icd=icdl
        self.res_icd[0]=icdl
        
    """---------------------- Functions ---------------------"""
    def __f_charge_density(self,g,v):
        tmp=abs(v)
        if  tmp>100:
            return 1/g*tmp/v
        return np.sinh(v) / (1 + 2*g*np.sinh(v/2)**2)

    def plot_charge_density(self,g,v):
        tmp=np.zeros(len(v))
        tmp[np.where(abs(v)>100)]=1

        return np.sinh(v) / (1 + 2*g*np.sinh(v/2)**2)*(1-tmp)+tmp*1/g

    def __f_ode(self,icd,dummy ):
        v0, v1, v2, v3 = icd
        dydt = [v1, v2, v3, (v2 - self.__f_charge_density(self.gamma, v0)) / self.delta**2]
        return dydt
    def __f_ode_d0(self,icd,dummy ):
        v0, v1 = icd
        dydt = [v1,  self.__f_charge_density(self.gamma, v0)]
        return dydt

    def __f_cost_function_v1v2(self,icd,v0):
        tmpr=self.__erange_idx
        if self.delta!=0:
            tmp = [v0, icd[0], icd[1], 0]
            return odeint(self.__f_ode, tmp, self.cp_x)[tmpr[0]:tmpr[1]: self.evstep, 0]
        else:
            tmp = [v0, icd[0] ]
            return odeint(self.__f_ode_d0, tmp, self.cp_x)[tmpr[0]:tmpr[1]: self.evstep, 0]

    
    def spline(self):
        def process(icd):
            def icd_sort(icd,keyv):
                v0=icd[:,0]
                v1=icd[:,1]
                v2=icd[:,2]
                v3=icd[:,3]
                idx=np.argsort(keyv)
                v0=v0[idx]
                v1=v1[idx]
                v2=v2[idx]
                v3=v3[idx]
                return [v0,v1,v2,v3]

            v0,v1,v2,v3=icd_sort(icd,icd[:,1])
            v1tov0=CubicSpline(v1,v0)
            v1tov2=CubicSpline(v1,v2)
            v1tov3=CubicSpline(v1,v3)

            v0,v1,v2,v3=icd_sort(icd,icd[:,0])
            v0tov1=CubicSpline(v0,v1)
            v0tov2=CubicSpline(v0,v2)
            v0tov3=CubicSpline(v0,v3)
            return v1tov0, v1tov2, v1tov3, v0tov1, v0tov2, v0tov3
        res=process(self.res_icd)
        self.spline_v1tov0=res[0]
        self.spline_v1tov2=res[1]
        self.spline_v1tov3=res[2]
        self.spline_v0tov1=res[3]
        self.spline_v0tov2=res[4]
        self.spline_v0tov3=res[5]
    #def __f_cost_function_v1v2(self,icd,v0):
    #    tmp = [v0, icd[0], icd[1], 0]
    #    tmpr=self.__erange_idx

    #    return sum(map(abs,odeint(self.__f_ode, tmp, self.cp_x)[tmpr[0]:tmpr[1]: 1, 0:2]))
        
    """--------------------------------------------------------------"""

    def __initial_calculation(self):
        res= opt.leastsq(self.__f_cost_function_v1v2, self.res_icd[0, 1:3], args = self.cp_volt[0] )[0]
        self.res_icd[0] = [self.cp_volt[0], res[0], res[1],0]
        for i,v in enumerate(self.cp_volt[1: self.__init_points], start = 1):
            res = opt.leastsq(self.__f_cost_function_v1v2, self.res_icd[i-1, 1:3],
                     args = v)[0]
            self.res_icd[i] = [v, res[0], res[1], 0]
            print("V(0)\t\tV'(0)\t\tV''(0)\t\tV'''(0)")
            print(self.res_icd[i])
        return

    def __calc_parameters(self,r=1):
        for i,v in enumerate(self.cp_volt[self.__init_points:], start = self.__init_points):
            res = opt.leastsq(self.__f_cost_function_v1v2,
                    self.res_icd[i-1, 1:3]+r*(self.res_icd[i-1,1:3]-self.res_icd[i-2,1:3]),
                     args = v)[0]
            self.res_icd[i] = [v, res[0], res[1], 0]
            print("V(0)\t\tV'(0)\t\tV''(0)\t\tV'''(0)")
            print(self.res_icd[i])
        return 0

    def reeval_calculation(self):
        self.set_evrange()
        print(self.cp_volt)
        for i,v in enumerate(self.cp_volt):
            res = opt.leastsq(self.__f_cost_function_v1v2, self.res_icd[i, 1:3],
                     args = v)[0]
            self.res_icd[i] = [v, res[0], res[1], 0]
            print(self.res_icd[i])
        return

    def __calc_setup(self):
        self.__set_icd()
        self.__initial_calculation()

    def solve_bvp(self):
        print(self.__erange_idx)
        self.__calc_setup()
        self.__calc_parameters()
        self.spline()
        self.scaler()

    def save_model(self,filename=None):
        if filename==None:
            print("file name input is required")
            return
        np.savetxt("icd_d%.01f_g%.02f.csv"%(self.delta,self.gamma),
                self.res_icd, delimiter=",")

    """
    def plot_sol(self,i):
        if len(self.res_icd)<i-1:
            print("input i is larger than the length of res_icd")
            return
        print(self.res_icd[i])
        print(self.cp_x)
        x=self.cp_x
        y=odeint(self.__f_ode, self.res_icd[i], self.cp_x)[:,0]
        print(y)
        tmp="V0:%.03f V"%(self.res_icd[i,0] )
        plt.subplots_adjust(wspace=0.4)
        plt.subplot(121)
        
        plt.plot(x,y)
        plt.ylim(-y[0],y[0]*1.1)
        plt.xlim(0,100)
        plt.xlabel("x/lambda")
        plt.ylabel("volt [normalize]")
        plt.title(tmp)

        plt.subplot(122)
        y2=self.plot_charge_density(self.gamma,y)
        plt.ylim(-y2[0],y2[0]*1.1)
        plt.xlim(0,100)
        plt.xlabel("x/lambda")
        plt.ylabel("charge density")
        plt.plot(x,y2)
        
        plt.show()
    """
    def plotter(self,x,v0,v1,v2,v3):
        if self.delta!=0:
            y=odeint(self.__f_ode, [v0,v1,v2,v3], self.cp_x)[:,0]
        else:
            y=odeint(self.__f_ode_d0, [v0,v1], self.cp_x)[:,0]
        tmp="V0:%.03f V"%(v0 )
        plt.subplots_adjust(wspace=0.4)
        plt.subplot(121)
        
        plt.plot(x,y)
        plt.ylim(-y[0],y[0]*1.1)
        plt.xlim(0,100)
        plt.xlabel("x/lambda")
        plt.ylabel("volt [normalize]")
        plt.title(tmp)

        plt.subplot(122)
        y2=self.plot_charge_density(self.gamma,y)
        plt.ylim(-y2[0],y2[0]*1.1)
        plt.xlim(0,100)
        plt.xlabel("x/lambda")
        plt.ylabel("charge density")
        plt.plot(x,y2)
        
        plt.show()



    def plot_sol_v1(self,v):
        self.spline()
        x=self.cp_x
        v0=self.spline_v1tov0(v)
        v1=v
        v2=self.spline_v1tov2(v)
        v3=self.spline_v1tov3(v)
        self.plotter(x,v0,v1,v2,v3) 
    def plot_sol_v0(self,v):
        self.spline()
        x=self.cp_x
        v0=v
        v1=self.spline_v0tov1(v)
        v2=self.spline_v0tov2(v)
        v3=self.spline_v0tov3(v)
        self.plotter(x,v0,v1,v2,v3) 

    def load_icds(self,file_name):
        self.res_icd=np.loadtxt(file_name,delimiter=",")

    def plot_icds(self):
        plt.scatter(self.cp_volt,self.res_icd[:,1])
        plt.scatter(self.cp_volt,self.res_icd[:,2])
        plt.show()


    def scaler(self):
        R=constants.R
        ep0=constants.epsilon_0
        e=constants.e
        N_A=constants.N_A
        F=e*N_A

        lmd = np.sqrt(self.epsilon *ep0 * R * self.T/self.conc ) / (F)
        self.scaled_x=self.cp_x*lmd
        self.scaled_icd=self.res_icd*(R*self.T)/F
        self.scaler_x=lmd
        self.scaler_v=(R*self.T)/F

        return

    def show_model_config(self):
        print("delta: %.02f"%self.delta)
        print("gamma: %.02f"%self.gamma)
        print("evrange: [%.01f,%.01f]"%(self.evrange[0],self.evrange[1]))
        print("evstep: %.02f"%self.evstep)
        print("len cp_x: %.02f"%len(self.cp_x))
        print("len cp_volt: %.02f"%len(self.cp_volt))
        print("vstep: %.02f"%self.vstep)
        print("T: %.02f"%self.T)
        print("epsilon: %.02f"%self.epsilon)
        print("conc: %.02f"%self.conc)

    def profile_cd_v1(self,v1):
        v0=self.spline_v1tov0(v1)
        v2=self.spline_v1tov2(v1)
        v3=self.spline_v1tov3(v1)
        #print(self.delta,self.gamma)
        if self.delta!=0:
            return odeint(self.__f_ode, [v0, v1,v2,v3],self.cp_x)[:,0]
        else:
            return odeint(self.__f_ode_d0, [v0, v1],self.cp_x)[:,0]

    def profile_cation(self,v):
        idx=np.where(abs(v)>100)
        tmp=np.zeros(len(v))
        tmp[np.where(v<-100)]=1
        v[idx]=0
        return self.conc*np.exp(-v)/(1.+2*self.gamma*np.sinh(v/2)**2)*(1-tmp)+self.conc/self.gamma*tmp
    def profile_anion(self,v):
        idx=np.where(abs(v)>100)
        tmp=np.zeros(len(v))
        tmp[np.where(v>100)]=1
        v[idx]=0
        
        return self.conc*np.exp(v)/(1.+2*self.gamma*np.sinh(v/2)**2)*(1-tmp)+self.conc/self.gamma*tmp


    def debug(self,evrange):
        #v0,v2=self.spline()
        #y=odeint(self.__f_ode, [v0(v1),v1,v2(v1),0],self.cp_x)
        #plt.plot(self.cp_x,y)
        #plt.ylim(-v0(v1),v0(v1))
        #plt.show()
        self.plot_icds()
        self.reeval_calculation(evrange)
        np.savetxt("reev.csv",self.res_icd,delimiter=",")

class GCmodel:
    def __init__(self,conc=1000, epsilon=80,T=300):
        self.T=T
        self.conc=conc
        self.epsilon=epsilon
        self.R=constants.R
        self.e0=constants.epsilon_0
        self.F=constants.e*constants.N_A
        self.k=(2*self.F**2*conc/(epsilon*self.e0*self.R*self.T))**0.5
        self.v0=0
        self.q=0

    def v(self,x):
        self.k=(2*self.F**2*self.conc/(self.epsilon*self.e0*self.R*self.T))**0.5
        tmp=np.exp(-self.k*x)*np.tanh(self.F*self.v0/(4*self.R*self.T))
        v=np.arctanh(tmp)*4*self.R*self.T/self.F
        return v

    def f_v0(self,q):
        self.k=(2*self.F**2*self.conc/(self.epsilon*self.e0*self.R*self.T))**0.5
        tmp=(8*self.R*self.T*self.epsilon*self.e0*self.conc)**-0.5*q
        v0=np.arcsinh(tmp)*2*self.R*self.T/self.F
        self.v0=v0
        return v0
    def x2v(self,v):
        self.k=(2*self.F**2*self.conc/(self.epsilon*self.e0*self.R*self.T))**0.5
        cf=self.F/(4*self.T*self.R)
        return k**-1*np.log(np.tanh(cf*self.v0)/np.tanh(cf*v))

    def profile_anion(self,v):
        return self.conc*np.exp(v*self.F/(self.R*self.T))
    def profile_cation(self,v):
        return self.conc*np.exp(-v*self.F/(self.R*self.T))



class Kornyshev:
    def __init__(self,conc=2400, epsilon=15,T=300, gamma=1):
        self.T=T
        self.conc=conc
        self.epsilon=epsilon
        self.R=constants.R
        self.e0=constants.epsilon_0
        self.F=constants.e*constants.N_A
        self.k=(2*self.F**2*conc/(epsilon*self.e0*self.R*self.T))**0.5
        self.v0=0
        self.q=0
        self.gamma=gamma

    def f_ode(self,v,x):
        tmp=-(np.log(1-g+g*np.cosh(v*F/(R*T)))*4*R*T*conc/(epsilon*e0*self.gamma))**0.5
        return tmp 


    def v(self,x):
        return odeint(f_ode,self.v0,x) 

    def f_v0(self,q):
        v0= np.sign(q)*R*T/F*np.arccosh((-1+g+np.exp(q**2*g/(4*R*T*epsilon*e0*conc)))/g)
        self.v0=v0
        return v0
    def x2v(self,v):
        self.k=(2*self.F**2*self.conc/(self.epsilon*self.e0*self.R*self.T))**0.5
        cf=self.F/(4*self.T*self.R)
        return k**-1*np.log(np.tanh(cf*self.v0)/np.tanh(cf*v))

    def profile_anion(self,v):
        return self.conc*np.exp(v*self.F/(self.R*self.T))
    def profile_cation(self,v):
        return self.conc*np.exp(-v*self.F/(self.R*self.T))



class EDLSimulator(PBEsolver,GCmodel):
    def __init__(self,ILmodel="Bazant", Wmodel="GC",T=300,Ks=0.36):
        self.Ks=Ks
        models={"Bazant":PBEsolver, "GC": GCmodel, "Kornyshev":Kornyshev}
        self.IL=models[ILmodel](T=T)
        self.W=models[Wmodel](T=T,conc=100,epsilon=80)
        self.concset=1000*10**np.arange(-6,0+0.1,0.1)# mol/m^3
        self.excessAnionIL=np.zeros(len(self.concset))
        self.excessCationIL=np.zeros(len(self.concset))
        self.excessAnionW=np.zeros(len(self.concset))
        self.excessCationW=np.zeros(len(self.concset))
        self.xIL=np.arange(-5e-9,0+0.1e-9,0.1e-9)
        self.xW=np.arange(0,5e-9+0.1e-9,0.1e-9)
        self.F=constants.e*constants.N_A
        self.R=constants.R
        self.T=T

    def run_calc(self,dV=None,i=30,electrolyteW=80,du=0.2):
        e0=constants.epsilon_0
        if i!=None:
            electrolyteW=self.concset[i]
            
        #self.W.conc is a temporal variable
        self.W.conc=(electrolyteW+(electrolyteW**2+1000*4*self.Ks)**0.5)/2
        self.qscaler=self.IL.scaler_x/self.IL.scaler_v/(self.IL.epsilon*e0)
        #print(self.W.conc)
        #print(dV)
        if dV ==None:
            tmp=du+self.R*self.T/self.F*np.log(self.W.conc/1000)+self.R*self.T/2/self.F*np.log(self.Ks)
            print("dV%f"%tmp)
            dV=tmp
            
        #dV=0.08
        def f_eval(q):
            vIL=q*self.qscaler
            tmp=self.IL.scaler_v*self.IL.spline_v1tov0(vIL)
            tmp2=self.W.f_v0(q)
            return (tmp-tmp2)-dV
            #return (tmp-(tmp2-0*0.3e-9*q/(self.W.epsilon*self.W.e0)) )-dV
        
        q_init=-0.01

        res=opt.fsolve(f_eval,q_init)[0]
        #print(f_eval(res))
        #print("q value%f"%res)
        #print("given q:%f, potential W:%f"%(res,self.W.f_v0(res)))
        #print("given q:%f, potential IL:%f"%(res,self.IL.scaler_v*self.IL.spline_v1tov0(res*self.qscaler)))

        self.calc_excess(res,i)
        
        if i==58:
            self.plot_results(res,dV)

    def calc_excess(self,q,i,rangeIL=-5e-9):
        vIL=-self.IL.profile_cd_v1(q*self.qscaler)*self.IL.scaler_v
        #uxIL=-self.IL.cp_x*self.IL.scaler_x
        #idx=np.where(xIL>rangeIL)
        #anionIL=self.IL.profile_anion(vIL/self.IL.scaler_v)
        #cationIL=self.IL.profile_cation(vIL/self.IL.scaler_v)
        #anionW=self.W.profile_anion(q)
        #cationW=self.W.profile_anion(q)
        #excess_anionIL=simps(anionIL[idx]-self.IL.conc,xIL[idx])
        #excess_cationIL=simps(cationIL[idx]-self.IL.conc,xIL[idx])
        anionIL=self.IL.profile_anion(vIL/self.IL.scaler_v)
        cationIL=self.IL.profile_cation(vIL/self.IL.scaler_v)

        vW=self.W.v(self.xW)
        anionW=self.W.profile_anion(vW)
        cationW=self.W.profile_anion(vW)
        excess_anionIL=simps(anionIL[:len(self.xIL)]-self.IL.conc,self.xIL)
        excess_cationIL=simps(cationIL[:len(self.xIL)]-self.IL.conc,self.xIL)
        excess_anionW=simps(anionW[:len(self.xW)]-self.W.conc,self.xW)
        excess_cationW=simps(cationW[:len(self.xW)]-self.W.conc,self.xW)
        print(self.W.conc)

        if i!=None:
            self.excessAnionIL[i]=excess_anionIL
            self.excessCationIL[i]=excess_cationIL
            self.excessAnionW[i]=excess_anionW
            self.excessCationW[i]=excess_cationW
        else:
            return anionIL, cationIL, anionW, cationW
        


    def plot_results(self,q,dV):
        plt.subplot(121)
        vIL=-self.IL.profile_cd_v1(q*self.qscaler)*self.IL.scaler_v
        xIL=-self.IL.cp_x*self.IL.scaler_x
        plt.plot(xIL,vIL)

        xW=np.arange(0,10e-9, 0.1e-9)
        vW=-self.W.v(xW)-dV
        plt.plot(xW,vW)
        print(vIL[0],vW[0]+dV)

        plt.ylim(-dV,0.1)
        plt.xlim(-5e-9,5e-9)

        plt.subplot(122)
        anionIL=self.IL.profile_anion(vIL/self.IL.scaler_v)
        cationIL=self.IL.profile_cation(vIL/self.IL.scaler_v)
        plt.plot(xIL,anionIL)
        plt.plot(xIL,cationIL)
        plt.xlim(-5e-9,5e-9)

        vWplot=vW+dV
        anionW=self.W.profile_anion(vWplot)
        cationW=self.W.profile_cation(vWplot)
        plt.plot(xW,anionW)
        plt.plot(xW,cationW)
        plt.show()
        



def main():
    sime=EDLSimulator()
    with open("./d10g05","rb") as f:
        sime.IL=pickle.load(f)
        sime.IL.conc=2400
        sime.IL.spline()
        sime.IL.scaler()
    for i in range(61):
        sime.run_calc(i=i)
    print(sime.excessAnionIL)
    print(sime.excessAnionW)
    plt.plot(np.log10(sime.concset), sime.excessAnionIL)
    plt.plot(np.log10(sime.concset), sime.excessCationIL)
    plt.plot(np.log10(sime.concset), sime.excessAnionW)
    plt.show()

def prep():
    t=PBEsolver(evrange=[40,70],delta=10,gamma=0.8)
    t.solve_bvp()
    t.plot_sol_v0(0.1)
    with open("./d10g05","wb") as f:
        pickle.dump(t,f)
    with open("./d10g05","rb") as f:
        t=pickle.load(f)
    t.reeval_calculation()
    t.plot_sol_v0(100)

if __name__=="__main__":
    #prep()
    main()




