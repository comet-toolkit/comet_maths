"""Tests for classname module"""

"""___Built-In Modules___"""
from matheo.interpolation.interpolation import interpolate_1d,interpolate_1d_along_example, Interpolator
"""___Third-Party Modules___"""
import unittest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

"""___NPL Modules___"""
# import here
import punpy

"""___Authorship___"""
__author__ = "Pieter De Vis"
__created__ = "01/02/2021"
__maintainer__ = "Pieter De Vis"
__email__ = "pieter.de.vis@npl.co.uk"
__status__ = "Development"

prop = punpy.MCPropagation(1)

def function1(x):
    return 15*x-20

def function2(x):
    """The function to predict."""
    return x * np.sin(x*10)


class TestInterpolation(unittest.TestCase):
    def test_interpolation_1d(self):
        xi=np.arange(0,3.,0.2)
        yi=function2(xi)
        u_yi=0.05*yi
        yi = prop.generate_sample(yi,u_yi,corr_x="rand")[:,0]

        x=[0.33333,0.666666,1,1.33333,1.66666,2,2.3333]
        y=interpolate_1d(xi,yi,x,method="cubic")
        y2=interpolate_1d(xi,yi,x,method="quadratic")
        y,u_y,corr_y=interpolate_1d(xi,yi,x,method="cubic",add_error=False,return_uncertainties=True,return_corr=True)
        npt.assert_allclose(y,y2,rtol=2*np.max(u_y/y))

        xx=np.arange(0,2.5,0.01)
        xgpr = np.atleast_2d(xx).T
        yy,u_yy = interpolate_1d(xi,yi,xx,method="gpr",u_y_i=u_yi,min_scale=0.3,add_error=False,return_uncertainties=True,return_corr=True)
        print(yy.shape,xx.shape,u_yy.shape)

        plt.plot(xx,function2(xx),"b")
        plt.plot(xx,interpolate_1d(xi,yi,xx,method="cubic"),"r:")
        plt.plot(xx,interpolate_1d(xi,yi,xx,method="quadratic"),"g-")
        plt.plot(xx,yy,"m")
        plt.fill_between(xx,yy-1.9600*u_yy,(yy+1.9600*u_yy),alpha=0.25,
            fc="y",ec="None",label="95% confidence interval",lw=0)
        plt.plot(xi,yi,"bo")
        plt.errorbar(x,y,yerr=u_y,fmt="ro",ls='none')
        plt.show()

    def test_interpolation_1d_along_example(self):
        xi = np.arange(0,3.,0.2)
        yi = function2(xi)
        u_yi = 0.01*yi
        yi = prop.generate_sample(yi,u_yi,corr_x="rand").squeeze()
        x_HR = np.arange(0,3.,0.036)
        y_HR = function2(x_HR)
        u_y_HR = 0.9*np.ones_like(y_HR)
        corr_y_HR = np.ones((len(y_HR),len(y_HR)))  #+ 0.05 * np.eye(len(y_HR))
        y_HR = prop.generate_sample(y_HR,u_y_HR,corr_x="syst")[:,0]

        xx = np.arange(0,2.5,0.01)

        plt.plot(xx,function2(xx),"b")
        plt.plot(xi,yi,"ro")
        plt.plot(x_HR,y_HR,"go")
        plt.plot(xx,interpolate_1d(xi,yi,xx,method="cubic"),"r:")
        plt.plot(xx,interpolate_1d(x_HR,y_HR,xx,method="cubic"),"g:")
        plt.plot(xx,interpolate_1d_along_example(xi,yi,x_HR,y_HR,xx,method_hr="cubic",
                                                 relative=False),"g-.")
        plt.show()

    def test_interpolation_1d_along_example_unc(self):
        np.random.seed(123456)

        xi = np.arange(0,3.,0.25)
        yi = function2(xi)
        u_yi = 0.01*yi
        yi = prop.generate_sample(yi,u_yi,corr_x="rand").squeeze()
        x_HR = np.arange(0,3.,0.09)
        y_HR = function2(x_HR)
        u_y_HR_syst = 0.9*np.ones_like(y_HR)
        u_y_HR_rand = 0.02*y_HR
        cov_y_HR = (punpy.convert_corr_to_cov(np.ones((len(y_HR),len(y_HR))),u_y_HR_syst)
                    + punpy.convert_corr_to_cov(np.eye(len(y_HR)), u_y_HR_rand))
        corr_y_HR = punpy.correlation_from_covariance(cov_y_HR)
        u_y_HR    = punpy.uncertainty_from_covariance(cov_y_HR)

        y_HR = prop.generate_sample(y_HR,u_y_HR,corr_x=corr_y_HR)[:,0]

        xx = np.arange(0.1,2.5,0.02)
        y_gpr,u_y_gpr=interpolate_1d(xi,yi,xx,method="gpr",u_y_i=u_yi,min_scale=0.3,return_uncertainties=True)
        y_hr_gpr=interpolate_1d_along_example(xi,yi,x_HR,y_HR,xx,relative=False,method_main="gpr",method_hr="gpr",min_scale=0.3)
        #y_hr_gpr,u_y_hr_gpr=interpolate_1d_along_example(xi,yi,x_HR,y_HR,xx,relative=False,method_main="gpr",method_hr="gpr",u_y_i=u_yi,u_y_hr=u_y_HR,min_scale=0.3)
        y_hr=interpolate_1d_along_example(xi,yi,x_HR,y_HR,xx,relative=False,method_main="cubic",method_hr="cubic",min_scale=0.3)


        mcprop = punpy.MCPropagation(100,parallel_cores=4)

        # inp = Interpolator(relative=False,method_main="gpr",method_hr="gpr",
        #                    min_scale=0.3)
        # u_y_hr_gpr = mcprop.propagate_random(inp.interpolate_1d_along_example,
        #                                  [xi,yi,x_HR,y_HR,xx],
        #                                  [None,u_yi,None,u_y_HR,None],
        #                                  corr_x=[None,"rand",None,corr_y_HR,None])

        inp2 = Interpolator(relative=False,method_main="cubic",method_hr="cubic",min_scale=0.3)
        u_y_hr = mcprop.propagate_random(inp2.interpolate_1d_along_example,
                                         [xi,yi,x_HR,y_HR,xx],[None,u_yi,None,u_y_HR,None],
                                         corr_x=[None,"rand",None,corr_y_HR,None])

        plt.plot(xx,function2(xx),"b",label="True line")
        plt.plot(xi,yi,"ro", label="low-res data")
        plt.plot(x_HR,y_HR,"go", label="high-res data")
        plt.plot(xx,interpolate_1d(xi,yi,xx,method="cubic"),"r:", label="cubic spline interpolation")
        plt.plot(xx,y_gpr,"c:", label="GPR interpolation")
        plt.plot(xx,y_hr_gpr,"g-.", label="GPR interpolation with HR example")
        plt.plot(xx,y_hr,"m-.",label="cubic spline interpolation with HR example")
        plt.fill_between(xx,y_gpr-1.9600*u_y_gpr,(y_gpr+1.9600*u_y_gpr),alpha=0.25,fc="c",ec="None",
                         label="95% confidence interval",lw=0)
        # plt.fill_between(xx,y_hr_gpr-1.9600*u_y_hr_gpr,(y_hr_gpr+1.9600*u_y_hr_gpr),alpha=0.25,fc="g",ec="None",
        #                  label="95% confidence interval",lw=0)
        plt.fill_between(xx,y_hr-1.9600*u_y_hr,(y_hr+1.9600*u_y_hr),alpha=0.25,fc="m",ec="None",
                         label="95% confidence interval",lw=0)
        plt.legend(ncol=2, prop={'size': 6})
        plt.xlim([0.1,2.5])
        #plt.show()
        plt.savefig("interpolation_test.png",bbox_inches="tight")




if __name__ == "__main__":
    unittest.main()
