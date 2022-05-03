
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.interpolate import lagrange

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import punpy

prop = punpy.MCPropagation(1)



class Interpolator:
    def __init__(self, relative=True,method_hr="cubic",method_main="cubic",add_error=True,min_scale=0.3):
        self.relative = relative
        self.method_hr = method_hr
        self.method_main = method_main
        self.add_error = add_error
        self.min_scale=min_scale

    def interpolate_1d_along_example(self,x_i,y_i,x_hr,y_hr,x):
        """
        Method for interpolating between datapoints by following an example.
        The example can come from either models or higher-resolution observations.
        Here the example is assumed to have an unknown normalisation or poor absolute calibration,
        yet the low resolution data has a more precise calibration (and can thus be used to constrain the high-resolution model).

        :param x_i: Independent variable quantity x for the low resolution data
        :type x_i: ndarray
        :param y_i: measured variable quantity y for the low resolution data
        :type y_i: ndarray
        :param x_hr: Independent variable quantity x for the high resolution data
        :type x_hr: ndarray
        :param y_hr: measured variable quantity y for the high resolution data
        :type y_hr: ndarray
        :param x: Independent variable quantity x for which we are trying to obtain the measurand y
        :type x: ndarray
        :param relative: Boolean to indicate whether a relative normalisation (True) or absolute normalisation (False) should be used. Defaults to True.
        :type relative: bool (optional)
        :param method_hr: String to indicate which interpolation method should be used to interpolate between high resolution measurements. Defaults to cubic spline interpolation.
        :type method_hr: string (optional)
        :param method_main: Sting to indicate which interpolation method should be used to interpolate between normalised data (core interpolation step within the approach). Defaults to Gaussian Progress Regression.
        :type method_main: string (optional)
        :param return_uncertainties: Boolean to indicate whether interpolation uncertainties should be returned
        :type return_uncertainties: bool (optional)
        :param add_error: Boolean to indicate whether errors should be added to account for interpolation uncertainty (useful for MC). Defaults to False
        :type add_error: bool (optional)
        :return: The measurand y evaluated at the values x
        :rtype: ndarray
        """

        y_hr_i = interpolate_1d(x_hr,y_hr,x_i,method=self.method_hr,add_error=self.add_error,min_scale=self.min_scale)

        if x == x_hr:
            y_hr_out = y_hr
        else:
            y_hr_out = interpolate_1d(x_hr,y_hr,x,method=self.method_hr,add_error=self.add_error,min_scale=self.min_scale)

        if self.relative:
            y_norm_i = y_i/y_hr_i
        else:
            y_norm_i = y_i-y_hr_i

        y_norm_hr = interpolate_1d(x_i,y_norm_i,x,method=self.method_main,add_error=self.add_error,min_scale=self.min_scale)

        if self.relative:
            y_out = y_norm_hr*y_hr_out
        else:
            y_out = y_norm_hr+y_hr_out

        return y_out

def interpolate(x_i, y_i, x, method="linear",return_uncertainties=False,add_error=False):
    """
    Interpolates data to defined coordinates x

    :param x_i: initial coordinate data of y_i
    :type x_i: numpy.ndarray
    :param y_i: data to interpolate
    :type y_i: numpy.ndarray
    :param x: coordinate data to interpolate y_i to
    :type x: numpy.ndarray
    :param method: interpolation method to be used, defaults to linear
    :type method: string (optional)
    :param return_uncertainties: Boolean to indicate whether interpolation (model error) uncertainties should be calculated and returned. Defaults to False
    :type return_uncertainties: bool (optional)
    :param add_error: Boolean to indicate whether model error should be added to account for interpolation uncertainties (useful in Monte Carlo approaches). Defaults to False
    :type add_error: bool (optional)
    :return: interpolate data
    :rtype: numpy.ndarray
    """
    x_i = np.array(x_i)
    y_i = np.array(y_i)

    y_i = y_i[~np.isnan(x_i)]
    x_i = x_i[~np.isnan(x_i)]

    x_i = x_i[~np.isnan(y_i)]
    y_i = y_i[~np.isnan(y_i)]

    if x_i.ndim==1:
        interpolate_1d(x_i, y_i, x, method, return_uncertainties, add_error)


def interpolate_1d(x_i,y_i,x,method="cubic",u_y_i=None,min_scale=0.3,return_uncertainties=False,add_error=False,return_corr=False):
    """
    Interpolates 1D data to defined coordinates x in 1D

    :param x_i: initial coordinate data of y_i
    :type x_i: numpy.ndarray
    :param y_i: data to interpolate
    :type y_i: numpy.ndarray
    :param x: coordinate data to interpolate y_i to
    :type x: numpy.ndarray
    :param method: interpolation method to be used, defaults to linear
    :type method: string (optional)
    :param return_uncertainties: Boolean to indicate whether interpolation (model error) uncertainties should be calculated and returned. Defaults to False
    :type return_uncertainties: bool (optional)
    :param add_error: Boolean to indicate whether model error should be added to account for interpolation uncertainties (useful in Monte Carlo approaches). Defaults to False
    :type add_error: bool (optional)
    :return: interpolate data
    :rtype: numpy.ndarray
    """
    if method.lower()=="gpr":
        return gaussian_progress_regression(x_i, y_i, x, min_scale=min_scale, u_y_i=u_y_i, return_uncertainties=return_uncertainties,add_error=add_error,return_corr=return_corr)

    else:
        if method.lower() in ["linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"]:
            f_i = interp1d(x_i, y_i, kind=method.lower())
            y= f_i(x).squeeze()

        elif method.lower()=="ius":
            f_i = InterpolatedUnivariateSpline(x_i, y_i)
            y=f_i(x).squeeze()

        elif method.lower()=="lagrange":
            f_i = lagrange(x_i, y_i)
            y = f_i(x).squeeze()

        else:
            raise ValueError(
                "matheo.interpolation: this interpolation method (%s) is not implemented"%(
                    method))

        if return_uncertainties or add_error or return_corr:
            y_unc,y_corr = unc_interpolate_1d(x_i,y_i,x,y,method)
            y_corr[np.where(np.isnan(y_corr))]=0

        if add_error:
            y = prop.generate_sample(y,y_unc,corr_x=y_corr)[:,0].squeeze()

        if return_uncertainties:
            if return_corr:
                return y,y_unc,y_corr
            else:
                return y,y_unc

        else:
            if return_corr:
                return y,y_corr
            else:
                return y


def unc_interpolate_1d(x_i, y_i, x, y, method=None, methods=None):
    if methods is not None:
        return std_interpolation_methods(x_i,y_i,x,y,methods)
    elif method.lower() == "nearest":
        return std_interpolation_methods(x_i,y_i,x,y,["previous", "next", "linear"])
    elif method.lower() == "linear":
        return std_interpolation_methods(x_i,y_i,x,y,["nearest","quadratic"])
    elif method.lower() == "quadratic":
        return std_interpolation_methods(x_i,y_i,x,y,["linear","cubic"])
    elif method.lower() == "cubic":
        return std_interpolation_methods(x_i,y_i,x,y,["linear","quadratic"])
    elif method.lower() == "lagrange":
        return std_interpolation_methods(x_i,y_i,x,y,["quadratic","cubic"])
    else:
        raise ValueError("matheo.interpolation: uncertainties for the model error for this interpolation method (%s) are not yet implemented"%(method))


def std_interpolation_methods(x_i, y_i, x, y, methods=["linear", "quadratic", "cubic"]):
    data=np.zeros((len(x),len(methods)+1))
    for i in range(len(methods)):
        data[:,i] = interpolate_1d(x_i,y_i,x,methods[i]) - y
    return np.std(data, axis=1),np.corrcoef(data)




def gaussian_progress_regression(x_i, y_i, x,u_y_i=None,kernel="RBF",min_scale=0.01,max_scale=10000,return_uncertainties=True,add_error=False,return_corr=False):
    X = np.atleast_2d(x_i).T

    # Observations
    y = np.atleast_2d(y_i).T
    if u_y_i is None:
        alpha = 1e-10  # default value for GaussianProcessRegressor
    else:
        alpha = u_y_i**2

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x = np.atleast_2d(x).T

    # Instantiate a Gaussian Process model
    if kernel=="RBF":
        kernel_mod = C(1.0, (1e-9, 1e9))*RBF(length_scale=0.3, length_scale_bounds=(min_scale, max_scale))
    if kernel=="exp":
        kernel_mod = C(1.0, (1e-9, 1e9))*Matern(length_scale=0.3, length_scale_bounds=(min_scale, max_scale), nu=0.5)
    gp = GaussianProcessRegressor(kernel=kernel_mod,alpha=alpha,n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X,y)

    # print("KERNEL",gp.score(X,y),gp2.score(X,y))
    # print("KERNEL2",gp.log_marginal_likelihood(),gp2.log_marginal_likelihood())
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred,sigma = gp.predict(x,return_std=True)
    y_pred,cov = gp.predict(x,return_cov=True)

    #(sigma,punpy.uncertainty_from_covariance(cov),punpy.correlation_from_covariance(cov))
    if return_uncertainties:
        return y_pred.squeeze(),sigma
    else:
        return y_pred.squeeze()

def interpolate_1d_along_example(x_i,y_i,x_hr,y_hr,x,relative=True,u_y_i=None,
                                 u_y_hr=None,min_scale=0.3,method_hr="cubic",
                                 method_main="gpr",return_uncertainties=False,
                                 add_error=False):
    """
    Method for interpolating between datapoints by following an example.
    The example can come from either models or higher-resolution observations.
    Here the example is assumed to have an unknown normalisation or poor absolute calibration,
    yet the low resolution data has a more precise calibration (and can thus be used to constrain the high-resolution model).

    :param x_i: Independent variable quantity x for the low resolution data
    :type x_i: ndarray
    :param y_i: measured variable quantity y for the low resolution data
    :type y_i: ndarray
    :param x_hr: Independent variable quantity x for the high resolution data
    :type x_hr: ndarray
    :param y_hr: measured variable quantity y for the high resolution data
    :type y_hr: ndarray
    :param x: Independent variable quantity x for which we are trying to obtain the measurand y
    :type x: ndarray
    :param relative: Boolean to indicate whether a relative normalisation (True) or absolute normalisation (False) should be used. Defaults to True.
    :type relative: bool (optional)
    :param method_hr: String to indicate which interpolation method should be used to interpolate between high resolution measurements. Defaults to cubic spline interpolation.
    :type method_hr: string (optional)
    :param method_main: Sting to indicate which interpolation method should be used to interpolate between normalised data (core interpolation step within the approach). Defaults to Gaussian Progress Regression.
    :type method_main: string (optional)
    :param return_uncertainties: Boolean to indicate whether interpolation uncertainties should be returned
    :type return_uncertainties: bool (optional)
    :param add_error: Boolean to indicate whether errors should be added to account for interpolation uncertainty (useful for MC). Defaults to False
    :type add_error: bool (optional)
    :return: The measurand y evaluated at the values x
    :rtype: ndarray
    """
    if return_uncertainties:
        y_hr_i,u_y_hr_i = interpolate_1d(x_hr,y_hr,x_i,method=method_hr,u_y_i=u_y_hr,
                                         min_scale=min_scale,
                                         return_uncertainties=return_uncertainties)

        if x == x_hr:
            y_hr_out = y_hr
        else:
            y_hr_out,u_y_hr_out = interpolate_1d(x_hr,y_hr,x,method=method_hr,
                                                 u_y_i=u_y_hr,min_scale=min_scale,
                                                 return_uncertainties=return_uncertainties)

        if relative:
            y_norm_i = y_i/y_hr_i
            u_y_norm_i = ((u_y_i/y_i)**2+(u_y_hr_i/y_hr_i)**2)**0.5*y_norm_i
        else:
            y_norm_i = y_i-y_hr_i
            u_y_norm_i = ((u_y_i)**2+(u_y_hr_i)**2)**0.5

        y_norm_hr,u_y_norm_hr = interpolate_1d(x_i,y_norm_i,x,method=method_main,
                                               u_y_i=u_y_norm_i,min_scale=min_scale,
                                               return_uncertainties=return_uncertainties)
        y_norm_hr = y_norm_hr.squeeze()

        r = -u_y_hr_out/u_y_norm_hr

        if relative:
            y_out = y_norm_hr*y_hr_out
            u_y_out = ((u_y_norm_hr/y_norm_hr)**2+(
                    u_y_hr_out/y_hr_out+2*r*u_y_norm_hr/y_norm_hr*u_y_hr_out/y_hr_out)**2)**0.5*y_out
        else:
            y_out = y_norm_hr+y_hr_out
            u_y_out = ((u_y_norm_hr)**2+(u_y_hr_out)**2+2*r*u_y_norm_hr*u_y_hr_out)**0.5

        return y_out,u_y_out

    else:
        y_hr_i = interpolate_1d(x_hr,y_hr,x_i,method=method_hr,u_y_i=u_y_hr,
                                min_scale=min_scale)

        if x == x_hr:
            y_hr_out = y_hr
        else:
            y_hr_out = interpolate_1d(x_hr,y_hr,x,method=method_hr,u_y_i=u_y_hr,
                                      min_scale=min_scale)

        if relative:
            y_norm_i = y_i/y_hr_i
        else:
            y_norm_i = y_i-y_hr_i

        y_norm_hr = interpolate_1d(x_i,y_norm_i,x,method=method_main,u_y_i=u_y_i,
                                   min_scale=min_scale)

        if relative:
            y_out = y_norm_hr*y_hr_out
        else:
            y_out = y_norm_hr+y_hr_out


        return y_out