import sys
sys.path.append('../')
# tseuqlib
import tseuqlib.oti_moments as moti

# pyoti
import pyoti.sparse as oti
import pyoti.core as coti

import numpy as np
# import scipy.stats as stats
import matplotlib.pyplot as plt
# from scipy.special  import gamma, factorial2, binom
from tseuqlib.rv_moments import rv_central_moments
from tseuqlib.oti_util import build_rv_joint_moments
from _analytic import analytic_solution as ans
import pandas as pd
from tseuqlib.rv_moments import get_pdf_params

def func(x,alg=oti):
    y = 1
    for i in range(0,len(x)):
        y = y + ((.001*i+1)*x[i])**2
    return y

# Compute central moments of the Taylor series expansion of a 1D polynomial

# path to tseuqlib




# def func(x, n,alg=oti):
#     y = 1
#     for i in range(0,n):
#         y =x[i]*(y +  alg.sin(x[i]))
#     return y




'''
Order(int): 
                  is the order of taylor series approximation to the function
                  
order_of_moments(int): 
                  Defines how many total moments need to be calcuated based on the order of the polynomial and 
                  how many moments are needed of the TSE are needed to be calculated. 
                  Example If we want the mean varience, skewness and kurtosis of the TSE
                  then: order_of_moments = order*4
                  
parameter_distro(dictionary)
                  Defines the distrobutions that the random variables are coming from
rv_parameters(dictionary)
                  Defines the shape and scale parameters for the RV
        '''          
 
h = coti.get_dHelp()

       
lb    = -np.pi
ub    = np.pi

num_random_variables = 7
rv_dist = []
rv_params = []

dist = ['LN', 'LN', 'LN', 'LN', 'T', 'U', 'U']
means = np.array([7.1, 580, 4430, 114, 283, 389, 51e-3])
var    = np.array([.20, .20, .20, .20, .01, .20, .20])

shape, scale = get_pdf_params(dist, means,var*means)
    



order = 3
order_of_moments = order*2




# Generate the Moments vector:
mom = rv_central_moments([np.array(dist), shape, scale], order_of_moments)
mom.compute_central_moments()
rv_mu_joint = build_rv_joint_moments(mom.rv_mu)


shaps = []
sobols = []
sobols_total = []
time = np.arange(0,501,1 )
x = [0]*num_random_variables
for d in range(num_random_variables):
    x[d] = means[d] + oti.e(int(d+1), order=order)


arr = [i for i in range(1, num_random_variables+1)]
for ti in range(0,1, 1):
# for ti in range(0,2, 1):
    y = ans(x, ti = ti, base = oti)
    y = y[0,0]
    
    
    
    
    # compute moments of OTI number
    mu1_oti = moti.oti_expectation(y, rv_mu_joint)
    mu2_oti = moti.oti_central_moment(y, 2, rv_mu_joint)
    # mu5_oti = moti.oti_central_moment(y, 5, rv_mu_joint)
    
    print('Central moments of OTI number:')
    print('Mean:', mu1_oti)
    print('Var: ', mu2_oti)
    #print('mu5: ', mu5_oti)
    print()
    
    
    # print('Calculating varience decomposition')
    # sobol, Var= moti.oti_sobol_indices(y, rv_mu_joint)
    # sobol_main = sobol[1]
    # sobol_interactions = 0
    # for i in range(2,len(sobol)):
    #     sobol_interactions = 1  - sum(sobol_main)
    # sobol_main.append(sobol_interactions)
    # print('Calculating sobol indicies at time {}'.format(ti))
    print('Shaply Values')
    shap =  moti.oti_shaply_values_v2(y, rv_mu_joint)
    sum_shap = sum(shap[0,:])
    correction_factor  = mu2_oti/sum_shap
    shap = correction_factor*shap
    err = abs(sum(shap[0,:]) - mu2_oti)/mu2_oti
    print('Error{}'.format(err))
    shaps.append(shap.flatten())
    # sobols.append(sobol_main)
    # for i in range(0, num_random_variables):
    #     bases = list(set())
    #     sobal_total.append(1 - moti.oti_expectation(oti_selective_expectation(y, mu, basis), rv_mu_joint))

# data = pd.DataFrame(shaps, columns =['k', 'C_p', 'p', 'h_u', 't_inf', 't_w', 'b']) 

# # We need to transform the data from raw data to percentage (fraction)
# data_perc = data.divide(data.sum(axis=1), axis=0)
 
# # Make the plot
# plt.figure(1000)
# plt.stackplot(time,  data_perc["k"],  data_perc["C_p"],  data_perc["p"],data_perc["h_u"],data_perc["t_inf"],data_perc["t_w"],data_perc["b"], labels=['k', 'C_p', 'p', 'h_u', 't_inf', 't_w', 'b'])
# plt.legend(loc='upper left')
# plt.margins(0,0)
# plt.ylabel('Cumulative values of Shapley Values')
# plt.xlabel('Time (s)')
# plt.show()


# data = pd.DataFrame(sobols, columns =['k', 'C_p', 'p', 'h_u', 't_inf', 't_w', 'b', 'interactions']) 

# # We need to transform the data from raw data to percentage (fraction)
# data_perc = data.divide(data.sum(axis=1), axis=0)
 
# # Make the plot
# plt.figure(2000)
# plt.stackplot(time,  data_perc["k"],  data_perc["C_p"],  data_perc["p"],data_perc["h_u"],data_perc["t_inf"],data_perc["t_w"],data_perc["b"],data_perc["interactions"], labels=['k', 'C_p', 'p', 'h_u', 't_inf', 't_w', 'b', 'interactions'])
# plt.legend(loc='upper left')
# plt.margins(0,0)
# plt.ylabel('Cumulative values of Sobol indicies')
# plt.xlabel('Time (s)')
# plt.show()
