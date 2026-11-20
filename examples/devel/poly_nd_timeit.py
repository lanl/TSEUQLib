import sys
sys.path.append('../')
# tseuqlib
import tseuqlib.oti_moments as moti

# pyoti
import pyoti.sparse as oti
import pyoti.core as coti

import numpy as np

import matplotlib.pyplot as plt
import time
from tseuqlib.rv_moments import rv_central_moments
from tseuqlib.oti_util import build_rv_joint_moments
from tseuqlib.rv_moments import get_pdf_params
# from MC_shap import MC_shap 


def func(x,alg=oti):
    y = 1
    for i in range(0,len(x)):
        y = y + ((.001*i+1)*oti.sin(x[i]))**2
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
 

color_palette = ['#1f77b4', '#ff7f0e']       
labels = ['1st Order', '2nd Order']
variables = []
error = []
for k in range(2,3):
    num_vars = []
    for i in range(21,22):
        print('Num Vars: {}'.format(i))
        num_vars.append(i)
        dist = []
        means = np.zeros((i,))
        var = np.zeros((i,))
        num_random_variables = i
        for j in range(0,i):
            dist.append('N')
            means[j] = 10
            var[j] = 1
        
        shape, scale = get_pdf_params(dist, means,var)
            
    
        order = k
        order_of_moments = order*2
        
        
        
        
        # Generate the Moments vector:
        mom = rv_central_moments([np.array(dist), shape, scale], order_of_moments)
        mom.compute_central_moments()
        rv_mu_joint = build_rv_joint_moments(mom.rv_mu)
        
        
        shaps = []
        sobols = []
        sobols_total = []
      
        x = [0]*num_random_variables
        for d in range(num_random_variables):
            x[d] = means[d] + oti.e(int(d+1), order=order)
        y = func(x)
        mu2_oti = moti.oti_central_moment(y, 2, rv_mu_joint)
        st = time.time()
        sobol= moti.oti_sobol_indices(y, rv_mu_joint)
        et = time.time()
        st = time.time()
        # shap_v1 = moti.oti_shaply_values_v1(y, rv_mu_joint)
        shap_v2 = moti.oti_shaply_values_v2(y, rv_mu_joint)
        # sum_shap_2 = sum(shap_v2[0,:])
        # correction_factor  = mu2_oti/sum_shap_2
        # print(correction_factor)
        # shap_v2 = correction_factor*shap_v2
        # # shap_MC  = MC_shap(10**6, i)
        # # variables.append(i)
        # error.append(max(abs(shap_v2 - shap_v1)[0]))
        # print(error)
        
# plt.plot(variables, error)
        
        
        





# time_sobol[0] = time_sobol[0][0:84]
# num_vars = num_vars[0:84]
# np.save('sobal_time.npy', np.array(time_sobol))
# np.save('num_vars.npy',np.array(num_random_variables), )
# for k in range(0,2):
#     plt.figure(1, figsize=(8, 6))  # Adjust the figure size as needed
    
#     # Plot with blue solid line and circular markers
#     plt.plot(num_vars[0:84], time_sobol[k-1], color=color_palette[k-1], markersize=8, label='Time to compute Sobol Indices {} TSE'.format(labels[k-1]))
#     plt.yscale('log')
#     plt.xlabel('Number of Variables', fontsize=14)  # Adjust font size as needed
#     plt.ylabel('Time to Compute All Sobol Indices (seconds)', fontsize=14)  # Adjust font size as needed
#     plt.title('Time to Compute Sobol Indices vs. Number of Variables', fontsize=16)  # Adjust font size as needed
    
#     plt.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines
    
#     plt.legend(fontsize=12)  # Show legend with specified font size
    
#     plt.tight_layout()  # Adjust plot layout to prevent clipping of labels
    
#     plt.show()

# ratio = np.array(time_sobol[-1])/np.array(time_sobol[0])
# plt.figure(2, figsize=(8, 6))  # Adjust the figure size as needed

# # Plot with blue solid line and circular markers
# plt.plot(num_vars, ratio, color=color_palette[0], markersize=8, label='Ratio of time to compute Sobol Indices Quad/Linear TSE')
# plt.yscale('log')
# plt.xlabel('Number of Variables', fontsize=14)  # Adjust font size as needed
# plt.ylabel('Ratio of Time Quadratic/Linear', fontsize=14)  # Adjust font size as needed
# plt.title('Ratio of Time Quadratic/Linear TSE vs. Number of Variables', fontsize=16)  # Adjust font size as needed

# plt.grid(True, linestyle='--', alpha=0.7)  # Add grid with dashed lines

# plt.legend(fontsize=12)  # Show legend with specified font size

# plt.tight_layout()  # Adjust plot layout to prevent clipping of labels

# plt.show()