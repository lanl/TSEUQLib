import pyoti.sparse as oti
import pyoti.core as coti

# Set pyoti to print all coefficients.
oti.set_printoptions(terms_print=-1, float_format=".15g")

import numpy as np
np.set_printoptions(linewidth=120)

# path to tseuqlib
import sys
sys.path.append('../')

# tseuqlib
import tseuqlib as tuq 
import tseuqlib.rv_moments as rv
import tseuqlib.oti_moments as moti
import tseuqlib.oti_util as uoti

# Second order polynomial:
def funct(x):
    """
    Equation 2.92 in 
    
    INPUTS:
    - xi: OTI number.
    """
    
    return x1*alg.sin(x2)

h = coti.d_helper()

# ---------------------------------------------------------------------------------


extra_bases = []

# Random variable parameters
rv_pdf_name = np.array(['K']*4) # PDF of each variable
rv_a = np.array([1,1,2,2]) # shape parameter of each variable
rv_b = np.array([3,8,6,4]) # scale parameter of each variable

n_dim = len(rv_a) # number of random variables (dimensions)

# store random variable parameters in rv_params as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_stdev]
rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])

# order of Taylor series expansion
tse_order = 1


# random variable(s) with nominal values at the mean and perturbed with OTI imaginary directions
x = [0]*n_dim
for d in range(n_dim):
    x[d] = rv_params[3][d] + oti.e(int(d+1), order=tse_order)
#print('x =', x)


# create OTI number
y =  funct(x)
print(' Function evaluated at OTI-perturbed inputs:')
print('tse =', y)
print(" ---------------------------------------------------")
print()


# generate moments of the random variables
rv_moments_order = int(tse_order*4) #TODO: generalize for moments past 12th order

# compute central moments of the random variables
# rv moments object
rv_moments = rv.rv_central_moments(rv_params, rv_moments_order)
print('rv_moments =', rv_moments)
input('ii')
# compute moments
rv_moments.compute_central_moments()
print(rv_moments.rv_mu)



print(" ---------------------------------------------------")
print(" A total of 4 variables are considered with a       ")
print(" Kumaraswamy distribution. ")
print()
print(" mean values: ", rv_moments.rv_mean)
print(" central moments: ",)
# Print moment list.
for i in range(len(rv_moments.rv_mu)):
    print(f' - mu_{i+1}:', rv_moments.rv_mu[i])
print()
print(" Joint distribution built from the moments: (uncomment lines 79-81)")
# # for i in range(len(rv_mu_joint)):
# #     print(rv_mu_joint[i])
# # # end for
print()
print(" ---------------------------------------------------")
print()



# Build the joint distribution from uncorrelated variables.
rv_mu_joint = uoti.build_rv_joint_moments(rv_moments.rv_mu)


# create oti_moments object
oti_mu = moti.tse_uq(n_dim, rv_mu_joint, extra_bases=extra_bases)


# Expectation
mu_y   = oti_mu.expectation(y)
print('mu_y =', mu_y)

# variance
mu2     = oti_mu.central_moment(y, 2)
print('mu2 =', mu2)

# Third central moment
mu3 = oti_mu.central_moment(y, 3)
print('mu3 =', mu3)

# Fourth central moment
mu4 = oti_mu.central_moment(y, 4)
print('mu4 =', mu4)

'''
print( " Stats. results from OTI. Compare to Table 2.3 Pg 39 ")
print(f" Expected Value:   {mu_y.real:10.3f}")
print(f" Total Variance:   {mu2.real:10.3f}")
print(f" central_moment 3: {mu3.real:10.3f}")
print(f" central_moment 4: {mu4.real:10.3f}")
print()
'''
print(" ---------------------------------------------------")
print()

# Compute Sobol
sobol, Vy_hdmr, f_hdmr = oti_mu.sobol_indices(y)

Shap = oti_mu.shapley_values_v2(y)[0]

print(Shap)
input('ii')


print('sobol =')
for i in range(len(sobol)):
    print(f" -> Order --  {i}:")
    print('real sob:   ', [sobol[i][j] for j in range(len(sobol[i]))] )
    print('real Vi :   ', [Vy_hdmr[i][j] for j in range(len(Vy_hdmr[i]))] )
    print('real fi :   ', [f_hdmr[i][j] for j in range(len(f_hdmr[i]))] )
Vi = [Vy_hdmr[1][j] for j in range(len(Vy_hdmr[1]))]
Vij = [Vy_hdmr[2][j] for j in range(len(Vy_hdmr[2]))]
Vijk = [Vy_hdmr[3][j] for j in range(len(Vy_hdmr[3]))]
Si = [sobol[1][j] for j in range(len(Vy_hdmr[1]))]
Sij = [sobol[2][j] for j in range(len(Vy_hdmr[2]))]
Sijk = [sobol[3][j] for j in range(len(Vy_hdmr[3]))]
input('ii')

# analytical solutions of metrics (precomputed for the example in Section 3.1)
ev_an  = 209949406/675675
mu2_an = 400885535046969013/7761123995625
mu3_an = 16706254.3029196
mu4_an = 15315376784.8466

Vi_an =  [914603454068/326064375, 17169878229461417/10043807523750, 2443557721082767/198684774288, 29265383540220103/913073411250]#[2804.97817054686, 1709.49893144137, 12298.6662155617, 32051.5121562422] # [2804.97817054685, 1709.49893144332, 12298.6662155643, 32051.5121562420]
Vij_an = [743757176927/33817533750, 221786684023/1377295920, 3885725843299/40175230095, 4693801933757/11272511250, 114169056362554/456536705625, 486419013137/265621356] #[21.9932412121491, 161.03052423422196, 96.71944211738446, 416.39363489248126, 250.0764012058571, 1831.249642229268]
Vijk_an = [958844/3006003, 13844/16875, 829639771/135270135, 13274236336/3652293645] # [0.31897640073080424, 0.820385167155699, 6.133207244650748, 3.634493050980268]

Si_an = [Vi_an[i]/mu2_an for i in range(len(Vi_an))]
Sij_an = [Vij_an[i]/mu2_an for i in range(len(Vij_an))]
Sijk_an = [Vijk_an[i]/mu2_an for i in range(len(Vijk_an))]

print('Si_an =', Si_an)
print('Sij_an =', Sij_an)
print('Sijk_an =', Sijk_an)

def compute_shapley_values_for_four_variables(V1, V2, V3, V4,
                                              V12, V13, V14, V23, V24, V34,
                                              V123, V124, V134, V234, V1234):
    """
    Compute Shapley values for 4 variables given partial variances including interaction terms.

    Args:
        V1, V2, V3, V4: Variances contributed by variables 1 to 4 individually.
        V12, V13, V14, V23, V24, V34: Interaction variances between pairs of variables.
        V123, V124, V134, V234: Interaction variances among triplets of variables.
        V1234: Interaction variance among all four variables.

    Returns:
        shapley_values (dict): Shapley values for each variable.
        total_variance (float): Total variance.
    """
    total_variance = (V1 + V2 + V3 + V4 +
                      V12 + V13 + V14 + V23 + V24 + V34 +
                      V123 + V124 + V134 + V234 +
                      V1234)
    n_var = 4 
    import math
    Sh1 = (1/n_var)*( 1*math.comb(n_var-1,0)**-1*(V1) + 
        math.comb(n_var-1,1)**-1*(V1+V12) + 
        math.comb(n_var-1,1)**-1*(V1+V13) + 
        math.comb(n_var-1,1)**-1*(V1+V14) + 
        math.comb(n_var-1,2)**-1*(V1+V12+V13+V123) + 
        math.comb(n_var-1,2)**-1*(V1+V12+V14+V124) + 
        math.comb(n_var-1,2)**-1*(V1+V13+V14+V134) +
        math.comb(n_var-1,3)**-1*(V1+V12+V13+V14+V123+V124+V134+V1234) 
        ) # + /
    
    Sh2 = (1/n_var)*( 1*math.comb(n_var-1,0)**-1*(V2) + 
        math.comb(n_var-1,1)**-1*(V2+V12) + 
        math.comb(n_var-1,1)**-1*(V2+V23) + 
        math.comb(n_var-1,1)**-1*(V2+V24) + 
        math.comb(n_var-1,2)**-1*(V2+V12+V23+V123) + 
        math.comb(n_var-1,2)**-1*(V2+V12+V24+V124) + 
        math.comb(n_var-1,2)**-1*(V2+V23+V24+V234)+
        math.comb(n_var-1,3)**-1*(V2+V12+V23+V24+V123+V124+V234+V1234) 
        )
    
    Sh3 = (1/n_var)*( 1*math.comb(n_var-1,0)**-1*(V3) + 
        math.comb(n_var-1,1)**-1*(V3+V13) + 
        math.comb(n_var-1,1)**-1*(V3+V23) + 
        math.comb(n_var-1,1)**-1*(V3+V34) + 
        math.comb(n_var-1,2)**-1*(V3+V13+V23+V123) +  # result_f123.fun
        math.comb(n_var-1,2)**-1*(V3+V13+V34+V134) + 
        math.comb(n_var-1,2)**-1*(V3+V23+V34+V234)+
        math.comb(n_var-1,3)**-1*(V3+V13+V23+V34+V123+V134+V234+V1234) 
        )
    
    Sh4 = (1/n_var)*( 1*math.comb(n_var-1,0)**-1*(V4) + 
        math.comb(n_var-1,1)**-1*(V4+V14) + 
        math.comb(n_var-1,1)**-1*(V4+V24) + 
        math.comb(n_var-1,1)**-1*(V4+V34) + 
        math.comb(n_var-1,2)**-1*(V4+V14+V24+V124) +  # result_f123.fun
        math.comb(n_var-1,2)**-1*(V4+V14+V34+V134) + 
        math.comb(n_var-1,2)**-1*(V4+V24+V34+V234)+
        math.comb(n_var-1,3)**-1*(V4+V14+V24+V34+V124+V134+V234+V1234) 
        )

    print('Sh1 =', Sh1)
    print('Sh2 =', Sh2)
    print('Sh3 =', Sh3)
    print('Sh4 =', Sh4)
    print()

    # Compute Shapley values for each variable
    shapley_values = [Sh1, Sh2, Sh3, Sh4]

    return shapley_values

Shap_an = compute_shapley_values_for_four_variables(Vi_an[0], Vi_an[1], Vi_an[2], Vi_an[3],
                                            Vij_an[0], Vij_an[1], Vij_an[3], Vij_an[2], Vij_an[4], Vij_an[5],
                                            Vijk_an[0],Vijk_an[1],Vijk_an[2],Vijk_an[3], 0)
print('Shap_an =', Shap_an)
print(' sum Shap_an =', np.sum(Shap_an))

# relative errors
mu_err = 1 - (ev_an/mu_y)**-1
mu2_err = 1 - (mu2_an/mu2)**-1
mu3_err = 1 - (mu3_an/mu3)**-1
mu4_err = 1 - (mu4_an/mu4)**-1
Vi_err = [1 - (Vi_an[i]/Vi[i])**-1 for i in range(len(Vi))]
Si_err = [1 - (Si_an[i]/Si[i])**-1 for i in range(len(Si))]
Vij_err = [1 - (Vij_an[i]/Vij[i])**-1 for i in range(len(Vij))]
Sij_err = [1 - (Sij_an[i]/Sij[i])**-1 for i in range(len(Sij))]
Vijk_err = [1 - (Vijk_an[i]/Vijk[i])**-1 for i in range(len(Vijk))]
Sijk_err = [1 - (Sijk_an[i]/Sijk[i])**-1 for i in range(len(Sijk))]
Shap_err = [1 - (Shap_an[i]/Shap[i]) for i in range(len(Shap))]

print()
print('mu_err =', mu_err)
print('mu2_err =', mu2_err)
print('mu3_err =', mu3_err)
print('mu4_err =', mu4_err)
print('Vi_err =', Vi_err)
print('Si_err =', Si_err)
print('Vij_err =', Vij_err)
print('Sij_err =', Sij_err)
print('Vijk_err =', Vijk_err)
print('Sijk_err =', Sijk_err)
print('Shap_err =', Shap_err)



input('i')

# print(y)

## remainder term
l_list = [1,2,3]
y_ev_rem = np.zeros((tse_order, len(l_list)))
y_mu2_rem = np.zeros((tse_order, len(l_list)))
y_mu3_rem = np.zeros((tse_order, len(l_list)))
y_mu4_rem = np.zeros((tse_order, len(l_list)))
y_Vi_rem = np.zeros((tse_order, len(l_list), n_dim))
y_Vij_rem = np.zeros((tse_order, len(l_list), len(sobol[2])))
y_Vijk_rem = np.zeros((tse_order, len(l_list), len(sobol[3])))

for n in range(tse_order):
    for il, l in enumerate(l_list):
        y_ev_rem[n,il] = oti_mu.expectation_remainder(y, n, l)
        y_mu2_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 2)
        y_mu3_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 3)
        y_mu4_rem[n,il] = oti_mu.central_moment_remainder(y, n, l, 4)
        Si_rem, Vi_rem = oti_mu.sobol_indices_remainder(y, n, l)
        y_Vi_rem[n,il,:] = Vi_rem[1]
        if (n)>-1:
            y_Vij_rem[n,il,:] = Vi_rem[2]
        if (n)>-1:
            y_Vijk_rem[n,il,:] = Vi_rem[3]



print('y_ev_rem =', y_ev_rem)
print('y_mu2_rem =', y_mu2_rem)
print('y_mu3_rem =', y_mu3_rem)
print('y_mu4_rem =', y_mu4_rem)
print('y_Vi_rem =', y_Vi_rem)
print('y_Vij_rem =', y_Vij_rem)
print('y_Vijk_rem =', y_Vijk_rem)



'''
print()

n = 5+1
k = 4

Yn = y.truncate_order(n)
print('Yn =', Yn)
Ynk = y.truncate_order(n+k)
print('Ynk =', Ynk)

E_rem = oti_mu.expectation(Ynk) - oti_mu.expectation(Yn)
print('E_rem =', E_rem)


E_err_true = 1- (Ef / oti_mu.expectation(Yn))
print('E_err_true =', E_err_true)

print(E_rem/Ef)


input('i')
'''

Ef_rel_err = np.zeros((tse_order, len(l_list)))
mu2_rel_err = np.zeros((tse_order, len(l_list)))
mu3_rel_err = np.zeros((tse_order, len(l_list)))
mu4_rel_err = np.zeros((tse_order, len(l_list)))
# Vi_rel_err = np.zeros_like(y_Si_rem)
Vi_rel_err = np.zeros((tse_order, len(l_list), n_dim))
Vij_rel_err = np.zeros((tse_order, len(l_list), len(sobol[2])))
Vijk_rel_err = np.zeros((tse_order, len(l_list), len(sobol[3])))

for n in range(tse_order):    
    for ck, k in enumerate(l_list):
        Ef_rel_err[n,ck] = 1-(y_ev_rem[n,ck]/(ev_an - oti_mu.expectation(y.truncate_order(n+1))))
        mu2_rel_err[n,ck] = 1-(y_mu2_rem[n,ck]/(mu2_an - oti_mu.central_moment(y.truncate_order(n+1),2)))
        mu3_rel_err[n,ck] = 1-(y_mu3_rem[n,ck]/(mu3_an - oti_mu.central_moment(y.truncate_order(n+1),3)))
        mu4_rel_err[n,ck] = 1-(y_mu4_rem[n,ck]/(mu4_an - oti_mu.central_moment(y.truncate_order(n+1),4)))
        Si_rem, Vi_rem, f_hdmr_rem = oti_mu.sobol_indices(y.truncate_order(n+1))
        if (n)>-1:
            Vi_rel_err[n,ck,:] = 1-(np.array(y_Vi_rem[n,ck,:])/(np.array(Vi_an) - np.array(Vi_rem[1])))
        if (n)>-1:
            Vij_rel_err[n,ck,:] = 1-(np.array(y_Vij_rem[n,ck,:])/(np.array(Vij_an) - np.array(Vi_rem[2])))
        if (n)>-1:
            Vijk_rel_err[n,ck,:] = 1-(np.array(y_Vijk_rem[n,ck,:])/(np.array(Vijk_an) - np.array(Vi_rem[3])))

print()
print('Ef_rel_err =', Ef_rel_err)
print('mu2_rel_err =', mu2_rel_err)
print('mu3_rel_err =', mu3_rel_err)
print('mu4_rel_err =', mu4_rel_err)
print('Vi_rel_err =', Vi_rel_err)
print('Vij_rel_err =', Vij_rel_err)
print('Vijk_rel_err =', Vijk_rel_err)



input('i')

