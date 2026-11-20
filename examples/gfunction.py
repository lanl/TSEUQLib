# Python utility
import sys
print(sys.executable)
import time
import numpy as np
# Pandas is use to print a table data as an output.
import pandas as pd
# Itertools is use for the variable combinations.
from itertools import combinations
# Path to  otilib_main library
# Make sure it points to the build path
# Importing pyoti library
import pyoti.sparse as oti

# Set pyoti to print all coefficients.
# '-1' prints complete TSE
oti.set_printoptions(terms_print=-1)

# path to tseuqlib
sys.path.append('../')
# Importing "tseuqlib" libraries
# Importing utilities
import tseuqlib.oti_util as uoti
# Importing moments
import tseuqlib.oti_moments as moti
# Importing random variable moments
import tseuqlib.rv_moments as rv

# Write the function.
def funct(xx, a=None):

    d = len(xx)
    
    if a is None:
        a = np.array([(i - 1) / 2 for i in range(1, d + 1)])
    
    prod = 1
    for i in range(d):
        xi = xx[i]
        ai = a[i]
        new1 = oti.abs(4 * xi - 2) + ai
        new2 = 1 + ai
        prod *= new1 / new2
    
    return prod

# Provide number of variables in the function.
# (pass a smaller number on the command line to try it quickly)
n_var = int(sys.argv[1]) if len(sys.argv) > 1 else 100

t_var = np.zeros(n_var)
t_sob = np.zeros(n_var)
for d in range(1, n_var+1):
    n_var = d

    # Random variable parameters
    # Identify the type of distribution for each variable.
    rv_pdf_name = np.array(['N'] * n_var)  # PDF (Probability Density Function) of each variable.
    # a = shape parameter of each random variable
    # '+oti.e(int(n_var+1),order=1)' is perturbing a parameter of interest along a new imaginary direction.
    rv_a = np.array([1] * n_var)
    # b = scale parameter of each random variable
    rv_b = np.array([1] * n_var)
    # Compute mean and standard deviation of random variables.
    # Store as rv_params = [rv_pdf_name, rv_a, rv_b, rv_mean, rv_standarddeviation]
    rv_params = rv.rv_mean_from_ab([rv_pdf_name, rv_a, rv_b])
    rv_mean = rv_params[3]
    rv_standarddeviation = rv_params[4]

    # Inside the " " are the titles of the columns in the table and this is the same throughout the script.
    df = pd.DataFrame({
        "Distribution": rv_pdf_name,
        "Shape": rv_a,
        "Scale": rv_b,
        "Mean": rv_mean,
        "Standard Deviation": rv_standarddeviation,
    })

    # Optional: adds a line for separation ('50' is the number of '=')
    print("=" * 80)  
    # Print the title and the DataFrame as a table
    print("Random Variable Parameters")
    print("-" * 80)
    # Print the DataFrame as a table
    # The ".to_spring" converts the DataFrame into a string representation.
    # The "index=False" means that the index will not be displayed in the string.
    print(df.to_string(index=False, justify='center'))
    print("=" * 80)










    # User provided order of Taylor Series Expansion
    tse_order_max = 2

    # Generate oti random variable(s) at the mean (rv_params[3]) by perturbing with OTI imaginary directions
    x = [0]*n_var
    # The first variable in rvmean correspond to the first element of the rvmean
    for d in range(n_var):
        # 'oti.e(int(d+1)' creates oti direction of 'order=tse_order_max'
        x[d] = rv_mean[d] + oti.e(int(d+1), order=tse_order_max)

    # Generate central moments of the random variables
    # User can change #4 to any.
    # Order must be >= order of TSE times the maximum order of central moments being calculated.
    # The variance and the Sobol' indices need moments up to order 2n. (mu3
    # and mu4 need 3n and 4n, where pyoti allows at most 10 inputs.)
    rv_moments_order = int(tse_order_max*2)

    # Compute central moments of the random variables
    # Rv moments object
    rv_moments = rv.rv_central_moments(rv_params, rv_moments_order)
    # Compute central moments
    rv_moments.compute_central_moments()
    rv_mu = rv_moments.rv_mu

    # Precompute the joint distribution from independent variables. Needed for moment calculations of the Taylor Series.
    rv_mu_joint = uoti.build_rv_joint_moments(rv_mu)

    # Defining central moments.
    moment_names = [f'mu_{i+1}' for i in range(len(rv_mu))]

    df = pd.DataFrame({
        "Central Moment": moment_names,
        "Values": rv_mu,
    })

    print("=" * 80)
    print("Central Moments")
    print("-" * 80)
    # The format '%.6f' means that numbers will be displayed with six decimal places.
    # The "justify" is to make sure table contents are centered in the table.
    print(df.to_string(index=False, float_format='%.6f', justify='center'))
    print("=" * 80)









    # Creating the Taylor Series Expansion
    y = funct(x)

    print("=" * 80)
    print(' Function evaluated at OTI-perturbed inputs')
    print("-" * 80)
    print('TSE =', y)
    print("=" * 80)





    # Define indices of random variables starting at 1.
    active_bases = [i for i in range(1, n_var+1)]

    # Define variables use for derivatives with respect to random variable parameters (mean/sd)
    # This tells the program which imaginary bases is used for computing the derivatives.
    # For example if you have 5 variables, the next variable is the imaginary base (6).
    active_bases_derivs = []

    # create oti_moments object
    # 'active_bases' and 'extra_bases' are arguments inside 'tse_uq'
    oti_mu = moti.tse_uq(n_var, rv_mu_joint) #, active_bases=active_bases,
#                         extra_bases=active_bases_derivs)

    # Compute expectation of Taylor Series
    mu_y = oti_mu.expectation(y)

    # Compute derivatives of expected value
    # Extract sensitivities of central moments with respect to rv parameters
    try:
        dev_db1 = mu_y.get_deriv(active_bases_derivs[0])
        print('Derivative of Expected Value =', dev_db1)
    except:
        pass

    # Compute total variance of Taylor Series


    t1 = time.time()
    mu2 = oti_mu.central_moment(y, 2)
    t_var[d - 1]  = time.time() - t1
    # mu3 (unused here) needs moments up to order 3n
    # mu3 = oti_mu.central_moment(y, 3)

    try:
        #Compute derivatives of total variance.
        dmu2_db1 = mu2.get_deriv(active_bases_derivs[0])
        print('Derivative of Total Variance =', dmu2_db1)
    except:
        pass

    print("*" * 35)
    print(f" Expected Value:   {mu_y.real:10.3f}")
    print(f" Total Variance:   {mu2.real:10.3f}")
    print("*" * 35)











    # Setting display options to show all data
    # To show all columns
    pd.set_option('display.max_columns', None)
    # To show all rows
    pd.set_option('display.max_rows', None)
    # To ensure that full content of each column is visible
    pd.set_option('display.max_colwidth', None)
    # Display floats in exponential form with 5 significant digits
    pd.set_option('display.float_format', '{:.5e}'.format)

    # Compute Sobol indices and partial variance

    t1 = time.time()
    sobol, Vy_hdmr, f_hdmr = oti_mu.sobol_indices(y)
    t_sob[d - 1]  = time.time() - t1

    # Initializing a dictionary to store results
    data = {
        'Order': [],
        'Variable Combination': [],
        'Si': [],
        'Vi': [],
        'Derivatives of Si': [],
        'Derivatives of Vi': []
    }

    # Iterate through the Sobol indices starting from order 1
    for i in range(1, len(sobol)):  # Select 1 to avoid "order 0". "Order 0" is the total variance.
        # Extract real parts of Sobol and variance indices (no derivatives being extracted)
        # Sobol/Vy_hdmr[order][index]
        real_si = [sobol[i][j].real for j in range(len(sobol[i]))]
        real_vi = [Vy_hdmr[i][j].real for j in range(len(Vy_hdmr[i]))]
        
        # Initializing derivatives
        Derivatives_si = []
        Derivatives_vi = []

        # Extract partial derivatives of the expected value and variance with respect to the mean of the first variable.    
        try:
            Derivatives_si = [sobol[i][j].get_deriv(active_bases_derivs[0]) for j in range(len(sobol[i]))]
            Derivatives_vi = [Vy_hdmr[i][j].get_deriv(active_bases_derivs[0]) for j in range(len(Vy_hdmr[i]))]
        except:
            pass

        # Append data to the lists. Append is a method used to add an item to the end of a list.
        data['Order'].append(i)
        data['Variable Combination'].append(f"Order {i}")
        data['Si'].append(real_si)
        data['Vi'].append(real_vi)
        data['Derivatives of Si'].append(Derivatives_si)
        data['Derivatives of Vi'].append(Derivatives_vi)

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Prepare a new DataFrame for vertical results
    real_data = {
        'Order': [],
        'Variable Combination': [],
        'Si': [],
        'Vi': [],
    }

    # Prepare to store derivatives data separately
    deriv_data = {
        'Order': [],
        'Variable Combination': [],
        'Derivatives of Si': [],
        'Derivatives of Vi': [],
    }

    # Populate the new DataFrame with vertical results
    for i in range(len(df)):
        order = df['Order'][i]
        real_si_values = df['Si'][i]
        real_vi_values = df['Vi'][i]
        # Generate combinations for the current order
        order_combinations = list(combinations(range(1, n_var + 1), order))

        # Combine Si and Vi values with their corresponding combinations
        for idx in range(len(real_si_values)):
            real_data['Order'].append(order)
            real_data['Variable Combination'].append(order_combinations[idx] if idx < len(order_combinations) else None)
            real_data['Si'].append(real_si_values[idx] if idx < len(real_si_values) else None)
            real_data['Vi'].append(real_vi_values[idx] if idx < len(real_vi_values) else None)

    # Create the vertical results DataFrame
    vertical_results_df = pd.DataFrame(real_data)


np.save('t_var.npy',t_var)
np.save('t_sob.npy',t_sob)

t_var =  np.load('t_var.npy')
t_sob =  np.load('t_sob.npy')

print(t_var)
print(t_sob)

# Handle derivatives in a separate DataFrame
for i in range(len(df)):
    order = df['Order'][i]
    order_combinations = list(combinations(range(1, n_var + 1), order))
    deriv_si_values = df['Derivatives of Si'][i]
    deriv_vi_values = df['Derivatives of Vi'][i]
    
    for idx in range(len(deriv_si_values)):
        deriv_data['Order'].append(order)
        deriv_data['Variable Combination'].append(order_combinations[idx] if idx < len(order_combinations) else None)
        deriv_data['Derivatives of Si'].append(deriv_si_values[idx] if idx < len(deriv_si_values) else None)
        deriv_data['Derivatives of Vi'].append(deriv_vi_values[idx] if idx < len(deriv_vi_values) else None)

# Creating the derivatives DataFrame
deriv_results_df = pd.DataFrame(deriv_data)

print("=" * 80)
print("Sobol Indices:")
print("-" * 80)
print(vertical_results_df.to_string(index=False, justify='center'))
print("-" * 80)
print("Partial Derivatives:")
print("-" * 80)
print(deriv_results_df.to_string(index=False, justify='center'))
print("=" * 80)
