# Central moments of the Taylor series expansion, represented by OTI numbers

import pyoti.sparse as oti
import pyoti.core as coti

import numpy as np

from math import comb

from itertools import chain, combinations


# # Uncomment for profiling.
# #  - Install using pip install line-profiler
# #  - Decorate the function you want with @profile.
# #  - To profile, run your application with 
# #        LINE_PROFILE=1 python filename.py
# #
from line_profiler import profile


def get_extra_bases(abases, n_dim):
    extra_bases = []
    for a in abases:
        if a not in range(1, n_dim+1):
            extra_bases.append(a)
    return extra_bases


class tse_uq():

    def __init__(self, n_dim, mu, extra_bases=[]):
        
        '''
        Args:
            - n_dim (int): number of random variables
            - mu (list): joint central moments of the random variables
        '''
        
        self.n_dim = n_dim
        self.mu = mu

        self.extra_bases = extra_bases
        
        return
    
    
    # @profile
    def expectation(self, y):
        """
        Compute the expectation of the TSE built by the OTI number y.
            
             E[ y ]
            
        Args:
            - y (oti number):  OTI with the TSE.
        
        """
        if type(y)==float:
            return y
        
        # Direction helper from OTI.
        h     = coti.get_dHelp()

        # get active bases in y
        abases = y.get_active_bases()
        
        # define number of bases
        if len(abases)!=0:
            nbasis = self.n_dim # abases[-1]
        else:
            return y.real
        
        # get additional bases in y used for perturbation
        extra_bases = self.extra_bases # get_extra_bases(abases, self.n_dim)
        
        # Gets the order of the number.
        order = y.order 
        
        
        # use conditional expectation if extra_bases exists
        if len(extra_bases)>0:
            
            # get contribution of sensitiviity from the mean value
            M = self.conditional_expectation(y, extra_bases)
            #Mr = ym.real
            #Mi = ym.get_deriv([5])
            '''            
            # get contribution of sensitivity from the moments
            for ordi in range(1,order+1):
            
                nimord = coti.ndir_order(nbasis, ordi)
                
                for idx in range(nimord):
                    
                    # Get the imaginary coeff. for the direction given by [idx, ordi]
                    coeff = y[[idx,ordi]]
                    
                    mu_oi = self.mu[ordi-1][idx]
                    
                    tmp = coeff*mu_oi
                    try:
                        mui = tmp.get_deriv(extra_bases)
                    except:
                        mui = 0
                    Mi+=mui
            '''
            return M # Mr + Mi*oti.e(extra_bases[0], order = order)
        
        
        # initialize expectation of Taylor series with the zero'th-order term
        M = y.real

        for ordi in range(1,order+1):
         
            nimord = coti.ndir_order(nbasis, ordi)
            
            for idx in range(nimord):
                
                # Get the imaginary coeff. for the direction given by [idx, ordi]
                coeff = y[[idx,ordi]]
                
                # get the joint central moment of the random variables
                mu_oi = self.mu[ordi-1][idx]
                
                # substitute joint central moment with delta_xi in Eq. 11
                M += coeff*mu_oi
        
        return M


    
    # @profile
    def central_moment(self, y, k, Ey=None):
        
        """
        @brief Computes the central moment of the TSE expansion given by the OTI number y.
             E[ ( y - E[y] )**k ]
        
        INPUTS:
        @param[in] y: OTI with the TSE of the function of the value obtained.
        @param[in] k: Order of the moment operator. Must be greater or equal than 2.
        @param[in] mu: structure with the moments of the variables in y.
        @param[in] Ey: (optional) Known value of the expectation of y (E[y]).
        
        EXAMPLE USAGE:
            
            def func(x,alg=oti):
                return alg.sin(x**2)
            
            x = 0.5+oti.e(1,order = 5)
            
            y = func(x)
            mu_x = 0.5
            si_x = 0.5
            
            # Generate the Moments vector:
            mu = [ [stats.norm.moment(i, loc=0, scale=si_x)] for i in range(1,50)] # computing the 49 first moments.
            
            mu1_tse = expectation(y,mu)
            mu2_tse = central_moment(y,2,mu)
            mu3_tse = central_moment(y,3,mu)
            mu4_tse = central_moment(y,4,mu)
            mu5_tse = central_moment(y,5,mu)
            
            print('Mean:',mu1_tse)
            print('Var: ',mu2_tse)
            print('mu3: ',mu3_tse)
            print('mu4: ',mu4_tse)
            print('mu5: ',mu5_tse)
        
        """
        if type(y)==float:
            return 0

        # get active basis of y
        abases = y.get_active_bases()

        # get any additional basis not in original problem (bases used for sensitivities)
        extra_bases = get_extra_bases(abases, self.n_dim)

        # Compute expected value of y.
        if Ey is None:
            Ey = self.expectation(y)
        
        # Change y truncation order of y to k*order
        newOrd = y.order*k
        e1_null = oti.e(1,order=newOrd)
        
        # Dummy way to increase truncation order.
        yprime = y + 0 * e1_null
        
        # integrand for k'th central moment
        yprime = (yprime-Ey)**k
        #print('yprime =', yprime)

        # Evaluate expectation operator
        M = self.expectation(yprime)
        
        return M
    
    def central_moment_hdmr(self, y, k, Ey=None):
        
        """
        @brief Computes the central moment of the TSE expansion given by the OTI number y.
             E[ ( y - E[y] )**k ]
        
        INPUTS:
        @param[in] y: OTI with the TSE of the function of the value obtained.
        @param[in] k: Order of the moment operator. Must be greater or equal than 2.
        @param[in] mu: structure with the moments of the variables in y.
        @param[in] Ey: (optional) Known value of the expectation of y (E[y]).
        
        """
        if type(y)==float:
            return 0

        # get active basis of y
        abases = y.get_active_bases()

        # get any additional basis not in original problem (bases used for sensitivities)
        extra_bases = get_extra_bases(abases, self.n_dim)

        # Compute expected value of y.
        if Ey is None:
            Ey = self.expectation(y)
        
        # Change y truncation order of y to k*order
        newOrd = y.order*k
        e1_null = oti.e(1,order=newOrd)
        
        # Dummy way to increase truncation order.
        yprime = y + 0 * e1_null
        
        # integrand for k'th central moment
        yprime = (yprime-Ey)**k
        #print('yprime =', yprime)

        # Evaluate expectation operator
        M = self.conditional_expectation_hdmr(yprime, extra_bases)
        
        return M
    
    # =================================================================================
    # @profile
    def conditional_expectation(self, y, basis, inc_mom=True):
        """
        @brief Computes the conditional expectation of the TSE 
               from the OTI number y.
            
             E[ y | x_{basis[0]}, x_{basis[1]}, ... ]
            
        @param[in] y:  OTI with the TSE.
        @param[in] mu: structure with the moments of the variables in y.
        @param[in] basis: (int or list) Basis to excluded from expectation.
        
        """

        if type(y)==float:
            return y
        
        if type(basis)== int:
            blist = [basis]
        else:
            blist = basis
        
        if len(basis)==0:
            return self.expectation(y)
        
        # Direction helper from OTI.
        h     = coti.get_dHelp()
        
        order = y.order # Gets the order of the number.
        
        # get active basis in the OTI number
        abases = y.get_active_bases()
        
        if len(abases)!=0:
            nbasis = abases[-1]
        else:
            return y.real
        
        # extra_bases = [] # self.extra_bases
        
        # Assume that the result is an OTI with certain 
        # number of basis and truncation order.
        M = oti.sotinum(y.real, len(basis), order)
        
        for ordi in range(1,order+1):
            
            nimord = coti.ndir_order(nbasis,ordi)
            
            for idx in range(nimord):
                
                # Get the imaginary coeff.
                coeff = y[[idx,ordi]]
                
                i_mu = idx
                o_mu = ordi
                
                # Start with real direction.
                i_rem = 0
                o_rem = 0
                
                for i in blist:
                    
                    # Remove imdir i from the main dir.
                    im = i-1
                    
                    while True:
                        
                        res = coti.div_imdir_idxord(i_mu,o_mu,im,1)
                        
                        if type(res)==int:
                            #Result is not further divisible by the basis.
                            break
                        else:
                            # Update the new values of idx,order
                            i_mu, o_mu = res
                            
                            if o_rem == 0:
                                
                                i_rem = im
                                o_rem = 1
                                
                            else:
                                
                                i_rem, o_rem = h.mult_dir(im,1,i_rem,o_rem)
                
                
                # Get remaining direction:
                if o_mu ==0:
                    
                    # The direction is fully divisible by the basis.
                    i_rem = idx
                    o_rem = ordi
                    mu_oi = 1.0 # zeroth order central moment
                
                else:
                    
                    # Find the remaining of the derivative.
                    mu_oi = self.mu[o_mu-1][i_mu]
                
                # print(coeff,mu_oi)
                #print(h.get_fulldir(idx,ordi))
                #print(coeff)
                #print(mu_oi.real)
                M[[i_rem,o_rem]] += coeff*mu_oi.real
        
        #print('M =', M)
        #input('i')

        extra_bases = self.extra_bases
        nbasis = self.n_dim
        Mi1 = 0 
        Mi2 = 0
        Mi3 = 0

        
        if inc_mom and len(extra_bases)>0:
            # basis of the parameter perturbation (was hard-coded as 5)
            eb = extra_bases[0]
            # get contribution of sensitivity from the moments
            for ordi in range(1,order+1):
            
                nimord = coti.ndir_order(nbasis, ordi)
                
                for idx in range(nimord):
                    
                    # Get the imaginary coeff. for the direction given by [idx, ordi]
                    coeff = y[[idx,ordi]]
                    
                    mu_oi = self.mu[ordi-1][idx]
                    
                    tmp = coeff*mu_oi
                    try:
                        Mi1 += tmp.get_deriv(extra_bases)
                    except:
                        pass
                    try:
                        Mi2 += tmp.get_deriv([[eb,2]])
                    except:
                        pass
                    try:
                        Mi3 += tmp.get_deriv([[eb,3]])
                    except:
                        pass
    
            Mi1_idx = coti.imdir([[eb,1]])
            M[[Mi1_idx[0], 1]] += Mi1
            Mi2_idx = coti.imdir([[eb,2]])
            M[[Mi2_idx[0], 2]] += Mi2
            Mi3_idx = coti.imdir([[eb,3]])
            M[[Mi3_idx[0], 3]] += Mi3
            
            #print('M5i =', M[[Mi1_idx[0], 1]])
    
        return M
    
    def conditional_expectation_hdmr(self, y, basis, inc_mom=True):
        """
        @brief Computes the conditional expectation of the TSE 
               from the OTI number y.
            
             E[ y | x_{basis[0]}, x_{basis[1]}, ... ]
            
        @param[in] y:  OTI with the TSE.
        @param[in] mu: structure with the moments of the variables in y.
        @param[in] basis: (int or list) Basis to excluded from expectation.
        
        """

        if type(y)==float:
            return y
        
        if type(basis)== int:
            blist = [basis]
        else:
            blist = basis
        
        if len(basis)==0:
            return self.expectation(y)
        
        # Direction helper from OTI.
        h     = coti.get_dHelp()
        
        order = y.order # Gets the order of the number.
        
        # get active basis in the OTI number
        abases = y.get_active_bases()
        
        if len(abases)!=0:
            nbasis = self.n_dim # abases[-1]
        else:
            return y.real
        
        # extra_bases = [] # self.extra_bases
        
        # Assume that the result is an OTI with certain 
        # number of basis and truncation order.
        M = oti.sotinum(y.real, len(basis), order)
        
        for ordi in range(1,order+1):
            
            nimord = coti.ndir_order(nbasis,ordi)
            
            for idx in range(nimord):
                
                # Get the imaginary coeff.
                coeff = y[[idx,ordi]]
                
                i_mu = idx
                o_mu = ordi
                
                # Start with real direction.
                i_rem = 0
                o_rem = 0
                
                for i in blist:
                    
                    # Remove imdir i from the main dir.
                    im = i-1
                    
                    while True:
                        
                        res = coti.div_imdir_idxord(i_mu,o_mu,im,1)
                        
                        if type(res)==int:
                            #Result is not further divisible by the basis.
                            break
                        else:
                            # Update the new values of idx,order
                            i_mu, o_mu = res
                            
                            if o_rem == 0:
                                
                                i_rem = im
                                o_rem = 1
                                
                            else:
                                
                                i_rem, o_rem = h.mult_dir(im,1,i_rem,o_rem)
                
                
                # Get remaining direction:
                if o_mu ==0:
                    
                    # The direction is fully divisible by the basis.
                    i_rem = idx
                    o_rem = ordi
                    mu_oi = 1.0 # zeroth order central moment
                
                else:
                    
                    # Find the remaining of the derivative.
                    mu_oi = self.mu[o_mu-1][i_mu]
                
                # print(coeff,mu_oi)
                #print(h.get_fulldir(idx,ordi))
                #print(coeff)
                #print(mu_oi.real)
                M[[i_rem,o_rem]] += coeff*mu_oi.real
        
        #print('M =', M)
        #input('i')
        
        extra_bases = self.extra_bases
        nbasis = self.n_dim
        Mi1 = 0 
        Mi12 = 0 
        Mi2 = 0
        Mi3 = 0

        
        if inc_mom and len(extra_bases)>0:
            # basis of the parameter perturbation (was hard-coded as 5)
            eb = extra_bases[0]
            # get contribution of sensitivity from the moments
            for ordi in range(1,order+1):
            
                nimord = coti.ndir_order(nbasis, ordi)
                
                for idx in range(nimord):
                    
                    # Get the imaginary coeff. for the direction given by [idx, ordi]
                    coeff = y[[idx,ordi]]
                    
                    mu_oi = self.mu[ordi-1][idx]
                    
                    tmp = coeff*mu_oi
                    try:
                        Mi1 += tmp.get_deriv(extra_bases)
                    except:
                        pass
                    try:
                        Mi12 += tmp.get_deriv([1,5])
                    except:
                        pass
                    try:
                        Mi2 += tmp.get_deriv([[eb,2]])
                    except:
                        pass
                    try:
                        Mi3 += tmp.get_deriv([[eb,3]])
                    except:
                        pass
    
            Mi1_idx = coti.imdir([[eb,1]])
            M[[Mi1_idx[0], 1]] += Mi1
            Mi12_idx = coti.imdir([1,5])
            M[Mi12_idx] += Mi12
            Mi2_idx = coti.imdir([[eb,2]])
            M[[Mi2_idx[0], 2]] += Mi2
            Mi3_idx = coti.imdir([[eb,3]])
            M[[Mi3_idx[0], 3]] += Mi3
            
            #print('M5i =', M[[Mi1_idx[0], 1]])
    
        return M


    # ---------------------------------------------------------------------------------

    # =================================================================================
    

    # ---------------------------------------------------------------------------------

    # @profile
    def sobol_indices(self, y, max_order = None):
        
        """
        @brief Computes the sobol up to n interactions using the 
               conditional expectation of the TSE built by the 
               OTI number y.
        
        E[ y | x_{basis[0]}, x_{basis[1]}, ... ]
             
        INPUTS:
        @param y: OTI with the TSE.
        @param mu: structure with the moments of the variables in y.
        @param max_order (Optional) Integer representing the maximum order of 
                         interaction terms for the calculations.
        
        """
        # HDMRs:
        # Use conditional expectation to take variables out of the consideration
        #
        
        # if type(y)==float:
        #    return

        abases = y.get_active_bases()
        
        extra_bases = self.extra_bases # get_extra_bases(abases, self.n_dim)
        
        # get f0
        f0 = self.expectation(y)
        
        # build HDMR
        HDMR, term_struct, srch_struct = self.build_hdmr(y, max_order=max_order, Ey=f0)
        
        # Compute partial variances:
        
        # Start with the Partial Variances.
        Vtotal = self.central_moment(y, 2)
        
        # store partial variances
        V = [[Vtotal]]
        
        # compute all variances of the HDMR
        for ordi in range(1,len(HDMR)):
            HDMR_i = HDMR[ordi]
            nterms = len(HDMR_i)
            V_i = [0]*nterms
            for j in range(nterms):
                #print('HDMR_i[j] =', HDMR_i[j])
                V_i[j] = self.central_moment(HDMR_i[j],2,0)
                #print(V_i[j])
                #input('ii')
            V.append(V_i)
        
        # Vtotal = V[0][0]
        
        sobol = [[1]]
        
        # Now we can compute the Sobol from Partial Variances of the HDMRs.
        for ordi in range(1,len(V)):
            V_i = V[ordi]
            nterms = len(V_i)
            sobol_i = [0]*nterms
            for j in range(nterms):
                sobol_i[j] = V[ordi][j]/Vtotal
            sobol.append(sobol_i)
        return sobol, V


    # HDMR representation
    # @profile
    def build_hdmr(self, y, max_order=None, Ey=None):
        """
        @brief Computes the High Dimensional Model Representation of
               the TSE built using OTI number y.
                
               This uses the conditional expectation function.
                
                E[ y | x_{basis[0]}, x_{basis[1]}, ... ]
             
        INPUTS:
        @param y: OTI with the TSE.
        @param mu: structure with the moments of the variables in y.
        @param max_order (Optional) Integer representing the maximum order of 
                         interaction terms for the calculations.

        """
        # HDMRs:
        # Use conditional expectation to take variables out of the consideration
        #
        
        if Ey is None:
            f0   = self.expectation(y)
        else:
            f0 = Ey
        
        
        tse_order = y.order
        
        # get active basis in the OTI number
        abases = y.get_active_bases()
        
        extra_bases = self.extra_bases # [] # get_extra_bases(abases, self.n_dim)
        
        if len(abases)!=0:
            dim = self.n_dim # abases[-1]
        else:
            raise ValueError('OTI number not perturbed.')
        # end if

        var_arr = list( range(1, dim+1) )[::-1]
        
        srch_struct = [[0]]
        
        term_struct = [[0]]
        HDMR_functions = [[f0]]
        
        if max_order is None:
            max_order = min(tse_order,dim)
        else:
            max_order = min(max_order,min(tse_order,dim))
        # end
        
        # There should be a function that computes the HDMR representation.
        for i in range(1, max_order+1):
            
            terms = list(combinations(var_arr,i))[::-1] 
            
            nterms = len(terms)
            
            srch_struct_level    = [0]*nterms
            HDMR_functions_level = [0]*nterms
            
            for i_term in range(len(terms)):
                
                term = terms[i_term][::-1]
                
                terms[i_term] = term
                
                idx = coti.imdir(term)[0]
                
                srch_struct_level[i_term] = idx
                
                # add extra_bases to conditional expectation
                term_exclude = list(term).copy()
                # print('extra_bases =', extra_bases)
                if len(extra_bases)>0:
                    for b in extra_bases:
                        term_exclude.append(b)
                #print('term_exclude =', term_exclude)
                f_hdmr = self.conditional_expectation(y, term_exclude, inc_mom=True)-f0
                #print('f_hdmr =', f_hdmr)
                #rem_vars = [element for element in var_arr if element not in term]
                #if (len(term)==1) and (1 not in term):
                ##    for j in rem_vars:
                #    f_hdmr -= self.conditional_expectation(y, term_exclude, inc_mom=True)
                #    #f_hdmr -= self.conditional_expectation_hdmr(y, [1], inc_mom=True)
                #    f_hdmr += self.conditional_expectation_hdmr(y, [1,5], inc_mom=True)
                #    print('f_hdmr =', f_hdmr)
                #    print()
                #    input('i')

                #print('f_hdmr =', f_hdmr)
                #input('i')
                #f_hdmr = self.conditional_expectation(y, term_exclude, inc_mom=False)-f0
                #print()
                #print(f_hdmr)
                #input('iiii')
                
                # Get the terms needed for subtraction
                rhs_comp = self.comp_rhs_terms(term, srch_struct)
                
                for ordj in range(1,i):
                    
                    for idx in rhs_comp[ordj-1]:
                        
                        f_hdmr -= HDMR_functions[ordj][idx]
                
                
                HDMR_functions_level[i_term] = f_hdmr
            
            srch_struct.append(srch_struct_level)
            term_struct.append(terms)
            HDMR_functions.append(HDMR_functions_level)
        
        return HDMR_functions, term_struct, srch_struct


    # @profile
    def shapley_values_v2(self, y, max_order = None):
        
        """
        Computes the Shapley interaction effects using varience decomposition
             
        Args:
        @param y: OTI with the TSE.
        @param mu: structure with the moments of the variables in y.
        @param max_order (Optional) Integer representing the maximum order of 
                          interaction terms for the calculations.
        
        """
        # HDMRs:
        # Use selective expectation to take variables out of the consideration
        #
        f0   = self.expectation(y)
        
        HDMR,term_struct, srch_struct  = self.build_hdmr(y, max_order=max_order, Ey = f0)
        
        # Compute partial variances:
        
        # Start with the Partial Variances.
        Vtotal = self.central_moment(y,2,Ey = f0)
        
        V = [[Vtotal]]
        
         # Now we can compute all the Variances of the HDMRs.
        for ordi in range(1,len(HDMR)):
            HDMR_i = HDMR[ordi]
            nterms = len(HDMR_i)
            V_i = [0]*nterms
            for j in range(nterms):
                
                V_i[j] = self.central_moment(HDMR_i[j],2,0)
                
            V.append(V_i)
            
        tse_order = y.order
        
        # get active basis in the OTI number
        abases = y.get_active_bases()
         
        if len(abases)!=0:
            dim = self.n_dim # abases[-1]
        else:
            raise ValueError('OTI number not perturbed.')
        # end if
         
        var_arr = list( range(1, dim+1) )[::-1]
        var_arr_tup = tuple(range(1, dim+1))
        
        
        
        if max_order is None:
            max_order = min(tse_order,dim)
        else:
            max_order = min(max_order,min(tse_order,dim))
        
        
        var_array_set = set(var_arr)
        shap = np.zeros((1,dim))
        dict_idx_terms = {}
        for i in range(1,max_order+1):
            summation_terms = list(combinations(var_arr,i))
            for j in summation_terms:
                idx = coti.imdir(j)[0]
                dict_idx_terms['{0},{1}'.format(j[::-1],idx)] =  np.searchsorted(srch_struct[i], idx)
        
        #
        dict_idx_terms = [dict()]
        dict_idx_terms[0][0]=0 # zeroth order case.
        
        list_imidx_terms = []
        list_imidx_terms.append([0]) # zeroth order case.
        
        for i in range(1,max_order+1):
            local_dict = {}
            srch_local = srch_struct[i]
            for local_idx in range(len(srch_local)):
                global_idx = srch_local[local_idx]
                local_dict[global_idx]= local_idx
            dict_idx_terms.append(local_dict)
        
        
        
        # Create dictionary to reference tau later
        combinations_tau = [dim-1, dim]
        dict_tau = [dict()]
        dict_tau[0][()]=0 
        
        for l in combinations_tau:
            summation_terms = list(combinations(var_arr,l))
            local_dict = {}
            for j in summation_terms:
                tau_j = 0
                for k in range(1,min(l+1, tse_order+1)):
                        summation_terms_V_j =  set(combinations(j,k))
                        for idx1 in summation_terms_V_j:
                            idx_j = coti.imdir(idx1)[0]
                            idx_j  = dict_idx_terms[k][idx_j]
                            var = V[k][idx_j]/k
                            tau_j = tau_j + var
                local_dict[tuple(sorted(j))] = tau_j
            dict_tau.append(local_dict)
        
        
        #Compute shapely value
        # object dtype, so entries can carry the OTI parameter sensitivity
        shap = np.zeros((1,dim), dtype=object)
        for i in range(1,dim+1):
            arr_set_shap_i = var_array_set - set([i])
            summation_terms = list(combinations(arr_set_shap_i,dim-1))
            for j in summation_terms:
                idx_j = tuple(j)
                tau_j_plus_i= dict_tau[2][var_arr_tup]
                tau_j = dict_tau[1][tuple(sorted(idx_j))]
                shap[0,i-1] =  (tau_j_plus_i - tau_j)
        
        return shap
    
    
    
    
    
    # @profile
    def comp_rhs_terms(self, term, srch_struct):
        '''
        Compute the summation of terms to subtract from the conditional expectation when computing the HDMR
        
        Args:
            - term:
            - srch_struct:
        
        '''
        
        order = len(term)
        
        index = []
        for i in range(2, order):
            idx_level = []
            combos = list(combinations(term, i))
            for j in combos:
                idx_term_level = coti.imdir(j)[0]
                idx_level.append(np.searchsorted(srch_struct[i], idx_term_level))
            index.append(idx_level)
            
        return_index = [[i -1 for i in term]]
        for i in range(0,len(index)):
            return_index.append(index[i])
        
        return return_index
    
    
    
    # remainder term functions
    
    def tse_remainder(self, y, n, l):
        '''
        Compute approximation of the remainder term of the n'th-order Taylor series expansion.
        R_n,l = Y_{n+l} - Y_{n}
        
        '''
        
        # if y.order<(n+l):
        #     raise ValueError('order of oti number is not large enough to compute remainder term in oti_uq.tse_remainder()')
        
        try: 
            int(l)
        except: 
            raise ValueError('l must be an integer in oti_uq.tse_remainder()')
        if l<1:
            raise ValueError('l must be an integer greater than 1 in oti_uq.tse_remainder()')
        
        if n==0:
            return y.truncate_order(n+l) - y.real
        else:
            return y.truncate_order(n+l) - y.truncate_order(n)
            
        
    
    def expectation_remainder(self, y, n, l):
        '''
        Compute approximation of the remainder term of the expectation of the n'th-order Taylor series expansion.
        R_n,l = Y_{n+l} - Y_{n}
        
        '''
        
        # if y.order<(n+l):
        #     raise ValueError('order of oti number is not large enough to compute remainder term in oti_uq.tse_remainder()')
        try: 
            int(l)
        except: 
            raise ValueError('l must be an integer in oti_uq.expectation_remainder()')
        if l<1:
            raise ValueError('l must be an integer greater than 1 in oti_uq.expectation_remainder()')
        
        if n==0:
            return self.expectation(y.truncate_order(n+l+1)) - self.expectation(y.real)
        else:
            return self.expectation(y.truncate_order(n+l+1)) - self.expectation(y.truncate_order(n+1))
   

    def central_moment_remainder(self, y, n, l, k):
        '''
        Compute approximation of the remainder term of the k'th central moment of the n'th-order Taylor series expansion.
        R_n,l = Y_{n+l} - Y_{n}
        
        '''
        
        # if y.order<(n+l):
        #     raise ValueError('order of oti number is not large enough to compute remainder term in oti_uq.tse_remainder()')
        try: 
            int(l)
        except: 
            raise ValueError('l must be an integer in oti_uq.expectation_remainder()')
        if l<1:
            raise ValueError('l must be an integer greater than 1 in oti_uq.expectation_remainder()')
        
        if n==0:
            return self.central_moment(y.truncate_order(n+l+1), k) - self.central_moment(y.real, k)
        else:
            return self.central_moment(y.truncate_order(n+l+1), k) - self.central_moment(y.truncate_order(n+1), k)
    

    def sobol_indices_remainder(self, y, n, l):
        '''
        Compute approximation of the remainder term of the k'th central moment of the n'th-order Taylor series expansion.
        R_n,l = Y_{n+l} - Y_{n}
        
        '''
        
        # if y.order<(n+l):
        #     raise ValueError('order of oti number is not large enough to compute remainder term in oti_uq.tse_remainder()')
        try: 
            int(l)
        except: 
            raise ValueError('l must be an integer in oti_uq.expectation_remainder()')
        if l<1:
            raise ValueError('l must be an integer greater than 1 in oti_uq.expectation_remainder()')
        
        # if n==-1:
        #    return self.sobol_indices(y.truncate_order(n+l+1)) - self.sobol_indices(y.real)
        # else:
        Si_rem = []
        Vi_rem = []
        Si_Ynl, Vi_Ynl = self.sobol_indices(y.truncate_order(n+l+1))
        Si_Yn, Vi_Yn = self.sobol_indices(y.truncate_order(n+1))
        
        for i in range(len(Si_Ynl)):
            Si_rem_tmp = [0]*len(Si_Ynl[i])
            Vi_rem_tmp = [0]*len(Si_Ynl[i])
            for j, s in enumerate(Si_Ynl[i]):
                Si_rem_tmp[j] = s - Si_Yn[i][j]
                Vi_rem_tmp[j] = Vi_Ynl[i][j] - Vi_Yn[i][j]
            Si_rem.append(Si_rem_tmp)
            Vi_rem.append(Vi_rem_tmp)

        return Si_rem, Vi_rem
