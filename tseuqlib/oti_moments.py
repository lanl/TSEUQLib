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

from .oti_util import pyoti_max_bases



class tse_uq():

    def __init__(self, n_dim, mu, extra_bases=[]):
        
        '''
        Args:
            - n_dim (int): number of random variables
            - mu (list): joint central moments of the random variables
            - extra_bases (list): optional, kept for compatibility. Every
              OTI basis above n_dim is treated as a distribution parameter.
        '''
        
        self.n_dim = n_dim
        self.mu = mu
        
        # Bases the moments carry: the distribution parameters they depend on.
        self.mu_bases = sorted(set(b for row in mu for m in row
                                   if hasattr(m, 'get_active_bases')
                                   for b in m.get_active_bases()))

        self.extra_bases = extra_bases
        
        return
    
    
    def check_pyoti_bases(self, y, order):
        '''
        Raise a ValueError, instead of letting pyoti crash, if y and the
        moments need more OTI bases than pyoti supports at this order.
        '''
        bases = set(self.mu_bases)
        if hasattr(y, 'get_active_bases'):
            bases |= set(y.get_active_bases())
        cap = pyoti_max_bases(order)
        if len(bases) > cap:
            raise ValueError(
                f"pyoti supports at most {cap} OTI bases at truncation order "
                f"{order}, and this needs {len(bases)} (inputs and distribution "
                f"parameters). The k'th central moment of an order-n expansion "
                f"needs order n*k: use fewer parameters per run, or a lower "
                f"order.")
    
    
    # @profile
    def expectation(self, y):
        """
        Compute the expectation of the TSE built by the OTI number y.
            
             E[ y ]
            
        OTI bases above n_dim are distribution parameters: the result
        carries the derivative with respect to each of them.
            
        Args:
            - y (oti number):  OTI with the TSE.
        
        """
        return self.conditional_expectation(y, [])


    
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

        # Compute expected value of y.
        if Ey is None:
            Ey = self.expectation(y)
        
        # Change y truncation order of y to k*order
        newOrd = y.order*k
        self.check_pyoti_bases(y, newOrd)
        e1_null = oti.e(1,order=newOrd)
        
        # Dummy way to increase truncation order.
        yprime = y + 0 * e1_null
        
        # integrand for k'th central moment
        yprime = (yprime-Ey)**k
        #print('yprime =', yprime)

        # Evaluate expectation operator
        M = self.expectation(yprime)
        
        return M
    
    
    # =================================================================================
    # @profile
    def conditional_expectation(self, y, basis):
        """
        @brief Computes the conditional expectation of the TSE 
               from the OTI number y.
            
             E[ y | x_{basis[0]}, x_{basis[1]}, ... ]
        
        OTI bases above n_dim are distribution parameters. They are always
        kept, and moments that depend on them are multiplied into each kept
        term, so the result carries the derivative with respect to every
        parameter.
            
        @param[in] y:  OTI with the TSE.
        @param[in] mu: structure with the moments of the variables in y.
        @param[in] basis: (int or list) Basis to excluded from expectation.
        
        """

        if not hasattr(y, 'get_active_bases'):
            return y
        
        if type(basis)== int:
            blist = [basis]
        else:
            blist = list(basis)
        
        # Direction helper from OTI.
        h     = coti.get_dHelp()
        
        order = y.order # Gets the order of the number.
        
        # get active basis in the OTI number
        abases = y.get_active_bases()
        
        if len(abases)!=0:
            nbasis = abases[-1]
        else:
            return np.float64(y.real)
        
        self.check_pyoti_bases(y, order)
        
        # Keep the requested bases and every parameter basis.
        keep = sorted(set(blist) | set(b for b in abases if b > self.n_dim))
        
        # The result is a real number if nothing is kept, otherwise an
        # OTI number in the kept bases.
        if len(keep)==0:
            M = y.real
        else:
            M = oti.sotinum(y.real, len(keep), order)
        
        for ordi in range(1,order+1):
            
            nimord = coti.ndir_order(nbasis,ordi)
            
            for idx in range(nimord):
                
                # Get the imaginary coeff.
                coeff = y[[idx,ordi]]
                
                if coeff == 0:
                    continue
                
                i_mu = idx
                o_mu = ordi
                
                # Start with real direction.
                i_rem = 0
                o_rem = 0
                
                for i in keep:
                    
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
                    mu_oi = 1 # zeroth order central moment
                
                else:
                    
                    # Find the remaining of the derivative.
                    mu_oi = self.mu[o_mu-1][i_mu]
                
                term = coeff*mu_oi
                
                if not hasattr(term, 'get_active_bases'):
                    
                    # A plain moment: add it to the kept direction.
                    if o_rem == 0:
                        M += term
                    else:
                        M[[i_rem,o_rem]] += term
                
                else:
                    
                    # The moment depends on parameters: multiply it into
                    # the kept direction, so d mu/dp reaches every kept term.
                    if o_rem == 0:
                        M = M + term
                    else:
                        e_rem = oti.sotinum(0.0, 1, order)
                        e_rem[[i_rem,o_rem]] = 1.0
                        M = M + term*e_rem
        
        # A real result is a numpy float, as before, so dividing by an exact
        # zero (e.g. a vanishing partial variance) gives inf, not an error.
        if not hasattr(M, 'get_active_bases'):
            M = np.float64(M)
        
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
        


        abases = y.get_active_bases()
        
        # get f0
        f0 = self.expectation(y)
        

        # Compute partial variances:
        
        # Start with the Partial Variances.
        Vtotal = self.central_moment(y, 2)
        
        # store partial variances
        V = [[Vtotal]]
        sobol = [[1]]
        #if type(y)==float:
        if len(abases)==0:
            f_hdmr = [[1]]
            for i in range(self.n_dim):
                V.append([0]*len(list(combinations(range(self.n_dim),(i+1)))))
                sobol.append([0]*len(list(combinations(range(self.n_dim),(i+1)))))
                f_hdmr.append([0]*len(list(combinations(range(self.n_dim),(i+1)))))
            return sobol, V, f_hdmr
        
        # build HDMR
        HDMR, term_struct, srch_struct = self.build_hdmr(y, max_order=max_order, Ey=f0)
        
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
        

        
        # Now we can compute the Sobol from Partial Variances of the HDMRs.
        for ordi in range(1,len(V)):
            V_i = V[ordi]
            nterms = len(V_i)
            sobol_i = [0]*nterms
            for j in range(nterms):
                sobol_i[j] = V[ordi][j]/Vtotal
            sobol.append(sobol_i)
        
        # Orders above the expansion order have zero partial variance:
        # pad them, so the result runs to n_dim (or to max_order).
        top = self.n_dim if max_order is None else min(max_order, self.n_dim)
        for ordi in range(len(HDMR), top+1):
            nterms = len(list(combinations(range(self.n_dim), ordi)))
            HDMR.append([np.float64(0.0)]*nterms)
            V.append([np.float64(0.0)]*nterms)
            sobol.append([np.float64(0.0)]*nterms)
        return sobol, V, HDMR


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
                
                f_hdmr = self.conditional_expectation(y, term) -f0
                
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
        
        # truncate_order(m) keeps the orders below m
        if n==0:
            return y.truncate_order(n+l+1) - y.real
        else:
            return y.truncate_order(n+l+1) - y.truncate_order(n+1)
            
        
    
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
        Si_Ynl, Vi_Ynl, hdmr_Ynl = self.sobol_indices(y.truncate_order(n+l+1))
        Si_Yn, Vi_Yn, hdmr_Yn = self.sobol_indices(y.truncate_order(n+1))
        
        for i in range(len(Si_Ynl)):
            Si_rem_tmp = [0]*len(Si_Ynl[i])
            Vi_rem_tmp = [0]*len(Si_Ynl[i])
            for j, s in enumerate(Si_Ynl[i]):
                Si_rem_tmp[j] = s - Si_Yn[i][j]
                Vi_rem_tmp[j] = Vi_Ynl[i][j] - Vi_Yn[i][j]
            Si_rem.append(Si_rem_tmp)
            Vi_rem.append(Vi_rem_tmp)

        return Si_rem, Vi_rem
