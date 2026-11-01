from .util import create_symbolic_variables
from .oti_util import gen_OTI_indices

import numpy as np
import sympy as sy
from scipy.special import gamma
from scipy.special import beta
from scipy.special import factorial2
from scipy.special import gammaln, polygamma
from scipy import stats

import pyoti.sparse as oti

import math
# from scipy.special import beta

# def gamma_int(n):
#     '''
#     gamma function, only for n is a positive integer
#     '''
#     from math import factorial
#     
#     try:
#         n = int(n)
#     except:
#         raise ValueError('n must be a positive integer (check parameters in Kumaraswamy distribution)')
#     
#     return factorial(n-1)


def gamma_oti(z):
	# from cmath import sin, sqrt, pi, exp

    """
    Lanczos approximation of the gamma function, adapted from:
    wikipedia.org/wiki/Lanczos_approximation
    """
    g = 8
    n = 12
    p = [
        0.9999999999999999298,
        1975.3739023578852322,
        -4397.3823927922428918,
        3462.6328459862717019,
        -1156.9851431631167820,
        154.53815050252775060,
        -6.2536716123689161798,
        0.034642762454736807441,
        -7.4776171974442977377e-7,
        6.3041253821852264261e-8,
        -2.7405717035683877489e-8,
        4.0486948817567609101e-9
    ]
    '''
    g = 7
    n = 9
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    '''
    g = 4.7421875
    n = 15
    p = [0.99999999999999709182, 57.156235665862923517, -59.597960355475491248, 14.136097974741747174, -0.49191381609762019978, .33994649984811888699e-4, .46523628927048575665e-4, -.98374475304879564677e-4, .15808870322491248884e-3, -.21026444172410488319e-3, .21743961811521264320e-3, -.16431810653676389022e-3, .84418223983852743293e-4, -.26190838401581408670e-4, .36899182659531622704e-5]

    EPSILON = 1e-07
    def drop_imag(z):
        if abs(z.imag) <= EPSILON:
            z = z.real
        return z
    
    def gamma(z):
        # z = complex(z)
        if z.real < 0.5:
            y = np.pi / (oti.sin(np.pi * z) * gamma(1 - z))  # Reflection formula
        else:
            z -= 1
            x = p[0]
            for i in range(1, len(p)):
                x += p[i] / (z + i)
            t = z + g + 0.5
            y = np.sqrt(2 * np.pi) * oti.exp((z+0.5)*oti.log(t)) * oti.exp(-t) * x
        return y # drop_imag(y)
    
    return gamma(z)

#def beta(a,b):
#    return (gamma(a)*gamma(b))/gamma(a+b)

def lgamma_oti(z):
    '''
    log Gamma that also accepts OTI numbers: the Taylor series of log Gamma
    at the real part, with coefficients polygamma(k-1, z0)/k!, applied to the
    rest of z. Accurate to about 1e-14 in every derivative (the Lanczos
    gamma_oti above loses digits in higher derivatives).
    '''
    if not hasattr(z, 'order'):
        return gammaln(z)
    z0 = z.real
    dz = z - z0
    out = gammaln(z0) + 0*dz
    term = 1
    for k in range(1, z.order+1):
        term = term*dz
        out = out + polygamma(k-1, z0)/math.factorial(k)*term
    return out

def beta_oti(a,b):
    '''Beta function that also accepts OTI numbers (positive arguments).'''
    # return (gamma_oti(a)*gamma_oti(b))/gamma_oti(a+b)
    if not (hasattr(a, 'order') or hasattr(b, 'order')):
        return beta(a, b)
    return oti.exp(lgamma_oti(a) + lgamma_oti(b) - lgamma_oti(a+b))

# def beta_int(a,b):
#     return (gamma_int(a)*gamma_int(b))/gamma_int(a+b)

def convert_raw_moments_to_central_moments(mu_r, mean):
    '''
    Convert raw moments to central moments
    Inputs:
        mu_r (list):
            list of n'th raw moments
        mean (float):
            mean of PDF
    Returns:
        mu (float):
            n'th central moment of the Kumaraswamy distribution
    '''
    
    from scipy.special import binom
    
    mu = [0]*len(mu_r-1) # don't include mu_0
    
    for r in range(len(mu_r)-1):
        for j in range(r+1):
            mu[r]+=binom(j,r)*((-1)**j)*mu_r[len(mu_r)-1-j]*mean**j
    
    return mu

def get_pdf_params(rv_pdf_name, rv_mean, rv_stdev):
    
    """Compute the parameters for probability distributions.

    This function computes the parameters (rv_a and rv_b) for various probability distributions
    based on the distribution names (rv_pdf_name), mean values (rv_mean), and standard deviations
    (rv_stdev).

    Parameters
    ----------
    rv_pdf_name : list of str
        List of distribution names for each variable.
        Possible values:
            - 'N' for normal (Gaussian) distribution.
            - 'U' for uniform distribution.
            - 'LN' for log-normal distribution.
            - 'B' for beta distribution.
            - 'T' for triangle (symmetric) distribution.
    rv_mean : array_like
        Mean values of the variables.
    rv_stdev : array_like
        Standard deviations of the variables.

    Returns
    -------
    rv_a : ndarray
        Generic PDF shape parameter (a) for each variable.
    rv_b : ndarray
        Generic PDF scale parameter (b) for each variable.

    Notes
    -----
    For each distribution type, the function computes the appropriate parameters rv_a and rv_b
    based on the mean and standard deviation of the variables.

    Examples
    --------
    >>> rv_pdf_name = ['N', 'U', 'LN', 'B', 'T']
    >>> rv_mean = [0, 0, 1, 0.5, 0]
    >>> rv_stdev = [1, 1, 0.5, 0.1, 0.2]
    >>> get_pdf_params(rv_pdf_name, rv_mean, rv_stdev)
    (array([...]), array([...]))
    """
    
    # convert mean and standard deviation to inputs for distribution
    nVar = len(rv_mean)
    rv_a = np.zeros(nVar)
    rv_b = np.zeros(nVar)
    
    c_var=0
    for i_pdf_name in rv_pdf_name:
        if i_pdf_name=='N': # normal (Gaussian)
            rv_a[c_var] = rv_mean[c_var]
            rv_b[c_var] = rv_stdev[c_var]
            
        elif i_pdf_name=='U': # uniform
            rv_a[c_var] = rv_mean[c_var] - np.sqrt(3)*rv_stdev[c_var] # 0.5*(x+y)-a
            rv_b[c_var] = rv_mean[c_var] + np.sqrt(3)*rv_stdev[c_var] # (1/sqrt(12))*(y-x)-b
            
        elif i_pdf_name=='LN': # log-normal
            rv_a[c_var] = 0.5*(2*np.log(rv_mean[c_var])-np.log((rv_mean[c_var]**2+rv_stdev[c_var]**2)/rv_mean[c_var]**2)) # exp(x+0.5*y^2)-a # rv_mean[c_var]  # 0.5*(2*np.log(rv_mean[c_var])-np.log((rv_mean[c_var]**2+rv_stdev[c_var]**2)/rv_mean[c_var]**2)) # exp(x+0.5*y^2)-a
            rv_b[c_var] = np.abs(np.sqrt(np.log((rv_mean[c_var]**2+rv_stdev[c_var]**2)/rv_mean[c_var]**2))) # sqrt((exp(y^2)-1)*exp(2*x+y^2))-b # rv_stdev[c_var] # np.abs(np.sqrt(np.log((rv_mean[c_var]**2+rv_stdev[c_var]**2)/rv_mean[c_var]**2))) # sqrt((exp(y^2)-1)*exp(2*x+y^2))-b
            
        elif i_pdf_name=='B': # beta 
            rv_a[c_var] = (rv_mean[c_var]**2-rv_mean[c_var]**3-rv_mean[c_var]*rv_stdev[c_var])/rv_stdev[c_var]
            rv_b[c_var] = ((-1+rv_mean[c_var])*(-rv_mean[c_var]+rv_mean[c_var]**2+rv_stdev[c_var]))/rv_stdev[c_var]
            
        elif i_pdf_name=='T': # triangle (symmetric) 
            rv_a[c_var] = rv_mean[c_var] - np.sqrt(6)*rv_stdev[c_var]
            rv_b[c_var] = rv_mean[c_var] + np.sqrt(6)*rv_stdev[c_var]
            
        c_var+=1
        
    return rv_a, rv_b

class rv_central_moments():
    
    def __init__(self, rv_params, rv_moments_order):
        '''
        Parameters:
            rv_params = [rv_pdf_name, rv_a, rv_b, ...]
                rv_pdf_name (list): List of strings representing the type of probability distribution for each random variable.
                    List of distribution names for each variable.
                    Possible values:
                        - 'N' for normal (Gaussian) distribution.
                        - 'U' for uniform distribution.
                        - 'LN' for log-normal distribution.
                        - 'B' for beta distribution.
                        - 'T' for triangle (symmetric) distribution.
                        - 'K' for Kumaraswamy distribution
                rv_a (list): List of parameters representing the location or shape of the probability distribution.
                rv_b (list): List of parameters representing the scale or spread of the probability distribution.
            rv_moments_order (int): Highest k'th central moment that will be computed
        '''
        
        self.rv_params = rv_params
        self.rv_moments_order = rv_moments_order
        return
    
    def compute_central_moments(self):
        
        """
        Calculate the moments of random variables based on their probability distribution functions (PDFs).
        
        Returns:
            Tuple: Tuple containing arrays of moments for each random variable.
        """
        
        # extract info from rv_params
        rv_pdf_name, rv_a, rv_b = self.rv_params[:3]
        
        # number of dimensions
        n_dim = len(rv_pdf_name)
        
        pdfInp = sy.MutableDenseNDimArray(np.zeros(n_dim))
        rv_mean = [0]*n_dim
        lb = [0]*n_dim
        ub = [0]*n_dim
        
        rv_mu = [[0]*n_dim for _ in range(self.rv_moments_order)]
        
        
        for i, pdfi in enumerate(rv_pdf_name):
            
            if pdfi == "U": # Uniform
                '''
                Currently only supports integers for rv_a and rv_b
                so that perturbations can be made. Requires the gamma function
                to generate central moments of random variables.
                '''
                
                lb[i]    = rv_a[i]
                ub[i]    = rv_b[i]
                rv_mean[i] = (rv_a[i] + (rv_b[i]-rv_a[i])/2)
                
                for j in range(1,self.rv_moments_order+1):
                    if j%2 == 1:
                        rv_mu[j-1][i] = 0
                    else:
                        rv_mu[j-1][i] = ((rv_a[i]-rv_b[i])**j + (rv_b[i] - rv_a[i])**j)/(2**(j+1)*(j+1))
                
                def mgf_uniform(a, b, e, order):
                    '''
                    Moment Generating function for Uniform Distribution.
                    - x: List of inputs.
                    - a: lower bound
                    - b: upper bound
                    - e: evaluation point of t
                    - alg: (Optional), algebra backend to use.
                    
                    '''
                    # Notice important things:
                    # 1. Real part is zero.
                    # 2. The order is increased by 1 because of l'Hopital's rule.
                    t = oti.e(e, order=order+1) 
                    
                    # Evaluate the Moment generating function 
                    num = (oti.exp(t*b)-oti.exp(t*a))
                    den = (t*(b-a))
                    
                    # The next line essentially applies L'Hopital's rule.
                    MGF = num.extract_im(e)/den.extract_im(e)
                    
                    return MGF
                
                mgf = mgf_uniform(rv_a[i], rv_b[i], 1, self.rv_moments_order)
                '''
                for j in range(1,self.rv_moments_order):
                    
                    rv_mu.append(
                        n_dim*[mgf.get_deriv([[1,j]])]
                    )
                '''    
                
            elif pdfi == "N": # Normal
                
                lb[i]    = -np.inf
                ub[i]    = np.inf
                rv_mean[i] = rv_a[i]
                
                for j in range(1,self.rv_moments_order+1):
                    if j%2 == 1:
                        rv_mu[j-1][i] = 0
                    else:
                        rv_mu[j-1][i] = rv_b[i]**j*factorial2((j-1), exact = True)
            
            elif pdfi == "LN": # Log-Normal
                
                lb[i]    = -np.inf
                ub[i]    = np.inf
                rv_mean[i] = rv_a[i]
                
                raw_moments = [[] for i in range(self.rv_moments_order+1)]
                raw_moments[0] = [1 for i in range(n_dim)]
                for j in range(1,self.rv_moments_order+1):
                    mu = shape[i]
                    sigma_2 = scale[i]
                    raw_moments[j].append(np.exp(j*rv_a[i] + .5*j**2*rv_b[i]))
                    central_moment = 0
                    for k in range(0,j+1):
                        central_moment = central_moment + math.comb(j,k)*(-1)**k*raw_moments[j- k][0]*rv_mean[i]**k
                    rv_mu[j-1][i] = central_moment
            
            elif pdfi == "T": # Triangular
                
                lb[i]    = rv_a[i]
                ub[i]    = rv_b[i]
                rv_mean[i] = 0.5*(rv_a[i]+rv_b[i])
                
                
                def mgf_triangular( a, b, c, e, order):
                    '''
                    Moment Generating function for Uniform Distribution.
                    - x: List of inputs.
                    - a: lower bound
                    - b: upper bound
                    - c: mode
                    - e: evaluation point of t
                    - alg: (Optional), algebra backend to use.
                    
                    '''
                    # Notice important things:
                    # 1. Real part is zero.
                    # 2. The order is increased by 1 because of l'Hopital's rule.
                    t = oti.e(1, order=order+1) 
                    
                    # Evaluate the Moment generating function 
                    # num = (2*((b-c)*oti.exp(a*t)-(b-a)*oti.exp(c*t)+(c-a)*oti.exp(b*t)))
                    # den = ((b-a)*(c-a)*(b-c)*(t**2))
                    # num = (-2*((b-c)*oti.exp(a*t)-(b+a)*oti.exp(c*t)+(c-a)*oti.exp(b*t)))
                    # den = ((a-c)*(a-b)*(c-b)*t**2)
                    # num = (oti.exp(t*b)-oti.exp(t*a))
                    # den = (t*(b-a))
                    
                    num = (2*oti.exp(t*a)*(oti.exp(t*(c-a))*(t*(c-a)-1)+1)) + (2*oti.exp(t*c)*(-t*(b-c)+1-oti.exp(-t*(b-c))))
                    den = t**2*(b-a)*(c-a) + t**2*(b-a)*(b-c)
                    
                    # num = 4*(oti.exp((a*t)/2) - oti.exp((b*t)/2))**2
                    # den = t**2*(a-b)**2
                    
                    print(num)
                    print(num.extract_im(e))
                    print(den)
                    print(den.extract_im(e))
                    # The next line essentially applies L'Hopital's rule.
                    MGF = num.extract_im(e)/den.extract_im(e)
                    
                    print(MGF)
                    
                    return MGF
                
                # mgf = mgf_triangular(rv_a[i], rv_b[i], (rv_a[i]+rv_b[i])/2, 1, self.rv_moments_order)
                # 
                # raw_moments = [[] for i in range(self.rv_moments_order+1)]
                # raw_moments[0] = [1 for i in range(n_dim)]
                # for j in range(1,self.rv_moments_order+1):
                #     raw_moments[j].append(mgf.get_deriv([[1,j]]))
                #     central_moment = 0
                #     for k in range(0,j+1):
                #         central_moment = central_moment + math.comb(j,k)*(-1)**k*raw_moments[j- k][0]*rv_mean[i]**k
                #     rv_mu[j-1][i] = central_moment
                
                # for j in range(1,self.rv_moments_order+1):
                #     if j%2 == 1:
                #         rv_mu[j-1][i] = 0
                #     else:
                #         coeff = (2**(j-1))*((2**(j-1))-1)*3
                #         print('coeff =', coeff)
                #         rv_mu[j-1][i] = ((rv_a[i]-rv_b[i])**j)/coeff
                
                c = (rv_a[i]+rv_b[i])/2
                
                for k in range(1,self.rv_moments_order+1):
                    if k%2 == 1:
                        rv_mu[k-1][i] = 0
                    else:
                        coeff = (2/((1+k)*(2+k)))
                        term1 = 3**-k*(2*c-rv_b[i]-rv_a[i])**k
                        term2 = ((-3**-k*(2*c-rv_b[i]-rv_a[i])**k + 3**-k*(-c+2*rv_b[i]-rv_a[i])**k)*(-c+2*rv_b[i]-rv_a[i])**2)/(9*(-c+rv_b[i])*(rv_b[i]-rv_a[i]))
                        term3 = ((-3**-k*(2*c-rv_b[i]-rv_a[i])**k + 3**-k*(-c-rv_b[i]+2*rv_a[i])**k)*(-c-rv_b[i]+2*rv_a[i])**2)/(9*(c-rv_a[i])*(rv_b[i]-rv_a[i]))
                        print('coeff =', coeff)
                        rv_mu[k-1][i] = coeff*(term1 + term2 + term3)
                
                
                ## for j in range(1, self.rv_moments_order+1):
                ##     print(mgf.get_deriv([[1,j]]))
                ##     input('i')
                ##     rv_mu[j-1][i] = mgf.get_deriv([[1,j]])
                ##     
                
                
                # raw_moments = [[] for i in range(self.rv_moments_order+1)]
                # raw_moments[0] = [1 for i in range(n_dim)]
                # 
                # for j in range(1,self.rv_moments_order+1):
                #     c = .5
                #     dist = stats.triang(c, loc = rv_a[i], scale = rv_b[i])
                #     raw_moments[j].append(dist.moment(j))
                #     central_moment = 0
                #     for k in range(0,j+1):
                #         central_moment = central_moment + math.comb(j,k)*(-1)**k*raw_moments[j- k][0]*rv_mean[i]**k
                #     rv_mu[j-1][i] = central_moment
            
            elif pdfi == "K": # Kumaraswamy
                
                lb[i]    = 0
                ub[i]    = 1
                rv_mean[i] = rv_b[i]*beta_oti(1+(1/rv_a[i]), rv_b[i])
                
                raw_moments = [[] for i in range(self.rv_moments_order+1)]
                raw_moments[0] = [1 for i in range(n_dim)]
                for j in range(1,self.rv_moments_order+1):
                    raw_moments[j].append(rv_b[i]*beta_oti(1+(j/rv_a[i]), rv_b[i]))
                    central_moment = 0
                    for k in range(0,j+1):
                        central_moment = central_moment + math.comb(j,k)*(-1)**k*raw_moments[j- k][0]*rv_mean[i]**k
                    rv_mu[j-1][i] = central_moment
        
        self.lb = lb
        self.ub = ub
        self.rv_mean = rv_mean
        self.rv_mu = rv_mu
        
        return


def rv_mean_from_ab(rv_params):
    import pyoti.sparse as oti
    # import scipy as sc
    """
    Calculates the mean and standard deviation of a random variable given its parameters.
    
    Args:
        rv_params (list): List containing parameters of the random variable: [rv_pdf_name, rv_a, rv_b].
    
    Returns:
        list: List containing the random variable name, lower bound, upper bound, mean, and standard deviation.
    """
    
    # create rv central moment object
    rv_moments = rv_central_moments(rv_params, 2)
    # compute moments
    rv_moments.compute_central_moments()
    
    print(rv_moments)
    
    # get mean and standard deviation
    rv_mean = rv_moments.rv_mean
    mx02 = rv_moments.rv_mu[1]
    rv_stdev    = [np.sqrt(mx02[i]) for i in range(len(mx02))]
    
    return [rv_params[0],rv_params[1],rv_params[2], rv_mean, rv_stdev]


def joint_central_moments(rv_mu, rv_moments_order):
    '''
    Compute joint central moments of random variables, assuming all variables are independent
    
    Parameters:
        rv_mu (list):
            list of lists that contain central moments of each random variable
        rv_moments_order (int):
            highest order of central moment
    
    Returns:
        list of lists of joint central moments of random variables
    '''
    
    n_dim = len(rv_mu[0])
    
    # generate indices used for multiplying central moments
    indices = gen_OTI_indices(n_dim, rv_moments_order)
    
    rv_mu_joint = [0]*len(indices)
    k=0 # moment counter
    for mu_k in indices:
        rv_mu_joint_k = [0]*len(mu_k)
        cj=0 # index counter
        for i in mu_k: # loop through indices in mu_k
            tmp=1
            for j in i: # loop through each index in i
                tmp*=rv_mu[j[1]-1][j[0]-1]
            rv_mu_joint_k[cj] = tmp
            cj+=1
        rv_mu_joint[k] = rv_mu_joint_k
        k+=1
    
    return rv_mu_joint
