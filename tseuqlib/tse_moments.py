# closed-form equations of central moments of a Taylor series expansion

import numpy as np
from .util import extract_derivatives
import sympy as sy

class tse_central_moment():
    
    def __init__(self, oti_number, n_dim, tse_order_max, rv_moments):
        
        self.oti_number = oti_number
        self.n_dim = n_dim
        self.tse_order_max = tse_order_max
        
        self.Y0, self.S1, self.S2, self.S3, self.S4, \
        self.S5 = extract_derivatives(oti_number, n_dim, tse_order_max)
        
        self.mx02, self.mx03, \
        self.mx04, self.mx05, self.mx06, self.mx07, self.mx08, self.mx09, \
        self.mx10, self.mx11, self.mx12 = rv_moments[1:]
        
        return
    
    
    # - Expected Value ------------------------------------
    
    # def 
    def tseEvFirst(self):
        """
        Calculate the first-order expected value.
        
        Returns
        -------
        tuple
            First-order expected value.
        """
        self.evFirst = self.Y0
        return self.Y0
    
    
    def tseEvSecond(self):
        """
        Calculate the second-order expected value.
        
        Returns
        -------
        tuple
            Second-order expected values.
        """
        try:
            evFirst = self.evFirst
        except:
            evFirst = self.tseEvFirst()
        
        S2 = self.S2
        mx02 = self.mx02
        
        n = len(mx02)
        nPoints = np.size(S2, 0)
        
        evSecond = np.zeros(nPoints)
        ev2i = np.zeros([nPoints, n])
        
        for c1 in range(nPoints):
            for i in range(n):
                ev2i[c1, i] = 0.5 * S2[c1, i, i] * mx02[i]
                
        for c1 in range(nPoints):
            evSecond[c1] = evFirst[c1] + np.sum(ev2i[c1, :])
        
        self.evSecond = evSecond
        return evSecond
    
    
    def tseEvThird(self):
        """
        Calculate the third-order expected value.
        
        Returns
        -------
        tuple
            Third-order expected values.
        """
        
        try:
            evSecond = self.evSecond
        except:
            evSecond = self.tseEvSecond()
        
        S3 = self.S3
        mx03 = self.mx03
        
        n = len(mx03)
        nPoints = np.size(S3, 0)

        evThird = np.zeros(nPoints)
        ev3i = np.zeros([nPoints, n])

        for c1 in range(nPoints):
            for i in range(n):
                ev3i[c1, i] = (1 / 6) * S3[c1, i, i, i] * mx03[i]

        for c1 in range(nPoints):
            evThird[c1] = evSecond[c1] + sum(ev3i[c1, :])
        
        self.evThird = evThird
        return evThird
    
    
    def tseEvFourth(self):
        """
        Calculate the fourth-order expected value.
        
        Returns
        -------
        tuple
            Fourth-order expected values.
        """
        try:
            evThird = self.evThird
        except:
            evThird = self.tseEvThird()
        
        S4 = self.S4
        mx02 = self.mx02
        mx04 = self.mx04
        
        n = len(mx02)
        nPoints = np.size(S4, 0)
        
        evFourth = np.zeros(nPoints)
        ev4i = np.zeros([nPoints, n])
        ev4ij = np.zeros([nPoints, n, n])
        
        for c1 in range(nPoints):
            for i in range(n):
                ev4i[c1, i] = (1 / 24) * S4[c1, i, i, i, i] * mx04[i]
                for j in range(n):
                    if i != j:
                        ev4ij[c1, i, j] = (1 / 8) * S4[c1, i, i, j, j] * mx02[i] * mx02[j]
                        
        for c1 in range(nPoints):
            evFourth[c1] = evThird[c1] + sum(ev4i[c1, :]) + np.sum(ev4ij[c1, :, :])
        
        self.evFourth = evFourth
        return evFourth
    
    
    def tseEvFifth(self):
        """
        Calculate the fifth-order expected value.
        
        Returns
        -------
        tuple
            Fifth-order expected values.
        """
        try:
            evFourth = self.evFourth
        except:
            evFourth = self.tseEvFourth()
        
        S5 = self.S5
        mx02 = self.mx02
        mx03 = self.mx03
        mx05 = self.mx05
        
        n = len(mx02)
        nPoints = np.size(S5, 0)
        
        evFifth = np.zeros(nPoints)
        ev5i = np.zeros([nPoints, n])
        ev5ij = np.zeros([nPoints, n, n])
        
        for c1 in range(nPoints):
            for i in range(n):
                ev5i[c1, i] = (S5[c1, i, i, i, i, i] * mx05[i]) / 120
                for j in range(n):
                    if i != j:
                        # over ordered pairs this one term covers both (2, 3)
                        # and (3, 2); adding its mirror counted them twice
                        ev5ij[c1, i, j] = (S5[c1, i, i, j, j, j] * mx02[i] * mx03[j]) / 12
                                          
        for c1 in range(nPoints):
            evFifth[c1] = evFourth[c1] + sum(ev5i[c1, :]) + np.sum(ev5ij[c1, :, :])
        
        self.evFifth = evFifth
        return evFifth
    
    
    # - Variance  ------------------------------------
    
    def tseVarFirst(self):
        """
        Calculate the variance of the first-order Taylor series expansion.
        
        Returns
        -------
        tuple
            - First-order variance.
            - Main effect Sobol indices of the first order Taylor series expansion.
            - Main effect partial variances of the first order Taylor series expansion.
            (A first-order expansion has no interaction terms.)
        """
        
        S1 = self.S1
        mx02 = self.mx02
        
        n = len(mx02)
        nPoints = np.size(S1, 0)
        
        varFirst = np.zeros(nPoints)
        ViFirst = np.zeros([nPoints, n])
        
        for c1 in range(nPoints):
            for i in range(n):
                ViFirst[c1, i] = mx02[i] * S1[c1, i] ** 2
                
        for c1 in range(nPoints):
            varFirst[c1] = np.sum(ViFirst[c1, :])
            
        # Normalize partial variance to get Sobol indices estimates
        SiFirst = np.zeros([nPoints, n])
        
        for c1 in range(nPoints):
            if varFirst[c1] != 0:
                for i in range(n):
                    SiFirst[c1, i] = ViFirst[c1, i] / varFirst[c1]
        
        self.varFirst = varFirst
        self.SiFirst = SiFirst
        self.ViFirst = ViFirst
        return ([varFirst], [SiFirst], [ViFirst])
    
    def tseVarSecond(self):
        """
        Calculate the second-order variance.
        
        Returns
        -------
        tuple
            - Second-order variance.
            - Main effect Sobol indices of the second order Taylor series expansion.
            - Two variable interaction effect Sobol indices of the second order Taylor series expansion.
            - Main effect partial variances of the second order Taylor series expansion.
            - Two variable interaction effect partial variances of the second order Taylor series expansion.
        """
        
        try:
            varFirst = self.varFirst
            ViFirst = self.ViFirst
        except:
            var_o1, Si_o1, Vi_o1 = self.tseVarFirst()
            varFirst = var_o1[0]
            SiFirst = Si_o1[0]
            ViFirst = Vi_o1[0]
        
        S1 = self.S1
        S2 = self.S2
        mx02 = self.mx02
        mx03 = self.mx03
        mx04 = self.mx04
        
        n = len(mx02)
        nPoints = np.size(S1, 0)
        
        varSecond = np.zeros(nPoints)
        ViSecond = np.zeros([nPoints, n])
        VijSecond = np.zeros([nPoints, n, n])
        
        for c1 in range(nPoints):
            for i in range(n):
                ViSecond[c1, i] = (S2[c1, i, i] ** 2 * mx04[i]) * sy.Rational(.25) \
                                  - (S2[c1, i, i] ** 2 * mx02[i] ** 2) * sy.Rational(.25) \
                                  + S1[c1, i] * S2[c1, i, i] * mx03[i]
            for i in range(n - 1):
                for j in range(i + 1, n):
                    VijSecond[c1, i, j] = (S2[c1, i, j] ** 2 * mx02[i] * mx02[j])
                    
        for c1 in range(nPoints):
            varSecond[c1] = varFirst[c1] + np.sum(ViSecond[c1, :]) + np.sum(VijSecond[c1, :, :])
            
        # Sobol indices
        SiSecond = np.zeros([nPoints, n])
        SijSecond = np.zeros([nPoints, n, n])
        
        for c1 in range(nPoints):
            if varSecond[c1] != 0:
                for i in range(n):
                    ViSecond[c1, i] = ViSecond[c1, i] + ViFirst[c1, i]
                    SiSecond[c1, i] = ViSecond[c1, i] / varSecond[c1]
                    
                if n >= 2:
                    for i in range(n - 1):
                        for j in range(i + 1, n):
                            SijSecond[c1, i, j] = VijSecond[c1, i, j] / varSecond[c1]
        
        self.varSecond = varSecond
        self.SiSecond = SiSecond
        self.SijSecond = SijSecond
        self.ViSecond = ViSecond
        self.VijSecond = VijSecond
        return ([varSecond], [SiSecond, SijSecond], [ViSecond, VijSecond])
    
    
    def tseVarThird(self):
        """
        Calculate the third-order variance.
        
        Returns
        -------
        tuple
            - Third-order variance.
            - Main effect Sobol indices of the third order Taylor series expansion.
            - Two variable interaction effect Sobol indices of the third order Taylor series expansion.
            - Third-order interaction Sobol indices.
            - Main effect partial variances of the third order Taylor series expansion.
            - Third-order joint partial variances. 
            - Two variable interaction effect partial variances of the third order Taylor series expansion.
        """
        
        try:
            varSecond = self.varSecond
            ViSecond = self.ViSecond
            VijSecond = self.VijSecond
        except:
            var_o2, Si_o2, Vi_o2 = self.tseVarSecond()
            varSecond = var_o2[0]
            SiSecond = Si_o2[0]
            SijSecond = Si_o2[1]
            ViSecond = Vi_o2[0]
            VijSecond = Vi_o2[1]
        
        S1 = self.S1
        S2 = self.S2
        S3 = self.S3
        mx02 = self.mx02
        mx03 = self.mx03
        mx04 = self.mx04
        mx05 = self.mx05
        mx06 = self.mx06
        
        n = len(mx02)
        nPoints = np.size(S1, 0)
        
        varThird = np.zeros(nPoints)
        ViThird = np.zeros([nPoints, n])
        VijThird = np.zeros([nPoints, n, n])
        VijkThird = np.zeros([nPoints, n, n, n])
        
        # Partial variance
        for c1 in range(nPoints):
            for i in range(n):
                ViThird[c1, i] = (1 / 36) * S3[c1, i, i, i] ** 2 * mx06[i] \
                                 - (1 / 36) * S3[c1, i, i, i] ** 2 * mx03[i] ** 2 \
                                 + (1 / 3) * S1[c1, i] * S3[c1, i, i, i] * mx04[i] \
                                 + (1 / 6) * S2[c1, i, i] * S3[c1, i, i, i] * mx05[i] \
                                 - (1 / 6) * S2[c1, i, i] * S3[c1, i, i, i] * mx02[i] * mx03[i]
                                 
                for a1 in range(n):
                    if a1 != i:
                        ViThird[c1, i] = ViThird[c1, i] \
                                         + (1 / 4) * S3[c1, a1, a1, i] ** 2 * mx02[i] * mx02[a1] ** 2 \
                                         + S1[c1, i] * S3[c1, a1, a1, i] * mx02[i] * mx02[a1] \
                                         + (1 / 2) * S2[c1, i, i] * S3[c1, a1, a1, i] * mx03[i] * mx02[a1] \
                                         + (1 / 6) * S3[c1, i, i, i] * S3[c1, a1, a1, i] * mx04[i] * mx02[a1]
                                         
                for a1 in range(n - 1):
                    for a2 in range(a1 + 1, n):
                        if (a1 != i) and (a2 != i):
                            ViThird[c1, i] = ViThird[c1, i] + (1 / 2) * S3[c1, a1, a1, i] * S3[c1, a2, a2, i] * mx02[
                                a1] * mx02[a2] * mx02[i]
                                
            for i in range(n - 1):
                for j in range(i + 1, n):
                    VijThird[c1, i, j] = S2[c1, i, j] * S3[c1, i, i, j] * mx03[i] * mx02[j] \
                                          + S2[c1, i, j] * S3[c1, i, j, j] * mx02[i] * mx03[j] \
                                          + (1 / 4) * S3[c1, i, i, j] ** 2 * mx04[i] * mx02[j] \
                                          - (1 / 4) * S3[c1, i, i, j] ** 2 * mx02[j] * mx02[i] ** 2 \
                                          + (1 / 2) * S3[c1, i, i, j] * S3[c1, i, j, j] * mx03[i] * mx03[j] \
                                          + (1 / 4) * S3[c1, i, j, j] ** 2 * mx02[i] * mx04[j] \
                                          - (1 / 4) * S3[c1, i, j, j] ** 2 * mx02[i] * mx02[j] ** 2
                                          
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                        for k in range(j + 1, n):
                            VijkThird[c1, i, j, k] = S3[c1, i, j, k] ** 2 * mx02[i] * mx02[j] * mx02[k]
                            
        for c1 in range(nPoints):
            varThird[c1] = varSecond[c1] + sum(ViThird[c1, :]) + np.sum(VijThird[c1, :, :]) + np.sum(VijkThird[c1, :, :, :])
            
        # Sobol Indices
        SiThird = np.zeros([nPoints, n])
        SijThird = np.zeros([nPoints, n, n])
        SijkThird = np.zeros([nPoints, n, n, n])
        
        for c1 in range(nPoints):
            if varThird[c1] != 0:
                for i in range(n):
                    ViThird[c1, i] = ViThird[c1, i] + ViSecond[c1, i]
                    if n >= 2:
                        for j in range(n):
                            VijThird[c1, i, j] = VijThird[c1, i, j] + VijSecond[c1, i, j]
                            
                for i in range(n):
                    SiThird[c1, i] = (ViThird[c1, i] / varThird[c1])  # Third order main sobol indices
                    
                    if n >= 2:
                        for i in range(n - 1):
                            for j in range(n):
                                SijThird[c1, i, j] = (VijThird[c1, i, j] / varThird[c1])  # Third order interaction (2 Variables) sobol indices
                                
                    if n >= 3:
                        for i in range(n - 2):
                            for j in range(n - 1):
                                for k in range(n):
                                    SijkThird[c1, i, j, k] = VijkThird[c1, i, j, k] / varThird[c1]  # Third order interaction (3 Variables) sobol indices
        
        self.varThird = varThird
        self.SiThird = SiThird
        self.SijThird = SijThird
        self.SijkThird = SijkThird
        self.ViThird = ViThird
        self.VijThird = VijThird
        self.VijkThird = VijkThird
        
        return ([varThird], [SiThird, SijThird, SijkThird], [ViThird, VijThird, VijkThird])
    
    
    def tseVarFourth(self):
        """
        Calculate the total variance of the fourth order Taylor series expansion.
        
        Returns
        -------
        tuple 
            Variance of the fourth order Taylor series expansion.
        """
        
        try:
            varThird = self.varThird
            # ViSecond = self.ViSecond
            # VijSecond = self.VijSecond
        except:
            var_o3, Si_o3, Vi_o3 = self.tseVarThird()
            varThird = var_o3[0]
            SiThird = Si_o3[0]
            SijThird = Si_o3[1]
            SijkThird = Si_o3[2]
            ViThird = Vi_o3[0]
            VijThird = Vi_o3[1]
            VijjThird = Vi_o3[2]
        
        S1 = self.S1
        S2 = self.S2
        S3 = self.S3
        S4 = self.S4
        mx02 = self.mx02
        mx03 = self.mx03
        mx04 = self.mx04
        mx05 = self.mx05
        mx06 = self.mx06
        mx07 = self.mx07
        mx08 = self.mx08
        
        n = len(mx02)
        nPoints = np.size(S1, 0)
        
        varFourth = np.zeros(nPoints)
        mu2Y4i = np.zeros([nPoints, n])
        mu2Y4ij = np.zeros([nPoints, n, n])
        mu2Y4ijk = np.zeros([nPoints, n, n, n])
        mu2Y4ijkl = np.zeros([nPoints, n, n, n, n])
        
        for c1 in range(nPoints):
            for i in range(n):
                mu2Y4i[c1, i] = (S4[c1,i,i,i,i]**2*mx08[i])/576 \
                - (S4[c1,i,i,i,i]**2*mx04[i]**2)/576 \
                + (S1[c1,i]*S4[c1,i,i,i,i]*mx05[i])/12 \
                + (S2[c1,i,i]*S4[c1,i,i,i,i]*mx06[i])/24 \
                + (S3[c1,i,i,i]*S4[c1,i,i,i,i]*mx07[i])/72 \
                - (S2[c1,i,i]*S4[c1,i,i,i,i]*mx02[i]*mx04[i])/24 \
                - (S3[c1,i,i,i]*S4[c1,i,i,i,i]*mx03[i]*mx04[i])/72
                
                for j in range(n):
                    if i != j:
                        mu2Y4ij[c1,i,j] = - (S4[c1,i,i,j,j]**2*mx02[i]**2*mx02[j]**2)/32 \
                        + (S4[c1,i,i,j,j]**2*mx04[i]*mx04[j])/32 \
                        + (S4[c1,i,j,j,j]**2*mx02[i]*mx06[j])/36 \
                        - (S4[c1,i,i,j,j]*S2[c1,j,j]*mx02[i]*mx02[j]**2)/4 \
                        + (S1[c1,i]*S4[c1,i,j,j,j]*mx02[i]*mx03[j])/3 \
                        + (S2[c1,i,j]*S4[c1,i,i,j,j]*mx03[i]*mx03[j])/4 \
                        + (S2[c1,i,j]*S4[c1,i,j,j,j]*mx02[i]*mx04[j])/3 \
                        + (S3[c1,i,i,j]*S4[c1,i,j,j,j]*mx03[i]*mx04[j])/6 \
                        + (S3[c1,i,j,j]*S4[c1,i,i,j,j]*mx03[i]*mx04[j])/4 \
                        + (S3[c1,i,j,j]*S4[c1,i,j,j,j]*mx02[i]*mx05[j])/6 \
                        + (S1[c1,j]*S4[c1,i,i,j,j]*mx02[i]*mx03[j])/2 \
                        + (S4[c1,i,i,i,j]*S4[c1,i,j,j,j]*mx04[i]*mx04[j])/36 \
                        + (S4[c1,i,i,j,j]*S4[c1,i,j,j,j]*mx03[i]*mx05[j])/12 \
                        - (S4[c1,i,i,i,i]*S2[c1,j,j]*mx02[j]*mx04[i])/24 \
                        + (S4[c1,i,i,i,j]*S2[c1,j,j]*mx03[i]*mx03[j])/6 \
                        + (S2[c1,i,i]*S4[c1,j,j,j,j]*mx02[i]*mx04[j])/24 \
                        + (S4[c1,i,i,j,j]*S2[c1,j,j]*mx02[i]*mx04[j])/4 \
                        - (S4[c1,i,i,i,i]*S3[c1,j,j,j]*mx03[j]*mx04[i])/72 \
                        + (S3[c1,i,i,i]*S4[c1,j,j,j,j]*mx03[i]*mx04[j])/72 \
                        + (S4[c1,i,i,i,j]*S3[c1,j,j,j]*mx03[i]*mx04[j])/18 \
                        + (S3[c1,i,i,j]*S4[c1,j,j,j,j]*mx02[i]*mx05[j])/24 \
                        + (S4[c1,i,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx05[j])/12 \
                        + (S4[c1,i,i,i,j]*S4[c1,j,j,j,j]*mx03[i]*mx05[j])/72 \
                        + (S4[c1,i,i,j,j]*S4[c1,j,j,j,j]*mx02[i]*mx06[j])/48 \
                        - (S4[c1,i,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j])/12 \
                        - (S4[c1,i,i,j,j]*S4[c1,j,j,j,j]*mx02[i]*mx02[j]*mx04[j])/48 
                        
                    for k in range(n):
                        if i!=j and i!=k and k!=j:
                            mu2Y4ijk[c1,i,j,k] = + (S4[c1,i,j,k,k]**2*mx02[i]*mx02[j]*mx04[k])/8 \
                            + (S3[c1,i,j,j]*S4[c1,i,k,k,k]*mx02[i]*mx02[j]*mx03[k])/9 \
                            + (S3[c1,i,j,k]*S4[c1,i,j,k,k]*mx02[i]*mx02[j]*mx03[k])/2 \
                            + (S4[c1,i,i,j,j]*S4[c1,i,j,k,k]*mx03[i]*mx02[k]*mx03[j])/8 \
                            + (S4[c1,i,i,j,k]*S4[c1,i,j,j,k]*mx03[i]*mx02[k]*mx03[j])/4 \
                            + (S4[c1,i,j,j,j]*S4[c1,i,i,k,k]*mx03[i]*mx02[k]*mx03[j])/32 \
                            + (S4[c1,i,j,j,k]*S4[c1,i,k,k,k]*mx02[i]*mx02[j]*mx04[k])/18 \
                            + (S2[c1,i,i]*S4[c1,j,j,k,k]*mx02[i]*mx02[j]*mx02[k])/8 \
                            + (S4[c1,i,i,j,k]*S2[c1,j,k]*mx02[i]*mx02[j]*mx02[k])/2 \
                            + (S3[c1,i,i,i]*S4[c1,j,j,k,k]*mx02[j]*mx03[i]*mx02[k])/24 \
                            + (S3[c1,i,i,j]*S4[c1,j,k,k,k]*mx02[i]*mx02[j]*mx03[k])/18 \
                            + (S4[c1,i,i,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[k])/2 \
                            + (S4[c1,i,i,k,k]*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx03[k])/4 \
                            + (5*S4[c1,i,i,i,j]*S4[c1,j,j,k,k]*mx03[i]*mx02[k]*mx03[j])/96 \
                            + (S4[c1,i,i,i,k]*S4[c1,j,j,j,k]*mx03[i]*mx02[k]*mx03[j])/36 \
                            + (S4[c1,i,i,j,k]*S4[c1,j,k,k,k]*mx02[i]*mx02[j]*mx04[k])/9 \
                            + (S4[c1,i,i,k,k]*S4[c1,j,j,k,k]*mx02[i]*mx02[j]*mx04[k])/16 \
                            - (S4[c1,i,i,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k])/8 \
                            - (S4[c1,i,i,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k])/24 \
                            - (S4[c1,i,i,k,k]*S4[c1,j,j,k,k]*mx02[i]*mx02[j]*mx02[k]**2)/16
                            
                        for l in range(n):
                            if i!=j and i!=k and i!=l and j!=k and j!=l and k!=l:
                                mu2Y4ijkl[c1,i,j,k,l] = + (S4[c1,i,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l])/24 \
                                + (S4[c1,i,i,j,k]*S4[c1,j,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/16 \
                                + (S4[c1,i,i,j,l]*S4[c1,j,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/24 \
                                + (S4[c1,i,i,k,l]*S4[c1,j,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/48
        
        for c1 in range(nPoints):
            varFourth[c1] = varThird[c1] + np.sum(mu2Y4i[c1, :]) + np.sum(mu2Y4ij[c1, :, :]) + np.sum(mu2Y4ijk[c1, :, :, :]) + np.sum(mu2Y4ijkl[c1, :, :, :, :])
        
        self.varFourth = varFourth
        return varFourth
    
    
    # - Third Central Moment (tcm) ----------------------------------------------------------------------------------------
    
    def tseTcmFirst(self):
        """
        Calculate the first order third central moment.
        
        Returns
        -------
        tuple
            First order third central moment.
        """
        
        S1 = self.S1
        mx03 = self.mx03
        
        n = len(mx03)
        nPoints = np.size(S1, 0)
        
        tcmFirst = np.zeros(nPoints)
        mu3Y1i = np.zeros([nPoints, n])
        
        for c1 in range(nPoints):
            for i in range(n):
                mu3Y1i[c1, i] = mx03[i] * S1[c1, i]**3
        
        for c1 in range(nPoints):
            tcmFirst[c1] = np.sum(mu3Y1i[c1, :])
        
        self.tcmFirst = tcmFirst
        return tcmFirst
    
    def tseTcmSecond(self):
        """
        Calculate the second order third central moment.
        
        Returns
        -------
        tuple
            Second order third central moment.
        """
        try:
            tcmFirst = self.tcmFirst
        except:
            tcmFirst= self.tseTcmFirst()
        
        S1 = self.S1
        S2 = self.S2
        mx02 = self.mx02
        mx03 = self.mx03
        mx04 = self.mx04
        mx05 = self.mx05
        mx06 = self.mx06
        
        n = len(mx03)
        nPoints = np.size(S1, 0)
        
        tcmSecond = np.zeros(nPoints)
        mu3Y2i = np.zeros([nPoints, n])
        mu3Y2ij = np.zeros([nPoints, n, n])
        mu3Y2ijk = np.zeros([nPoints, n, n, n])
        
        for c1 in range(nPoints):
            for i in range(n):
                mu3Y2i[c1, i] = + (S2[c1, i, i]**3 * mx06[i]) / 8 \
                    + (S2[c1, i, i]**3 * mx02[i]**3) / 4 \
                    - (3 * S1[c1, i]**2 * S2[c1, i, i] * mx02[i]**2) / 2 \
                    + (3 * S1[c1, i]**2 * S2[c1, i, i] * mx04[i]) / 2 \
                    + (3 * S1[c1, i] * S2[c1, i, i]**2 * mx05[i]) / 4 \
                    - (3 * S2[c1, i, i]**3 * mx02[i] * mx04[i]) / 8 \
                    - (3 * S1[c1, i] * S2[c1, i, i]**2 * mx02[i] * mx03[i]) / 2
                for j in range(n):
                    if i != j:
                        mu3Y2ij[c1, i, j] = + (S2[c1, i, j]**3 * mx03[i] * mx03[j]) / 2 \
                            + 3 * S2[c1, i, j]**2 * S1[c1, j] * mx02[i] * mx03[j] \
                            + (3 * S2[c1, i, j]**2 * S2[c1, j, j] * mx02[i] * mx04[j]) / 2 \
                            - (3 * S2[c1, i, i] * S2[c1, j, j]**2 * mx02[i] * mx02[j]**2) / 8 \
                            + (3 * S2[c1, i, i]**2 * S2[c1, j, j] * mx02[i]**2 * mx02[j]) / 8 \
                            - (3 * S2[c1, i, j]**2 * S2[c1, j, j] * mx02[i] * mx02[j]**2) / 2 \
                            + 3 * S1[c1, i] * S2[c1, i, j] * S1[c1, j] * mx02[i] * mx02[j] \
                            + 3 * S1[c1, i] * S2[c1, i, j] * S2[c1, j, j] * mx02[i] * mx03[j] \
                            + (3 * S2[c1, i, i] * S2[c1, i, j] * S2[c1, j, j] * mx03[i] * mx03[j]) / 4
                            
                    for k in range(n):
                        if i != j and i != k and k != j:
                            mu3Y2ijk[c1, i, j, k] = - (S2[c1, i, i] * S2[c1, j, k]**2 * mx02[i] * mx02[j] * mx02[k]) / 4 \
                                + (S2[c1, i, j]**2 * S2[c1, k, k] * mx02[i] * mx02[j] * mx02[k]) / 4 \
                                + S2[c1, i, j] * S2[c1, i, k] * S2[c1, j, k] * mx02[i] * mx02[j] * mx02[k]
                                
        for c1 in range(nPoints):
            tcmSecond[c1] = tcmFirst[c1] + np.sum(mu3Y2i[c1, :]) + np.sum(mu3Y2ij[c1, :, :]) + np.sum(mu3Y2ijk[c1, :, :, :])
        
        self.tcmSecond = tcmSecond
        return tcmSecond
    
    # Third Order Third Central Moment
    def tseTcmThird(self):
        
        try:
            tcmSecond = self.tcmSecond
        except:
            tcmSecond= self.tseTcmSecond()
        
        S1 = self.S1
        S2 = self.S2
        S3 = self.S3
        mx02 = self.mx02
        mx03 = self.mx03
        mx04 = self.mx04
        mx05 = self.mx05
        mx06 = self.mx06
        mx07 = self.mx07
        mx08 = self.mx08
        mx09 = self.mx09
        
        n = len(mx02)
        nPoints = np.size(S1,0)
        
        tcmThird = np.zeros(nPoints)
        mu3Y3i = np.zeros([nPoints,n]); mu3Y3ij = np.zeros([nPoints,n,n]); 
        mu3Y3ijk = np.zeros([nPoints,n,n,n]); mu3Y3ijkl = np.zeros([nPoints,n,n,n,n]);
        
        for c1 in range(nPoints):
            for i in range(n):
                mu3Y3i[c1,i] = + (S3[c1,i,i,i]**3*mx09[i])/216 \
                + (S3[c1,i,i,i]**3*mx03[i]**3)/108 \
                + (S1[c1,i]**2*S3[c1,i,i,i]*mx05[i])/2 \
                + (S1[c1,i]*S3[c1,i,i,i]**2*mx07[i])/12 \
                + (S2[c1,i,i]**2*S3[c1,i,i,i]*mx07[i])/8 \
                + (S2[c1,i,i]*S3[c1,i,i,i]**2*mx08[i])/24 \
                - (S3[c1,i,i,i]**3*mx03[i]*mx06[i])/72 \
                - (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]*mx03[i]**2)/2 \
                - (S1[c1,i]**2*S3[c1,i,i,i]*mx02[i]*mx03[i])/2 \
                - (S1[c1,i]*S3[c1,i,i,i]**2*mx03[i]*mx04[i])/6 \
                - (S2[c1,i,i]**2*S3[c1,i,i,i]*mx02[i]*mx05[i])/4 \
                - (S2[c1,i,i]**2*S3[c1,i,i,i]*mx03[i]*mx04[i])/8 \
                - (S2[c1,i,i]*S3[c1,i,i,i]**2*mx02[i]*mx06[i])/24 \
                - (S2[c1,i,i]*S3[c1,i,i,i]**2*mx03[i]*mx05[i])/12 \
                + (S2[c1,i,i]*S3[c1,i,i,i]**2*mx02[i]*mx03[i]**2)/12 \
                + (S2[c1,i,i]**2*S3[c1,i,i,i]*mx02[i]**2*mx03[i])/4 \
                + (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]*mx06[i])/2 \
                - (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]*mx02[i]*mx04[i])/2
                for j in range(n):
                    if i != j:
                        mu3Y3ij[c1,i,j] = + (S3[c1,i,j,j]**3*mx03[i]*mx06[j])/8 \
                        + (3*S1[c1,i]*S3[c1,i,j,j]**2*mx03[i]*mx04[j])/4 \
                        + (3*S2[c1,i,j]**2*S3[c1,i,j,j]*mx03[i]*mx04[j])/2 \
                        + (3*S2[c1,i,j]*S3[c1,i,j,j]**2*mx03[i]*mx05[j])/4 \
                        + (3*S3[c1,i,i,j]*S3[c1,i,j,j]**2*mx04[i]*mx05[j])/8 \
                        - (S3[c1,i,i,i]*S1[c1,j]**2*mx02[j]*mx03[i])/2 \
                        + (3*S3[c1,i,i,j]*S1[c1,j]**2*mx02[i]*mx03[j])/2 \
                        + (S1[c1,i]**2*S3[c1,j,j,j]*mx02[i]*mx03[j])/2 \
                        + (3*S3[c1,i,j,j]**2*S1[c1,j]*mx02[i]*mx05[j])/4 \
                        + (3*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[i]*mx05[j])/8 \
                        + (S2[c1,i,j]**2*S3[c1,j,j,j]*mx02[i]*mx05[j])/2 \
                        + (3*S3[c1,i,i,j]**2*S2[c1,j,j]*mx04[i]*mx04[j])/8 \
                        + (3*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[i]*mx06[j])/8 \
                        + (S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx02[i]*mx07[j])/24 \
                        + (S3[c1,i,i,j]**2*S3[c1,j,j,j]*mx04[i]*mx05[j])/8 \
                        + (S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx07[j])/8 \
                        - (3*S2[c1,i,i]*S3[c1,i,j,j]**2*mx02[i]**2*mx04[j])/8 \
                        - (S3[c1,i,i,i]*S2[c1,j,j]**2*mx02[j]**2*mx03[i])/8 \
                        + (S3[c1,i,i,i]**2*S2[c1,j,j]*mx02[j]*mx03[i]**2)/24 \
                        - (S2[c1,i,i]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]**2)/24 \
                        + (S2[c1,i,i]**2*S3[c1,j,j,j]*mx02[i]**2*mx03[j])/8 \
                        - (S3[c1,i,i,i]*S3[c1,j,j,j]**2*mx03[i]*mx03[j]**2)/72 \
                        + (S3[c1,i,i,i]**2*S3[c1,j,j,j]*mx03[i]**2*mx03[j])/72 \
                        + 3*S1[c1,i]*S3[c1,i,j,j]*S1[c1,j]*mx02[i]*mx03[j] \
                        + (3*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,j]*mx04[i]*mx04[j])/4 \
                        + (3*S2[c1,i,i]*S3[c1,i,j,j]*S1[c1,j]*mx03[i]*mx03[j])/2 \
                        + 3*S2[c1,i,j]*S3[c1,i,i,j]*S1[c1,j]*mx03[i]*mx03[j] \
                        + S1[c1,i]*S2[c1,i,j]*S3[c1,j,j,j]*mx02[i]*mx04[j] \
                        + (3*S1[c1,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx04[j])/2 \
                        + 3*S2[c1,i,j]*S3[c1,i,j,j]*S1[c1,j]*mx02[i]*mx04[j] \
                        + (S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx03[i]*mx04[j])/2 \
                        + (S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,j,j]*mx03[i]*mx04[j])/2 \
                        + (3*S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx03[i]*mx04[j])/4 \
                        + (3*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]*mx03[i]*mx04[j])/2 \
                        + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,j]*mx03[i]*mx04[j])/2 \
                        + (S1[c1,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx05[j])/2 \
                        + (3*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx05[j])/2 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,j,j,j]*mx04[i]*mx04[j])/12 \
                        + (S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx04[i]*mx04[j])/4 \
                        + (S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[i]*mx05[j])/4 \
                        + (S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx03[i]*mx05[j])/2 \
                        + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx03[i]*mx05[j])/4 \
                        + (S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx06[j])/2 \
                        + (3*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx04[j])/2 \
                        + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx04[i]*mx05[j])/12 \
                        + (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[i]*mx06[j])/4 \
                        + (S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx05[j])/2 \
                        + (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx06[j])/4 \
                        - (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*mx02[i]**2*mx03[j])/4 \
                        - (3*S1[c1,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx02[j]**2)/2 \
                        - (3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*mx02[i]**2*mx03[j])/4 \
                        - (3*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]*mx02[j]**2*mx03[i])/4 \
                        - (S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx02[i]**2*mx04[j])/4 \
                        - (S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]**2)/2 \
                        - (3*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx02[j]**2)/2 \
                        - (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]**2)/4 \
                        - (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]**2)/4 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]**2*mx02[i]*mx03[i]*mx04[j])/8 \
                        - (3*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]*mx03[j])/4 \
                        - (S2[c1,i,j]**2*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j])/2 \
                        - (3*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[i]*mx02[j]*mx04[j])/8 \
                        - (S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]*mx04[j])/12 \
                        - (S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx04[j])/8 \
                        - (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*mx02[i]*mx03[i]*mx03[j])/4 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]*mx02[i]*mx02[j]*mx03[i])/4 \
                        - (S1[c1,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j])/2 \
                        - (3*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx02[j]*mx03[j])/2 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx03[j])/12 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*mx02[i]*mx03[i]*mx03[j])/4 \
                        - (S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j])/4 \
                        - (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]*mx03[i]*mx03[j])/4 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx04[j])/12 \
                        - (S2[c1,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j])/4 \
                        - (S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j])/2 \
                        - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j])/12 \
                        - (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx04[j])/4
                    
                    for k in range(n):
                        if i!=j and i!=k and k!=j:
                            mu3Y3ijk[c1,i,j,k] =  + (S3[c1,i,j,k]**3*mx03[i]*mx03[j]*mx03[k])/6 \
                            - (3*S3[c1,i,j,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2)/4 \
                            - (S3[c1,i,i,i]*S2[c1,j,k]**2*mx02[j]*mx03[i]*mx02[k])/4 \
                            + (3*S2[c1,i,k]**2*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx03[k])/2 \
                            + (3*S3[c1,i,j,k]**2*S2[c1,j,k]*mx02[i]*mx03[j]*mx03[k])/2 \
                            - (S3[c1,i,i,i]*S3[c1,j,k,k]**2*mx02[j]*mx03[i]*mx04[k])/8 \
                            + (3*S3[c1,i,i,j]*S3[c1,j,k,k]**2*mx02[i]*mx03[j]*mx04[k])/8 \
                            + (S3[c1,i,i,k]*S3[c1,j,k,k]**2*mx02[i]*mx02[j]*mx05[k])/4 \
                            + (S3[c1,i,k,k]**2*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx04[k])/8 \
                            + (3*S3[c1,i,j,k]**2*S3[c1,j,k,k]*mx02[i]*mx03[j]*mx04[k])/2 \
                            + (S3[c1,i,k,k]**2*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx05[k])/8 \
                            + (3*S3[c1,i,j,k]**2*S1[c1,k]*mx02[i]*mx02[j]*mx03[k])/2 \
                            + (S2[c1,i,j]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k])/4 \
                            + (3*S3[c1,i,j,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k])/4 \
                            + (S3[c1,i,j,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx05[k])/4 \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2)/4 \
                            - (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2)/2 \
                            - (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2)/8 \
                            - (S3[c1,i,j,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/4 \
                            + (S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*mx03[i]*mx03[j]*mx03[k])/12 \
                            + (9*S1[c1,i]*S3[c1,i,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[k])/4 \
                            + (3*S1[c1,i]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx03[k])/4 \
                            + 3*S2[c1,i,j]*S2[c1,i,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[k] \
                            + 3*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[i]*mx02[j]*mx03[k] \
                            + (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S1[c1,j]*mx02[i]*mx02[j]*mx03[k])/4 \
                            + (3*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[i]*mx03[j]*mx03[k])/2 \
                            + 3*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[i]*mx03[j]*mx03[k] \
                            + (S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx03[k])/2 \
                            + (3*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,k]*mx02[i]*mx03[j]*mx03[k])/4 \
                            + (3*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx04[k])/4 \
                            + (9*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx04[k])/4 \
                            + (3*S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx04[k])/4 \
                            + (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,k]*mx02[i]*mx02[j]*mx04[k])/4 \
                            + 3*S2[c1,i,j]*S3[c1,i,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k] \
                            + (5*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k])/12 \
                            + (S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx03[k])/12 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k])/6 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx03[k])/4 \
                            + (3*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*mx02[i]*mx03[j]*mx04[k])/4 \
                            + (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx02[i]*mx03[j]*mx04[k])/2 \
                            + (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx05[k])/4 \
                            + (S1[c1,i]*S3[c1,i,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k])/3 \
                            + (3*S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k])/2 \
                            + S2[c1,i,k]*S3[c1,i,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k] \
                            + (3*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[k])/4 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k])/2 \
                            - (S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx03[k])/2 \
                            + (3*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,k,k]*mx02[i]*mx03[j]*mx03[k])/4 \
                            + (3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,k]*mx02[i]*mx03[j]*mx03[k])/4 \
                            + (3*S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,k,k]*mx02[i]*mx03[j]*mx03[k])/2 \
                            + (S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx04[k])/2 \
                            + (S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx04[k])/3 \
                            + (3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx04[k])/4 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k])/4 \
                            - (3*S2[c1,i,i]*S3[c1,j,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k])/2 \
                            + 3*S3[c1,i,i,j]*S2[c1,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k] \
                            + (3*S3[c1,i,i,k]*S2[c1,j,j]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k])/2 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k])/24 \
                            + (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k])/24 \
                            + (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx02[i]*mx03[j]*mx04[k])/4 \
                            + (S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx04[k])/2 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx05[k])/12 \
                            - (S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k])/4 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k])/2 \
                            + (S2[c1,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j])/4 \
                            + (S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k])/6 \
                            + (S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k])/2 \
                            + S3[c1,i,i,j]*S3[c1,j,k,k]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k] \
                            + (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k])/4 \
                            - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[k])/12 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[j]*mx03[i]*mx03[k])/4 \
                            + (S2[c1,i,i]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k])/12 \
                            + (3*S3[c1,i,i,j]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx03[j]*mx03[k])/4 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[i]*mx03[j]*mx03[k])/4 \
                            + (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx04[k])/6 \
                            + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k])/2 \
                            + (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k])/8 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx04[k])/12 \
                            + (S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx04[k])/4 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx04[k])/12 \
                            + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx05[k])/6 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx05[k])/8 \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/12 \
                            - (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/6 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/8
                        
                        for l in range(n):
                            if i!=j and i!=k and i!=l and j!=k and j!=l and k!=l:
                                mu3Y3ijkl[c1,i,j,k,l] = \
                                - (S2[c1,i,i]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l])/4 \
                                - (S3[c1,i,i,i]*S3[c1,j,k,l]**2*mx02[j]*mx03[i]*mx02[k]*mx02[l])/12 \
                                + (S3[c1,i,i,l]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l])/2 \
                                + (S3[c1,i,j,l]**2*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/4 \
                                + (S3[c1,i,j,k]**2*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/4 \
                                + (S3[c1,i,j,k]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/12 \
                                + (S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/6 \
                                + (2*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/3 \
                                + S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                + (S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/3 \
                                + (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/6 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/2 \
                                + (3*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/2 \
                                + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/4 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/6 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/12 \
                                + (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/6 \
                                + (2*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/3 \
                                - (S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/4 \
                                - (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/8 \
                                + S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/8 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/8 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/12 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/24 \
                                + (7*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/12 \
                                + (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/6 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/8 \
                                + (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/24 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/8 \
                                + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/4 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/8 \
                                + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/12 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/24
                                
        for c1 in range(nPoints):
            tcmThird[c1] = tcmSecond[c1] + np.sum(mu3Y3i[c1,:]) + np.sum(mu3Y3ij[c1,:,:]) + np.sum(mu3Y3ijk[c1,:,:,:]) + np.sum(mu3Y3ijkl[c1,:,:,:,:])
        
        self.tcmThird = tcmThird
        return tcmThird
    
    
    
    
    # - Fourth Central Moment (fcm) -----------------------------------------------------------------------------------------------------------------------------
    
    # First Order Fourth Central Moment
    def tseFcmFirst(self):
        
        S1 = self.S1
        mx02 = self.mx02
        mx04 = self.mx04
        
        n = len(mx02)
        nPoints = np.size(S1,0)
        
        fcmFirst = np.zeros(nPoints); # ,nTimeSteps
        mu4Y1i = np.zeros([nPoints,n]); mu4Y1ij = np.zeros([nPoints,n,n]);
        
        for c1 in range(nPoints):
            for i in range(n):
                mu4Y1i[c1,i] = mx04[i]*S1[c1,i]**4
                for j in range(n):
                    if i != j:
                        mu4Y1ij[c1,i,j] = 3*S1[c1,i]**2*S1[c1,j]**2*mx02[i]*mx02[j]
                        
        for c1 in range(nPoints):
            fcmFirst[c1] = np.sum(mu4Y1i[c1,:]) + np.sum(mu4Y1ij[c1,:,:])
        
        self.fcmFirst = fcmFirst
        return fcmFirst
    
    
    # Second Order Fourth Central Moment
    def tseFcmSecond(self):
        
        try:
            fcmFirst = self.fcmFirst
        except:
            fcmFirst= self.tseFcmFirst()
        
        S1 = self.S1
        S2 = self.S2
        mx02 = self.mx02
        mx03 = self.mx03
        mx04 = self.mx04
        mx05 = self.mx05
        mx06 = self.mx06
        mx07 = self.mx07
        mx08 = self.mx08
        
        n = len(mx02)
        nPoints = np.size(S1,0)
        
        fcmSecond = np.zeros(nPoints);
        mu4Y2i = np.zeros([nPoints,n]); mu4Y2ij = np.zeros([nPoints,n,n]); 
        mu4Y2ijk = np.zeros([nPoints,n,n,n]); mu4Y2ijkl = np.zeros([nPoints,n,n,n,n]);
        
        for c1 in range(nPoints):
            for i in range(n):
                mu4Y2i[c1,i] = (S2[c1,i,i]**4*mx08[i])/16 \
                - (3*S2[c1,i,i]**4*mx02[i]**4)/16 \
                + (3*S1[c1,i]**2*S2[c1,i,i]**2*mx06[i])/2 \
                + (3*S2[c1,i,i]**4*mx02[i]**2*mx04[i])/8 \
                + (3*S1[c1,i]**2*S2[c1,i,i]**2*mx02[i]**3)/2 \
                + 2*S1[c1,i]**3*S2[c1,i,i]*mx05[i] \
                + (S1[c1,i]*S2[c1,i,i]**3*mx07[i])/2 \
                - (S2[c1,i,i]**4*mx02[i]*mx06[i])/4 \
                - 2*S1[c1,i]**3*S2[c1,i,i]*mx02[i]*mx03[i] \
                - (3*S1[c1,i]*S2[c1,i,i]**3*mx02[i]*mx05[i])/2 \
                + (3*S1[c1,i]*S2[c1,i,i]**3*mx02[i]**2*mx03[i])/2 \
                - 3*S1[c1,i]**2*S2[c1,i,i]**2*mx02[i]*mx04[i]
                for j in range(n):
                    if i != j:
                        mu4Y2ij[c1,i,j] = + (S2[c1,i,j]**4*mx04[i]*mx04[j])/2 \
                        + 4*S2[c1,i,j]**3*S1[c1,j]*mx03[i]*mx04[j] \
                        + 2*S2[c1,i,j]**3*S2[c1,j,j]*mx03[i]*mx05[j] \
                        + (3*S2[c1,i,i]**2*S2[c1,j,j]**2*mx02[i]**2*mx02[j]**2)/16 \
                        + (3*S1[c1,i]**2*S2[c1,j,j]**2*mx02[i]*mx04[j])/2 \
                        + 6*S2[c1,i,j]**2*S1[c1,j]**2*mx02[i]*mx04[j] \
                        + (3*S2[c1,i,i]**2*S2[c1,j,j]**2*mx04[i]*mx04[j])/16 \
                        + (3*S2[c1,i,j]**2*S2[c1,j,j]**2*mx02[i]*mx06[j])/2 \
                        - (3*S1[c1,i]**2*S2[c1,j,j]**2*mx02[i]*mx02[j]**2)/2 \
                        - (3*S2[c1,i,i]**2*S2[c1,j,j]**2*mx02[i]**2*mx04[j])/8 \
                        + (3*S2[c1,i,j]**2*S2[c1,j,j]**2*mx02[i]*mx02[j]**3)/2 \
                        - 3*S2[c1,i,j]**2*S2[c1,j,j]**2*mx02[i]*mx02[j]*mx04[j] \
                        + (3*S2[c1,i,i]*S2[c1,i,j]**2*S2[c1,j,j]*mx02[i]**2*mx02[j]**2)/2 \
                        + 12*S1[c1,i]*S2[c1,i,j]*S1[c1,j]**2*mx02[i]*mx03[j] \
                        + 6*S1[c1,i]*S2[c1,i,j]**2*S1[c1,j]*mx03[i]*mx03[j] \
                        + 6*S2[c1,i,i]*S2[c1,i,j]*S1[c1,j]**2*mx03[i]*mx03[j] \
                        + (3*S1[c1,i]*S2[c1,i,i]*S2[c1,j,j]**2*mx03[i]*mx04[j])/2 \
                        + 3*S1[c1,i]*S2[c1,i,j]*S2[c1,j,j]**2*mx02[i]*mx05[j] \
                        + 6*S1[c1,i]*S2[c1,i,j]**2*S2[c1,j,j]*mx03[i]*mx04[j] \
                        + (3*S2[c1,i,i]*S2[c1,i,j]*S2[c1,j,j]**2*mx03[i]*mx05[j])/2 \
                        + (3*S2[c1,i,i]*S2[c1,i,j]**2*S2[c1,j,j]*mx04[i]*mx04[j])/2 \
                        + 6*S1[c1,i]**2*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx03[j] \
                        + 6*S2[c1,i,j]**2*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx05[j] \
                        - 2*S2[c1,i,j]**3*S2[c1,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - 3*S1[c1,i]*S2[c1,i,i]*S2[c1,j,j]**2*mx02[j]**2*mx03[i] \
                        - 6*S1[c1,i]*S2[c1,i,j]**2*S2[c1,j,j]*mx02[j]**2*mx03[i] \
                        - 3*S2[c1,i,i]*S2[c1,i,j]**2*S2[c1,j,j]*mx02[i]**2*mx04[j] \
                        + (3*S2[c1,i,i]**2*S1[c1,j]*S2[c1,j,j]*mx02[i]**2*mx03[j])/2 \
                        - 6*S1[c1,i]*S2[c1,i,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]*mx03[j] \
                        - 3*S2[c1,i,i]*S2[c1,i,j]*S2[c1,j,j]**2*mx02[j]*mx03[i]*mx03[j] \
                        - 6*S2[c1,i,j]**2*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx02[j]*mx03[j] \
                        + 3*S1[c1,i]*S2[c1,i,i]*S1[c1,j]*S2[c1,j,j]*mx03[i]*mx03[j] \
                        + 12*S1[c1,i]*S2[c1,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx04[j] \
                        + 6*S2[c1,i,i]*S2[c1,i,j]*S1[c1,j]*S2[c1,j,j]*mx03[i]*mx04[j] \
                        - 12*S1[c1,i]*S2[c1,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx02[j]**2 \
                        - 6*S2[c1,i,i]*S2[c1,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[j]**2*mx03[i]
                        
                    for k in range(n):
                        if i!=j and i!=k and k!=j:
                            mu4Y2ijk[c1,i,j,k] = + 3*S2[c1,i,k]**2*S2[c1,j,k]**2*mx02[i]*mx02[j]*mx04[k] \
                            + 3*S2[c1,i,j]**2*S1[c1,k]**2*mx02[i]*mx02[j]*mx02[k] \
                            + (3*S2[c1,i,j]**2*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx04[k])/4 \
                            - (3*S2[c1,i,j]**2*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]**2)/4 \
                            + (15*S2[c1,i,i]*S2[c1,i,j]*S2[c1,j,k]**2*mx03[i]*mx02[k]*mx03[j])/4 \
                            + (9*S2[c1,i,j]*S2[c1,i,k]**2*S2[c1,j,j]*mx03[i]*mx02[k]*mx03[j])/4 \
                            + 6*S2[c1,i,j]**2*S2[c1,i,k]*S2[c1,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 12*S2[c1,i,k]**2*S1[c1,j]*S2[c1,j,k]*mx02[i]*mx02[j]*mx03[k] \
                            + 3*S2[c1,i,i]*S2[c1,j,j]*S1[c1,k]**2*mx02[i]*mx02[j]*mx02[k] \
                            - 3*S2[c1,i,i]*S1[c1,j]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k] \
                            - 2*S2[c1,i,i]*S2[c1,j,k]**2*S1[c1,k]*mx02[i]*mx02[j]*mx03[k] \
                            + 2*S2[c1,i,k]**2*S2[c1,j,j]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k] \
                            - S2[c1,i,i]*S2[c1,j,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k] \
                            + S2[c1,i,k]**2*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k] \
                            + 3*S2[c1,i,j]**2*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k] \
                            + (3*S2[c1,i,i]*S2[c1,j,j]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]**2)/4 \
                            - (3*S2[c1,i,i]*S2[c1,j,j]**2*S2[c1,k,k]*mx02[i]*mx02[j]**2*mx02[k])/4 \
                            + S2[c1,i,i]*S2[c1,j,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            - S2[c1,i,k]**2*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            - 6*S2[c1,i,j]*S2[c1,i,k]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            + 3*S2[c1,i,i]*S2[c1,i,k]*S2[c1,j,j]*S2[c1,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            - 6*S1[c1,i]*S2[c1,i,j]*S1[c1,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k] \
                            + 12*S2[c1,i,j]*S2[c1,i,k]*S1[c1,j]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k] \
                            + 12*S2[c1,i,j]*S2[c1,i,k]*S1[c1,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k] \
                            + 12*S2[c1,i,j]*S2[c1,i,k]*S2[c1,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k] \
                            + 6*S2[c1,i,j]*S2[c1,i,k]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k] \
                            + 6*S2[c1,i,i]*S1[c1,j]*S2[c1,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]
                            
                        for l in range(n):
                            if i!=j and i!=k and i!=l and j!=k and j!=l and k!=l:
                                mu4Y2ijkl[c1,i,j,k,l] = \
                                + (3*S2[c1,i,j]**2*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l])/4 \
                                + (3*S2[c1,i,j]*S2[c1,i,k]*S2[c1,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/2 \
                                + (3*S2[c1,i,j]*S2[c1,i,l]*S2[c1,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/2
                                
        for c1 in range(nPoints):
            fcmSecond[c1] = fcmFirst[c1] + np.sum(mu4Y2i[c1,:]) + np.sum(mu4Y2ij[c1,:,:]) + np.sum(mu4Y2ijk[c1,:,:,:]) + np.sum(mu4Y2ijkl[c1,:,:,:,:])
        
        self.fcmSecond = fcmSecond
        return fcmSecond
    
    # Third Order Fourth Central Moment
    def tseFcmThird(self): 
        
        try:
            fcmSecond = self.fcmSecond
        except:
            fcmSecond= self.tseFcmSecond()
        
        S1 = self.S1
        S2 = self.S2
        S3 = self.S3
        mx02 = self.mx02
        mx03 = self.mx03
        mx04 = self.mx04
        mx05 = self.mx05
        mx06 = self.mx06
        mx07 = self.mx07
        mx08 = self.mx08
        mx09 = self.mx09
        mx10 = self.mx10
        mx11 = self.mx11
        mx12 = self.mx12
        
        n = len(mx02)
        nPoints = np.size(S1,0)
        
        fcmThird = np.zeros(nPoints)
        mu4Y3i = np.zeros([nPoints,n]); mu4Y3ij = np.zeros([nPoints,n,n]); mu4Y3ijk = np.zeros([nPoints,n,n,n]); mu4Y3ijkl = np.zeros([nPoints,n,n,n,n]);
        mu4Y3ijklm = np.zeros([nPoints,n,n,n,n,n]);  mu4Y3ijklmo = np.zeros([nPoints,n,n,n,n,n,n]);
        
        for c1 in range(nPoints):
            for i in range(n):
                mu4Y3i[c1,i] =  + (S3[c1,i,i,i]**4*mx12[i])/1296 \
                - (S3[c1,i,i,i]**4*mx03[i]**4)/432 \
                - (2*S1[c1,i]**3*S3[c1,i,i,i]*mx03[i]**2)/3 \
                + (S1[c1,i]**2*S3[c1,i,i,i]**2*mx08[i])/6 \
                + (S2[c1,i,i]**2*S3[c1,i,i,i]**2*mx10[i])/24 \
                + (S3[c1,i,i,i]**4*mx03[i]**2*mx06[i])/216 \
                + (2*S1[c1,i]**3*S3[c1,i,i,i]*mx06[i])/3 \
                + (S1[c1,i]*S3[c1,i,i,i]**3*mx10[i])/54 \
                + (S2[c1,i,i]**3*S3[c1,i,i,i]*mx09[i])/12 \
                + (S2[c1,i,i]*S3[c1,i,i,i]**3*mx11[i])/108 \
                - (S3[c1,i,i,i]**4*mx03[i]*mx09[i])/324 \
                + S1[c1,i]**2*S2[c1,i,i]*S3[c1,i,i,i]*mx07[i] \
                + (S1[c1,i]*S2[c1,i,i]**2*S3[c1,i,i,i]*mx08[i])/2 \
                + (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]**2*mx09[i])/6 \
                - (S1[c1,i]*S3[c1,i,i,i]**3*mx03[i]*mx07[i])/18 \
                - (S2[c1,i,i]**3*S3[c1,i,i,i]*mx02[i]*mx07[i])/4 \
                - (S2[c1,i,i]**3*S3[c1,i,i,i]*mx03[i]*mx06[i])/12 \
                - (S2[c1,i,i]*S3[c1,i,i,i]**3*mx02[i]*mx09[i])/108 \
                - (S2[c1,i,i]*S3[c1,i,i,i]**3*mx03[i]*mx08[i])/36 \
                - (S2[c1,i,i]**2*S3[c1,i,i,i]**2*mx02[i]**2*mx03[i]**2)/8 \
                + (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]**2*mx03[i]**3)/6 \
                + (S1[c1,i]*S3[c1,i,i,i]**3*mx03[i]**2*mx04[i])/18 \
                - (S1[c1,i]**2*S3[c1,i,i,i]**2*mx03[i]*mx05[i])/3 \
                - (S2[c1,i,i]*S3[c1,i,i,i]**3*mx02[i]*mx03[i]**3)/36 \
                - (S2[c1,i,i]**3*S3[c1,i,i,i]*mx02[i]**3*mx03[i])/4 \
                + (S2[c1,i,i]**3*S3[c1,i,i,i]*mx02[i]**2*mx05[i])/4 \
                + (S2[c1,i,i]*S3[c1,i,i,i]**3*mx03[i]**2*mx05[i])/36 \
                - (S2[c1,i,i]**2*S3[c1,i,i,i]**2*mx02[i]*mx08[i])/12 \
                - (S2[c1,i,i]**2*S3[c1,i,i,i]**2*mx03[i]*mx07[i])/12 \
                + (S1[c1,i]**2*S3[c1,i,i,i]**2*mx02[i]*mx03[i]**2)/6 \
                + (S2[c1,i,i]**2*S3[c1,i,i,i]**2*mx03[i]**2*mx04[i])/24 \
                + (S2[c1,i,i]**2*S3[c1,i,i,i]**2*mx02[i]**2*mx06[i])/24 \
                + (S2[c1,i,i]**2*S3[c1,i,i,i]**2*mx02[i]*mx03[i]*mx05[i])/6 \
                - S1[c1,i]**2*S2[c1,i,i]*S3[c1,i,i,i]*mx02[i]*mx05[i] \
                - S1[c1,i]**2*S2[c1,i,i]*S3[c1,i,i,i]*mx03[i]*mx04[i] \
                - S1[c1,i]*S2[c1,i,i]**2*S3[c1,i,i,i]*mx02[i]*mx06[i] \
                - (S1[c1,i]*S2[c1,i,i]**2*S3[c1,i,i,i]*mx03[i]*mx05[i])/2 \
                - (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]**2*mx02[i]*mx07[i])/6 \
                - (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]**2*mx03[i]*mx06[i])/3 \
                + (S2[c1,i,i]**3*S3[c1,i,i,i]*mx02[i]*mx03[i]*mx04[i])/4 \
                + (S2[c1,i,i]*S3[c1,i,i,i]**3*mx02[i]*mx03[i]*mx06[i])/36 \
                + S1[c1,i]*S2[c1,i,i]**2*S3[c1,i,i,i]*mx02[i]*mx03[i]**2 \
                + S1[c1,i]**2*S2[c1,i,i]*S3[c1,i,i,i]*mx02[i]**2*mx03[i] \
                + (S1[c1,i]*S2[c1,i,i]**2*S3[c1,i,i,i]*mx02[i]**2*mx04[i])/2 \
                + (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]**2*mx02[i]*mx03[i]*mx04[i])/3 
                
                for j in range(n):
                    if i != j:
                        mu4Y3ij[c1,i,j] = + (S3[c1,i,i,j]**4*mx04[j]*mx08[i])/16 \
                        + 2*S1[c1,i]**3*S3[c1,i,j,j]*mx02[j]*mx04[i] \
                        + (S1[c1,i]*S3[c1,i,i,j]**3*mx03[j]*mx07[i])/2 \
                        + (2*S2[c1,i,j]**3*S3[c1,i,i,i]*mx03[j]*mx06[i])/3 \
                        + (S2[c1,i,i]**3*S3[c1,i,j,j]*mx02[j]*mx07[i])/4 \
                        + 2*S2[c1,i,j]**3*S3[c1,i,i,j]*mx04[j]*mx05[i] \
                        + (S2[c1,i,i]*S3[c1,i,i,j]**3*mx03[j]*mx08[i])/4 \
                        + (S2[c1,i,j]*S3[c1,i,i,j]**3*mx04[j]*mx07[i])/2 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]**3*mx03[j]*mx09[i])/12 \
                        + (S3[c1,i,i,i]*S3[c1,i,j,j]**3*mx06[i]*mx06[j])/24 \
                        + (S3[c1,i,i,i]**3*S3[c1,i,j,j]*mx02[j]*mx10[i])/108 \
                        + (S3[c1,i,i,j]**3*S3[c1,i,j,j]*mx05[j]*mx07[i])/4 \
                        - (2*S3[c1,i,i,i]*S1[c1,j]**3*mx03[i]*mx03[j])/3 \
                        + (2*S1[c1,i]**3*S3[c1,j,j,j]*mx03[i]*mx03[j])/3 \
                        + (S3[c1,i,i,j]**3*S1[c1,j]*mx04[j]*mx06[i])/2 \
                        - (S3[c1,i,i,i]*S2[c1,j,j]**3*mx03[i]*mx06[j])/12 \
                        + (S2[c1,i,i]**3*S3[c1,j,j,j]*mx03[j]*mx06[i])/12 \
                        + (S3[c1,i,i,i]**3*S2[c1,j,j]*mx02[j]*mx09[i])/108 \
                        - (S2[c1,i,i]*S3[c1,j,j,j]**3*mx02[i]*mx09[j])/108 \
                        + (S3[c1,i,i,j]**3*S2[c1,j,j]*mx05[j]*mx06[i])/4 \
                        - (S3[c1,i,i,i]*S3[c1,j,j,j]**3*mx03[i]*mx09[j])/324 \
                        + (S3[c1,i,i,i]**3*S3[c1,j,j,j]*mx03[j]*mx09[i])/324 \
                        + (S3[c1,i,i,j]**3*S3[c1,j,j,j]*mx06[i]*mx06[j])/24 \
                        - (S3[c1,i,i,i]**2*S2[c1,j,j]**2*mx02[j]**2*mx03[i]**2)/24 \
                        + (S2[c1,i,i]**2*S3[c1,j,j,j]**2*mx02[i]**2*mx03[j]**2)/12 \
                        + (S3[c1,i,i,i]**2*S3[c1,j,j,j]**2*mx03[i]**2*mx03[j]**2)/432 \
                        + (3*S1[c1,i]**2*S3[c1,i,i,j]**2*mx02[j]*mx06[i])/2 \
                        + (3*S1[c1,i]**2*S3[c1,i,j,j]**2*mx04[i]*mx04[j])/2 \
                        + (3*S2[c1,i,i]**2*S3[c1,i,i,j]**2*mx02[j]*mx08[i])/8 \
                        + (S2[c1,i,j]**2*S3[c1,i,i,i]**2*mx02[j]*mx08[i])/6 \
                        + (3*S2[c1,i,i]**2*S3[c1,i,j,j]**2*mx04[j]*mx06[i])/8 \
                        + (3*S2[c1,i,j]**2*S3[c1,i,i,j]**2*mx04[j]*mx06[i])/2 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]**3*mx03[i]**2*mx06[j])/12 \
                        + (S3[c1,i,i,i]**2*S3[c1,i,i,j]**2*mx02[j]*mx10[i])/24 \
                        + (S3[c1,i,i,i]**2*S3[c1,i,j,j]**2*mx04[j]*mx08[i])/24 \
                        + (3*S3[c1,i,i,j]**2*S3[c1,i,j,j]**2*mx06[i]*mx06[j])/16 \
                        + (S3[c1,i,i,i]**2*S1[c1,j]**2*mx02[j]*mx06[i])/6 \
                        - (2*S2[c1,i,j]**3*S3[c1,j,j,j]*mx03[i]*mx03[j]**2)/3 \
                        + (S3[c1,i,i,i]**2*S2[c1,j,j]**2*mx04[j]*mx06[i])/24 \
                        + (S3[c1,i,i,i]**2*S3[c1,j,j,j]**2*mx06[i]*mx06[j])/432 \
                        - (S1[c1,i]**2*S3[c1,j,j,j]**2*mx02[i]*mx03[j]**2)/6 \
                        + (S3[c1,i,i,i]**2*S2[c1,j,j]**2*mx03[i]**2*mx04[j])/24 \
                        - (S2[c1,i,i]**2*S3[c1,j,j,j]**2*mx03[j]**2*mx04[i])/12 \
                        + (3*S3[c1,i,i,j]**2*S2[c1,j,j]**2*mx02[j]**3*mx04[i])/8 \
                        - (S2[c1,i,i]**2*S3[c1,j,j,j]**2*mx02[i]**2*mx06[j])/24 \
                        - (S3[c1,i,i,i]**2*S3[c1,j,j,j]**2*mx03[i]**2*mx06[j])/216 \
                        + (3*S3[c1,i,i,j]*S2[c1,j,j]**3*mx02[i]*mx02[j]**2*mx03[j])/4 \
                        - (3*S3[c1,i,i,j]**2*S2[c1,j,j]**2*mx02[j]*mx04[i]*mx04[j])/4 \
                        - (S2[c1,i,j]**2*S3[c1,j,j,j]**2*mx02[i]*mx03[j]*mx05[j])/3 \
                        - (3*S3[c1,i,j,j]**2*S2[c1,j,j]**2*mx02[i]*mx02[j]*mx06[j])/4 \
                        + (S3[c1,i,i,j]*S3[c1,j,j,j]**3*mx02[i]*mx03[j]**2*mx04[j])/36 \
                        - (S3[c1,i,i,j]**2*S3[c1,j,j,j]**2*mx03[j]*mx04[i]*mx05[j])/12 \
                        - (S3[c1,i,j,j]**2*S3[c1,j,j,j]**2*mx02[i]*mx03[j]*mx07[j])/12 \
                        + (S2[c1,i,j]**2*S3[c1,j,j,j]**2*mx02[i]*mx02[j]*mx03[j]**2)/6 \
                        + (3*S3[c1,i,j,j]**2*S2[c1,j,j]**2*mx02[i]*mx02[j]**2*mx04[j])/8 \
                        + (S3[c1,i,i,j]**2*S3[c1,j,j,j]**2*mx02[j]*mx03[j]**2*mx04[i])/24 \
                        + (S3[c1,i,j,j]**2*S3[c1,j,j,j]**2*mx02[i]*mx03[j]**2*mx04[j])/24 \
                        + 2*S1[c1,i]*S2[c1,i,j]**2*S3[c1,i,i,i]*mx02[j]*mx06[i] \
                        + 3*S1[c1,i]**2*S2[c1,i,i]*S3[c1,i,j,j]*mx02[j]*mx05[i] \
                        + 6*S1[c1,i]**2*S2[c1,i,j]*S3[c1,i,i,j]*mx02[j]*mx05[i] \
                        + (3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,j]**2*mx02[j]*mx07[i])/2 \
                        + (3*S1[c1,i]*S2[c1,i,i]**2*S3[c1,i,j,j]*mx02[j]*mx06[i])/2 \
                        + 6*S1[c1,i]*S2[c1,i,j]**2*S3[c1,i,i,j]*mx03[j]*mx05[i] \
                        + 6*S1[c1,i]**2*S2[c1,i,j]*S3[c1,i,j,j]*mx03[j]*mx04[i] \
                        + (3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,j]**2*mx04[j]*mx05[i])/2 \
                        + 3*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,j]**2*mx03[j]*mx06[i] \
                        + 3*S1[c1,i]*S2[c1,i,j]**2*S3[c1,i,j,j]*mx04[i]*mx04[j] \
                        + S2[c1,i,i]*S2[c1,i,j]**2*S3[c1,i,i,i]*mx02[j]*mx07[i] \
                        + S1[c1,i]**2*S3[c1,i,i,i]*S3[c1,i,j,j]*mx02[j]*mx06[i] \
                        + (S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,i,j]**2*mx02[j]*mx08[i])/2 \
                        + 3*S2[c1,i,i]*S2[c1,i,j]**2*S3[c1,i,i,j]*mx03[j]*mx06[i] \
                        + 3*S1[c1,i]**2*S3[c1,i,i,j]*S3[c1,i,j,j]*mx03[j]*mx05[i] \
                        + (3*S2[c1,i,i]**2*S2[c1,i,j]*S3[c1,i,i,j]*mx02[j]*mx07[i])/2 \
                        + (S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]**2*mx04[j]*mx06[i])/2 \
                        + (S1[c1,i]*S3[c1,i,i,i]**2*S3[c1,i,j,j]*mx02[j]*mx08[i])/6 \
                        + (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]**2*mx03[j]*mx07[i])/2 \
                        + 3*S2[c1,i,i]*S2[c1,i,j]**2*S3[c1,i,j,j]*mx04[j]*mx05[i] \
                        + (3*S2[c1,i,i]**2*S2[c1,i,j]*S3[c1,i,j,j]*mx03[j]*mx06[i])/2 \
                        + (5*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,j]**2*mx05[i]*mx05[j])/6 \
                        + (3*S1[c1,i]*S3[c1,i,i,j]**2*S3[c1,i,j,j]*mx04[j]*mx06[i])/2 \
                        + (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]**2*mx05[i]*mx05[j])/8 \
                        + 2*S1[c1,i]*S3[c1,i,i,i]*S1[c1,j]**2*mx02[j]*mx04[i] \
                        + 2*S1[c1,i]*S3[c1,i,i,j]*S1[c1,j]**2*mx03[i]*mx03[j] \
                        + 6*S1[c1,i]**2*S3[c1,i,i,j]*S1[c1,j]*mx02[j]*mx04[i] \
                        + 4*S1[c1,i]**2*S3[c1,i,j,j]*S1[c1,j]*mx03[i]*mx03[j] \
                        + 3*S1[c1,i]*S3[c1,i,i,j]**2*S1[c1,j]*mx03[j]*mx05[i] \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,i,j]**2*mx02[j]*mx09[i])/4 \
                        + (S2[c1,i,i]**2*S3[c1,i,i,i]*S3[c1,i,j,j]*mx02[j]*mx08[i])/4 \
                        + S2[c1,i,j]**2*S3[c1,i,i,i]*S3[c1,i,i,j]*mx03[j]*mx07[i] \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,j,j]**2*mx04[j]*mx07[i])/4 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]**2*S3[c1,i,j,j]*mx02[j]*mx09[i])/12 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]**2*mx03[j]*mx08[i])/2 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]**2*S3[c1,i,i,j]*mx02[j]*mx09[i])/6 \
                        + (3*S2[c1,i,i]**2*S3[c1,i,i,j]*S3[c1,i,j,j]*mx03[j]*mx07[i])/4 \
                        + S2[c1,i,j]**2*S3[c1,i,i,i]*S3[c1,i,j,j]*mx04[j]*mx06[i] \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]**2*mx05[j]*mx06[i])/4 \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]**2*S3[c1,i,j,j]*mx04[j]*mx07[i])/4 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]**2*mx05[j]*mx06[i])/2 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]**2*S3[c1,i,j,j]*mx03[j]*mx08[i])/6 \
                        + (3*S2[c1,i,j]**2*S3[c1,i,i,j]*S3[c1,i,j,j]*mx05[i]*mx05[j])/2 \
                        + (3*S2[c1,i,j]*S3[c1,i,i,j]**2*S3[c1,i,j,j]*mx05[j]*mx06[i])/2 \
                        + S2[c1,i,i]*S3[c1,i,i,i]*S1[c1,j]**2*mx02[j]*mx05[i] \
                        + 3*S2[c1,i,i]*S3[c1,i,i,j]*S1[c1,j]**2*mx03[j]*mx04[i] \
                        + 2*S2[c1,i,j]*S3[c1,i,i,i]*S1[c1,j]**2*mx03[j]*mx04[i] \
                        + S1[c1,i]**2*S3[c1,i,i,i]*S2[c1,j,j]*mx02[j]*mx05[i] \
                        + S1[c1,i]**2*S2[c1,i,i]*S3[c1,j,j,j]*mx03[j]*mx04[i] \
                        + 3*S1[c1,i]**2*S3[c1,i,i,j]*S2[c1,j,j]*mx03[j]*mx04[i] \
                        + (3*S2[c1,i,i]**2*S3[c1,i,i,j]*S1[c1,j]*mx02[j]*mx06[i])/2 \
                        + 2*S2[c1,i,j]**2*S3[c1,i,i,i]*S1[c1,j]*mx03[j]*mx05[i] \
                        + (S1[c1,i]*S2[c1,i,i]**2*S3[c1,j,j,j]*mx03[j]*mx05[i])/2 \
                        + (S1[c1,i]*S3[c1,i,i,i]**2*S2[c1,j,j]*mx02[j]*mx07[i])/6 \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]**2*S1[c1,j]*mx03[j]*mx06[i])/2 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]**2*S1[c1,j]*mx02[j]*mx07[i])/3 \
                        + (3*S2[c1,i,i]**2*S3[c1,i,j,j]*S1[c1,j]*mx03[j]*mx05[i])/2 \
                        + 3*S2[c1,i,j]**2*S3[c1,i,i,j]*S1[c1,j]*mx04[i]*mx04[j] \
                        + (3*S1[c1,i]*S3[c1,i,i,j]**2*S2[c1,j,j]*mx04[j]*mx05[i])/2 \
                        + 3*S2[c1,i,j]*S3[c1,i,i,j]**2*S1[c1,j]*mx04[j]*mx05[i] \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]**2*mx05[j]*mx07[i])/4 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]**2*S3[c1,i,j,j]*mx04[j]*mx08[i])/4 \
                        + (S3[c1,i,i,i]**2*S3[c1,i,i,j]*S3[c1,i,j,j]*mx03[j]*mx09[i])/12 \
                        + S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]**2*mx03[j]*mx05[i] \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]**2*mx04[j]*mx05[i])/4 \
                        + (S1[c1,i]**2*S3[c1,i,i,i]*S3[c1,j,j,j]*mx03[j]*mx05[i])/3 \
                        + (S2[c1,i,i]**2*S3[c1,i,i,i]*S2[c1,j,j]*mx02[j]*mx07[i])/4 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]**2*S2[c1,j,j]*mx02[j]*mx08[i])/12 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]**2*S1[c1,j]*mx03[j]*mx07[i])/2 \
                        + S1[c1,i]**2*S3[c1,i,i,j]*S3[c1,j,j,j]*mx04[i]*mx04[j] \
                        + (3*S2[c1,i,i]**2*S3[c1,i,i,j]*S2[c1,j,j]*mx03[j]*mx06[i])/4 \
                        + S2[c1,i,j]**2*S3[c1,i,i,i]*S2[c1,j,j]*mx04[j]*mx05[i] \
                        + (S3[c1,i,i,i]**2*S3[c1,i,i,j]*S1[c1,j]*mx02[j]*mx08[i])/6 \
                        + (S1[c1,i]*S3[c1,i,i,i]**2*S3[c1,j,j,j]*mx03[j]*mx07[i])/18 \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]**2*S2[c1,j,j]*mx04[j]*mx06[i])/4 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]**2*S2[c1,j,j]*mx03[j]*mx07[i])/6 \
                        + (2*S3[c1,i,i,i]*S3[c1,i,j,j]**2*S1[c1,j]*mx05[i]*mx05[j])/9 \
                        + (S2[c1,i,i]**2*S2[c1,i,j]*S3[c1,j,j,j]*mx04[j]*mx05[i])/2 \
                        + (3*S2[c1,i,i]**2*S3[c1,i,j,j]*S2[c1,j,j]*mx04[j]*mx05[i])/4 \
                        + (S3[c1,i,i,i]**2*S3[c1,i,j,j]*S1[c1,j]*mx03[j]*mx07[i])/6 \
                        + (5*S1[c1,i]*S3[c1,i,i,j]**2*S3[c1,j,j,j]*mx05[i]*mx05[j])/18 \
                        + (9*S2[c1,i,j]*S3[c1,i,i,j]**2*S2[c1,j,j]*mx05[i]*mx05[j])/8 \
                        + (2*S3[c1,i,i,j]**2*S3[c1,i,j,j]*S1[c1,j]*mx05[i]*mx05[j])/3 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx05[i]*mx05[j])/4 \
                        + (S2[c1,i,i]**2*S3[c1,i,i,i]*S3[c1,j,j,j]*mx03[j]*mx07[i])/12 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]**2*S3[c1,j,j,j]*mx03[j]*mx08[i])/36 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]**2*S2[c1,j,j]*mx04[j]*mx07[i])/4 \
                        + (S2[c1,i,i]**2*S3[c1,i,i,j]*S3[c1,j,j,j]*mx04[j]*mx06[i])/4 \
                        + (S2[c1,i,j]**2*S3[c1,i,i,i]*S3[c1,j,j,j]*mx05[i]*mx05[j])/6 \
                        + (S3[c1,i,i,i]**2*S3[c1,i,i,j]*S2[c1,j,j]*mx03[j]*mx08[i])/12 \
                        + (S2[c1,i,i]*S3[c1,i,i,j]**2*S3[c1,j,j,j]*mx05[j]*mx06[i])/4 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]**2*S3[c1,j,j,j]*mx04[j]*mx07[i])/18 \
                        + (S3[c1,i,i,i]**2*S3[c1,i,j,j]*S2[c1,j,j]*mx04[j]*mx07[i])/12 \
                        - S3[c1,i,i,i]*S1[c1,j]**2*S2[c1,j,j]*mx03[i]*mx04[j] \
                        - S2[c1,i,i]*S1[c1,j]**2*S3[c1,j,j,j]*mx02[i]*mx05[j] \
                        - (S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]**2*mx03[i]*mx05[j])/2 \
                        + (S2[c1,i,i]**2*S1[c1,j]*S3[c1,j,j,j]*mx04[i]*mx04[j])/2 \
                        + (S3[c1,i,i,i]**2*S1[c1,j]*S2[c1,j,j]*mx03[j]*mx06[i])/6 \
                        - (S2[c1,i,i]*S1[c1,j]*S3[c1,j,j,j]**2*mx02[i]*mx07[j])/6 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]**2*S3[c1,j,j,j]*mx05[j]*mx07[i])/12 \
                        + (S3[c1,i,i,i]**2*S3[c1,i,i,j]*S3[c1,j,j,j]*mx04[j]*mx08[i])/36 \
                        + (S3[c1,i,i,i]**2*S3[c1,i,j,j]*S3[c1,j,j,j]*mx05[j]*mx07[i])/36 \
                        - (S3[c1,i,i,i]*S1[c1,j]**2*S3[c1,j,j,j]*mx03[i]*mx05[j])/3 \
                        - (S2[c1,i,i]*S2[c1,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx07[j])/4 \
                        - (S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,j,j]**2*mx03[i]*mx07[j])/18 \
                        + (S3[c1,i,i,i]**2*S1[c1,j]*S3[c1,j,j,j]*mx04[j]*mx06[i])/18 \
                        - (S2[c1,i,i]*S2[c1,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx08[j])/12 \
                        - (S3[c1,i,i,i]*S2[c1,j,j]**2*S3[c1,j,j,j]*mx03[i]*mx07[j])/12 \
                        - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,j]**2*mx03[i]*mx08[j])/36 \
                        + (S3[c1,i,i,i]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx05[j]*mx06[i])/36 \
                        - (S2[c1,i,i]*S3[c1,i,j,j]**3*mx02[i]*mx03[i]*mx06[j])/4 \
                        - (3*S3[c1,i,i,j]*S2[c1,j,j]**3*mx02[i]*mx02[j]*mx05[j])/4 \
                        - (S3[c1,i,j,j]**3*S2[c1,j,j]*mx02[j]*mx03[i]*mx06[j])/4 \
                        - (S3[c1,i,i,j]*S3[c1,j,j,j]**3*mx02[i]*mx03[j]*mx07[j])/36 \
                        - (S3[c1,i,j,j]**3*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx06[j])/12 \
                        - (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]**2*mx03[i]**2*mx05[j])/2 \
                        - (S1[c1,i]*S2[c1,i,i]*S3[c1,j,j,j]**2*mx03[i]*mx03[j]**2)/3 \
                        + (3*S1[c1,i]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]**3)/2 \
                        - 3*S1[c1,i]**2*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]**2*mx03[i] \
                        - (3*S2[c1,i,i]*S3[c1,i,j,j]**2*S1[c1,j]*mx02[i]**2*mx05[j])/2 \
                        - (S1[c1,i]*S3[c1,i,i,i]*S3[c1,j,j,j]**2*mx03[j]**2*mx04[i])/9 \
                        - (3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[i]**2*mx05[j])/4 \
                        + (3*S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[j]**3*mx03[i])/4 \
                        + (3*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[j]**3*mx03[i])/2 \
                        - S2[c1,i,i]*S2[c1,i,j]**2*S3[c1,j,j,j]*mx02[i]**2*mx05[j] \
                        - (3*S2[c1,i,i]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[i]**2*mx06[j])/4 \
                        + (S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[j]**3*mx04[i])/4 \
                        + (S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]**3)/6 \
                        - S2[c1,i,j]**2*S3[c1,i,i,j]*S3[c1,j,j,j]*mx03[j]**2*mx04[i] \
                        - (S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx02[i]**2*mx07[j])/12 \
                        - (S2[c1,i,i]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx02[i]**2*mx07[j])/4 \
                        + (3*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]**3)/2 \
                        + (S3[c1,i,i,i]**2*S1[c1,j]*S2[c1,j,j]*mx03[i]**2*mx03[j])/6 \
                        - (S2[c1,i,i]**2*S1[c1,j]*S3[c1,j,j,j]*mx02[i]**2*mx04[j])/2 \
                        - 2*S2[c1,i,j]**2*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]**2 \
                        + (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx03[i]*mx03[j]**3)/12 \
                        - S3[c1,i,i,j]*S1[c1,j]**2*S3[c1,j,j,j]*mx02[i]*mx03[j]**2 \
                        + (S3[c1,i,i,i]**2*S1[c1,j]*S3[c1,j,j,j]*mx03[i]**2*mx04[j])/18 \
                        - (S2[c1,i,i]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]**2*mx05[j])/4 \
                        - (S3[c1,i,i,j]**2*S1[c1,j]*S3[c1,j,j,j]*mx03[j]**2*mx04[i])/2 \
                        + (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]**3)/12 \
                        - (S3[c1,i,i,i]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx03[i]**2*mx05[j])/36 \
                        - (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]**2*mx02[i]*mx03[i]*mx05[j])/2 \
                        - 3*S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[j]*mx03[i]*mx03[j] \
                        - 3*S1[c1,i]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]*mx04[j] \
                        - 2*S1[c1,i]*S2[c1,i,j]**2*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - (2*S1[c1,i]*S2[c1,i,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]*mx04[j])/3 \
                        - (3*S1[c1,i]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[j]*mx03[i]*mx04[j])/2 \
                        - S2[c1,i,j]**2*S3[c1,i,i,i]*S2[c1,j,j]*mx02[i]*mx03[i]*mx04[j] \
                        - (3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[j]*mx03[j]*mx04[i])/2 \
                        - S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,j]**2*mx02[j]*mx03[j]*mx04[i] \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S1[c1,j]*mx02[i]*mx03[i]*mx05[j])/2 \
                        - (3*S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[j]*mx03[i]*mx04[j])/2 \
                        - 3*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[j]*mx03[i]*mx04[j] \
                        - S1[c1,i]**2*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - 3*S2[c1,i,j]**2*S3[c1,i,i,j]*S2[c1,j,j]*mx02[j]*mx03[j]*mx04[i] \
                        - (S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx03[i]*mx03[j]*mx04[j])/3 \
                        - (S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,j,j]**2*mx03[i]*mx03[j]*mx04[j])/3 \
                        - (3*S2[c1,i,i]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[j]*mx04[i]*mx04[j])/4 \
                        - 3*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]*mx05[j] \
                        - 3*S2[c1,i,j]**2*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]*mx03[i]*mx04[j] \
                        - (S1[c1,i]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]*mx05[j])/3 \
                        - (S1[c1,i]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx04[j])/2 \
                        - (3*S2[c1,i,j]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[j]*mx03[i]*mx05[j])/2 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[i]*mx03[i]*mx05[j])/4 \
                        - (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,j,j,j]**2*mx02[i]*mx03[i]*mx06[j])/36 \
                        - (S2[c1,i,j]**2*S3[c1,i,i,i]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx05[j])/3 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[j]*mx04[i]*mx04[j])/2 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[i]*mx03[i]*mx06[j])/4 \
                        - (S2[c1,i,i]**2*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx05[j])/2 \
                        - (S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx03[j]*mx04[i]*mx04[j])/6 \
                        - (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,j,j,j]**2*mx03[j]*mx04[i]*mx04[j])/9 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[j]*mx04[j]*mx05[i])/4 \
                        - (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[j]*mx03[i]*mx05[j])/2 \
                        - (S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx03[i]*mx03[j]*mx05[j])/6 \
                        - (S2[c1,i,i]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx03[j]*mx04[i]*mx04[j])/4 \
                        - (S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx03[i]*mx03[j]*mx05[j])/3 \
                        - (3*S3[c1,i,i,j]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[j]*mx04[i]*mx05[j])/4 \
                        - S2[c1,i,j]**2*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx04[j] \
                        - (3*S3[c1,i,i,j]**2*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]*mx04[j]*mx05[i])/4 \
                        - (S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]*mx06[j])/3 \
                        - (S2[c1,i,j]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx05[j])/2 \
                        - 3*S3[c1,i,i,j]*S1[c1,j]**2*S2[c1,j,j]*mx02[i]*mx02[j]*mx03[j] \
                        - 3*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]*mx04[j] \
                        - S1[c1,i]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j] \
                        - (3*S3[c1,i,i,j]**2*S1[c1,j]*S2[c1,j,j]*mx02[j]*mx03[j]*mx04[i])/2 \
                        - (3*S3[c1,i,j,j]**2*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx02[j]*mx05[j])/2 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[i]*mx07[j])/36 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx03[j]*mx04[j]*mx05[i])/18 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx07[j])/12 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx03[j]*mx04[i]*mx05[j])/18 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx03[j]*mx04[j]*mx05[i])/12 \
                        - (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx03[i]*mx03[j]*mx06[j])/6 \
                        - (S3[c1,i,i,j]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx03[j]*mx04[i]*mx05[j])/4 \
                        - (S3[c1,i,i,j]**2*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[j]*mx04[j]*mx05[i])/4 \
                        - (S2[c1,i,i]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[j]*mx04[i])/4 \
                        - (S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]*mx05[j])/3 \
                        - S2[c1,i,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx05[j] \
                        - S2[c1,i,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx04[j] \
                        - (S3[c1,i,j,j]**2*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx05[j])/2 \
                        - (S3[c1,i,i,j]*S2[c1,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx06[j])/2 \
                        - (S3[c1,i,i,j]*S2[c1,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx05[j])/4 \
                        - (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx02[j]*mx07[j])/12 \
                        - (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]*mx06[j])/6 \
                        - (S3[c1,i,i,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx04[i]*mx05[j])/4 \
                        - (S3[c1,i,i,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx03[j]*mx04[i]*mx04[j])/4 \
                        - (S3[c1,i,j,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx07[j])/4 \
                        - (S3[c1,i,j,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx06[j])/4 \
                        + (3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]**2*mx02[j]**2)/2 \
                        + S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]**2*mx03[j]**2 \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]**2*mx02[j]**2)/2 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[i]**2*mx03[j]**2)/12 \
                        + (S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]**2*mx03[j]**2)/2 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]**2*mx02[i]*mx02[j]**2*mx03[i])/6 \
                        + S2[c1,i,j]**2*S3[c1,i,i,i]*S2[c1,j,j]*mx02[i]*mx02[j]**2*mx03[i] \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[i]**2*mx02[j]*mx03[j])/2 \
                        + (S1[c1,i]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx02[j]*mx03[j]**2)/6 \
                        + (3*S2[c1,i,i]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[i]**2*mx02[j]*mx04[j])/4 \
                        + (3*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]**2*mx03[j])/2 \
                        + (S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx02[i]**2*mx03[j]*mx04[j])/6 \
                        + (S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx02[j]*mx03[i]*mx03[j]**2)/12 \
                        + (S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx02[j]*mx03[i]*mx03[j]**2)/6 \
                        + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]**2*mx02[j]**2*mx03[i]*mx03[j])/4 \
                        + (S2[c1,i,i]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx02[i]**2*mx03[j]*mx04[j])/4 \
                        + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]**2*mx02[j]*mx03[j]**2*mx04[i])/36 \
                        + (S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]**2*mx02[i]*mx02[j]*mx03[j]**2)/6 \
                        + (S2[c1,i,i]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]**2*mx02[j]*mx03[j])/12 \
                        + S2[c1,i,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]**2*mx03[j] \
                        + (S3[c1,i,i,j]*S2[c1,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j]**2)/2 \
                        + (S3[c1,i,i,i]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]**2*mx03[j])/36 \
                        + (S3[c1,i,i,j]*S2[c1,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx02[j]**2*mx04[j])/4 \
                        + (S3[c1,i,i,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]**2*mx03[j]*mx04[i])/4 \
                        + 6*S1[c1,i]*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]*mx02[j]*mx06[i] \
                        + 6*S1[c1,i]*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*mx03[j]*mx05[i] \
                        + S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*mx02[j]*mx07[i] \
                        + 2*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]*mx02[j]*mx07[i] \
                        + 3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*mx03[j]*mx06[i] \
                        + 2*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*mx03[j]*mx06[i] \
                        + 6*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,j]*mx04[j]*mx05[i] \
                        + 6*S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,j]*S1[c1,j]*mx02[j]*mx05[i] \
                        + 4*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,i]*S1[c1,j]*mx02[j]*mx05[i] \
                        + 6*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,j]*S1[c1,j]*mx03[j]*mx04[i] \
                        + 12*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,j]*S1[c1,j]*mx03[j]*mx04[i] \
                        + S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]*mx02[j]*mx08[i] \
                        + S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*mx03[j]*mx07[i] \
                        + S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*mx03[j]*mx07[i] \
                        + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,j]*mx04[j]*mx06[i] \
                        + S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]*mx02[j]*mx06[i] \
                        + 2*S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*mx02[j]*mx06[i] \
                        + 2*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,i]*S1[c1,j]*mx02[j]*mx06[i] \
                        + 3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*mx03[j]*mx05[i] \
                        + 2*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,j]*mx03[j]*mx05[i] \
                        + 2*S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,j]*mx03[j]*mx05[i] \
                        + 6*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]*S1[c1,j]*mx03[j]*mx05[i] \
                        + (3*S1[c1,i]*S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,j,j]*mx04[i]*mx04[j])/2 \
                        + (3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx04[i]*mx04[j])/2 \
                        + (3*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]*mx04[i]*mx04[j])/2 \
                        + 3*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,j]*mx04[i]*mx04[j] \
                        + (9*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S1[c1,j]*mx04[i]*mx04[j])/2 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*mx03[j]*mx08[i])/2 \
                        + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*mx04[j]*mx07[i] \
                        + S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*mx02[j]*mx07[i] \
                        + (S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,j,j,j]*mx03[j]*mx06[i])/3 \
                        + S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*mx03[j]*mx06[i] \
                        + S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,j]*mx03[j]*mx06[i] \
                        + S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,j]*mx03[j]*mx06[i] \
                        + 2*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*mx03[j]*mx06[i] \
                        + S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx04[j]*mx05[i] \
                        + (2*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,j,j,j]*mx04[j]*mx05[i])/3 \
                        + S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx04[j]*mx05[i] \
                        + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]*mx04[j]*mx05[i] \
                        + 3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,j]*mx04[j]*mx05[i] \
                        + 2*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,j]*mx04[j]*mx05[i] \
                        + 2*S1[c1,i]*S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]*mx03[j]*mx04[i] \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*mx03[j]*mx07[i])/2 \
                        + (S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx04[j]*mx06[i])/3 \
                        + (S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,j,j,j]*mx04[j]*mx06[i])/3 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx04[j]*mx06[i])/2 \
                        + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*mx04[j]*mx06[i] \
                        + S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,j]*mx04[j]*mx06[i] \
                        + (5*S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx05[i]*mx05[j])/27 \
                        + (S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx05[i]*mx05[j])/4 \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx05[i]*mx05[j])/4 \
                        + (3*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx05[i]*mx05[j])/4 \
                        + S2[c1,i,i]*S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]*mx03[j]*mx05[i] \
                        + (S1[c1,i]*S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,j,j]*mx04[i]*mx04[j])/3 \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx04[i]*mx04[j])/2 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]*mx04[i]*mx04[j])/2 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx04[j]*mx07[i])/6 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx05[j]*mx06[i])/6 \
                        + (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx05[j]*mx06[i])/3 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx05[j]*mx06[i])/2 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,j,j]*mx04[j]*mx05[i])/3 \
                        + S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx04[j]*mx05[i] \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx06[i]*mx06[j])/12 \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx05[i]*mx05[j])/12 \
                        + (4*S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx05[i]*mx05[j])/27 \
                        - S2[c1,i,i]*S1[c1,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx06[j] \
                        - (S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[i]*mx06[j])/3 \
                        - 2*S1[c1,i]*S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,j,j]*mx02[i]**2*mx04[j] \
                        - 3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]**2*mx04[j] \
                        - 6*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S1[c1,j]*mx02[i]**2*mx04[j] \
                        - S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]**2*mx05[j] \
                        - 2*S1[c1,i]*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]**2 \
                        - 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]**2*mx05[j] \
                        - 6*S1[c1,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[j]**2*mx03[i] \
                        - S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[j]**2*mx04[i] \
                        - S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[j]**2*mx04[i] \
                        - S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]**2*mx06[j] \
                        - 3*S2[c1,i,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]**2*mx04[j] \
                        - 2*S1[c1,i]*S2[c1,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]**2 \
                        - 2*S1[c1,i]*S3[c1,i,j,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]**2 \
                        - (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[i]**2*mx05[j])/6 \
                        - (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx03[i]**2*mx05[j])/3 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx03[i]**2*mx05[j])/2 \
                        - S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]**2 \
                        - S2[c1,i,i]*S2[c1,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]**2 \
                        - S2[c1,i,i]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[i]**2*mx05[j] \
                        - S2[c1,i,i]*S3[c1,i,j,j]*S1[c1,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]**2 \
                        - 2*S2[c1,i,j]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]**2 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[i]**2*mx06[j])/6 \
                        - (S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[j]**2*mx04[i])/2 \
                        - (S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[j]**2*mx04[i])/3 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,j]*S3[c1,j,j,j]*mx03[j]**2*mx04[i])/3 \
                        - (S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]**2*mx06[j])/2 \
                        + S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx02[j]**2*mx03[i] \
                        + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]**2*mx02[j]*mx03[j] \
                        + (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx03[j]**2)/3 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]*mx03[i]**2*mx03[j])/2 \
                        + S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx02[j]**2*mx03[i] \
                        + S1[c1,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]**2*mx03[j] \
                        + (S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]**2*mx02[j]*mx04[j])/2 \
                        + (S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]**2*mx03[i]*mx03[j])/2 \
                        + S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]**2*mx03[i]*mx03[j] \
                        + S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j]**2 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx03[j]**2)/6 \
                        + (S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]**2*mx03[j]*mx04[i])/6 \
                        + (S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j]**2)/2 \
                        + S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]**2*mx03[j] \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]**2*mx02[i]*mx02[j]*mx03[i]*mx03[j])/2 \
                        + (S2[c1,i,j]**2*S3[c1,i,i,i]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[i]*mx03[j])/6 \
                        + (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S2[c1,j,j]*mx02[i]*mx02[j]*mx03[i]*mx04[j])/4 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]**2*mx02[i]*mx03[i]*mx03[j]*mx04[j])/18 \
                        + (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx03[j]*mx04[j])/12 \
                        + (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]**2*mx02[i]*mx02[j]*mx03[j]*mx04[j])/6 \
                        + (S3[c1,i,j,j]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j]*mx04[j])/4 \
                        - 6*S1[c1,i]*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - (2*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx04[j])/3 \
                        - S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx03[i]*mx04[j] \
                        - 2*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,j]*mx02[i]*mx03[i]*mx04[j] \
                        - 3*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]*mx03[j]*mx04[i] \
                        - 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]*mx03[j]*mx04[i] \
                        - 4*S1[c1,i]*S2[c1,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j] \
                        - 6*S1[c1,i]*S3[c1,i,j,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx02[j]*mx03[j] \
                        - (S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx05[j])/3 \
                        - S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx05[j] \
                        - (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx03[i]*mx05[j])/2 \
                        - S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx03[i]*mx05[j] \
                        - 3*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]*mx04[i]*mx04[j] \
                        + S2[c1,i,i]*S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx03[i]*mx03[j] \
                        - 2*S1[c1,i]*S2[c1,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - 2*S1[c1,i]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - 2*S2[c1,i,i]*S2[c1,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - 3*S2[c1,i,i]*S3[c1,i,j,j]*S1[c1,j]*S2[c1,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - 6*S2[c1,i,j]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[j]*mx03[i]*mx03[j] \
                        - 2*S1[c1,i]*S2[c1,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx04[j] \
                        - 6*S2[c1,i,j]*S3[c1,i,j,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx02[j]*mx04[j] \
                        - (S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx06[j])/2 \
                        - (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx06[j])/3 \
                        - S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx03[j]*mx04[i]*mx04[j] \
                        - S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*mx02[i]*mx03[i]*mx04[j] \
                        - (S1[c1,i]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[j]*mx04[i])/3 \
                        - S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,j]*S2[c1,j,j]*mx02[j]*mx03[j]*mx04[i] \
                        - S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx04[j] \
                        - S2[c1,i,i]*S2[c1,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx04[j] \
                        - 3*S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,j]*S2[c1,j,j]*mx02[j]*mx03[i]*mx04[j] \
                        - S1[c1,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx05[j] \
                        - S1[c1,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx04[j] \
                        - 2*S2[c1,i,j]*S3[c1,i,j,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx04[j] \
                        - (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx05[j])/6 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx05[j])/3 \
                        - (S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx04[i]*mx04[j])/2 \
                        - (S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx04[i]*mx04[j])/3 \
                        - (S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx05[j])/2 \
                        - (S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx04[j])/2 \
                        - S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx05[j] \
                        - S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx04[j] \
                        - S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx04[j] \
                        - S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx06[j] \
                        - S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx05[j] \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx06[j])/6 \
                        - (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx04[j]*mx05[i])/6 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx04[i]*mx05[j])/6 \
                        - (S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[j]*mx04[i]*mx04[j])/6 \
                        - (S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx06[j])/2 \
                        - (S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx05[j])/2 \
                        - S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx05[j] \
                        - S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx03[j]*mx04[j] \
                        + (S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[i]*mx03[j])/6 \
                        + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx02[j]*mx03[i]*mx03[j])/4 \
                        + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[i]*mx02[j]*mx03[i]*mx03[j] \
                        + (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[i]*mx03[j])/12 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[i]*mx03[j])/6 \
                        + (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[i]*mx04[j])/6 
                        
                    for k in range(n):
                        if i!=j and i!=k and k!=j:
                            mu4Y3ijk[c1,i,j,k] = + (S3[c1,i,j,k]**4*mx04[i]*mx04[j]*mx04[k])/6 \
                            + 3*S1[c1,i]**2*S3[c1,i,j,k]**2*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S2[c1,i,j]**2*S3[c1,i,j,k]**2*mx02[k]*mx04[i]*mx04[j] \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,j,k]**2*mx02[k]*mx04[j]*mx06[i])/2 \
                            + (3*S3[c1,i,i,k]**2*S3[c1,i,j,j]**2*mx02[k]*mx04[j]*mx06[i])/8 \
                            + (3*S3[c1,i,i,k]**2*S1[c1,j]**2*mx02[j]*mx02[k]*mx04[i])/2 \
                            + (3*S3[c1,i,i,j]**2*S2[c1,j,k]**2*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (3*S3[c1,i,i,k]**2*S2[c1,j,j]**2*mx02[k]*mx04[i]*mx04[j])/8 \
                            + (3*S3[c1,i,k,k]**2*S2[c1,j,k]**2*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (S3[c1,i,i,i]**2*S3[c1,j,j,k]**2*mx02[k]*mx04[j]*mx06[i])/24 \
                            + (S3[c1,i,i,j]**2*S3[c1,j,k,k]**2*mx04[i]*mx04[j]*mx04[k])/8 \
                            + (S3[c1,i,i,k]**2*S3[c1,j,j,k]**2*mx04[i]*mx04[j]*mx04[k])/16 \
                            + (3*S3[c1,i,k,k]**2*S3[c1,j,k,k]**2*mx02[i]*mx02[j]*mx08[k])/16 \
                            + (S2[c1,i,j]**2*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx06[k])/12 \
                            + (3*S3[c1,i,j,k]**2*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx06[k])/4 \
                            - (S3[c1,i,j,k]**3*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k]**2)/3 \
                            + (S3[c1,i,j,k]**2*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx08[k])/12 \
                            - (3*S3[c1,i,k,k]**2*S2[c1,j,j]**2*mx02[i]*mx02[j]**2*mx04[k])/8 \
                            - (S3[c1,i,i,k]**2*S3[c1,j,j,j]**2*mx02[k]*mx03[j]**2*mx04[i])/12 \
                            + (S3[c1,i,k,k]**2*S3[c1,j,j,j]**2*mx02[i]*mx03[j]**2*mx04[k])/24 \
                            - (S2[c1,i,j]**2*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]**2)/12 \
                            + (3*S3[c1,i,j,k]**2*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]**3)/4 \
                            + 2*S2[c1,i,j]**3*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S2[c1,i,j]*S3[c1,i,j,k]**3*mx04[i]*mx03[k]*mx04[j] \
                            + 2*S3[c1,i,i,j]*S3[c1,i,j,k]**3*mx03[k]*mx04[j]*mx05[i] \
                            + (S3[c1,i,i,j]**3*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx07[i])/4 \
                            - (S2[c1,i,i]*S3[c1,j,k,k]**3*mx02[i]*mx03[j]*mx06[k])/4 \
                            - (S3[c1,i,i,i]*S3[c1,j,k,k]**3*mx03[i]*mx03[j]*mx06[k])/48 \
                            + (S3[c1,i,i,j]**3*S3[c1,j,k,k]*mx02[k]*mx04[j]*mx06[i])/4 \
                            + (S3[c1,i,k,k]**3*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx06[k])/48 \
                            + 2*S3[c1,i,j,k]**3*S1[c1,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (S3[c1,i,i,j]**3*S2[c1,k,k]*mx02[k]*mx03[j]*mx06[i])/4 \
                            + S3[c1,i,j,k]**3*S2[c1,k,k]*mx03[i]*mx03[j]*mx05[k] \
                            + (S3[c1,i,j,k]**3*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx06[k])/3 \
                            + 3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,k]**2*mx02[j]*mx02[k]*mx05[i] \
                            + 12*S1[c1,i]*S2[c1,i,j]*S3[c1,i,j,k]**2*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S1[c1,i]*S2[c1,i,j]**2*S3[c1,i,k,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 2*S2[c1,i,j]*S2[c1,i,k]**2*S3[c1,i,i,j]*mx02[j]*mx02[k]*mx05[i] \
                            + 6*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,k]**2*mx02[k]*mx03[j]*mx05[i] \
                            + 6*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,k]**2*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,i]*S2[c1,i,j]**2*S3[c1,i,k,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 6*S2[c1,i,j]*S2[c1,i,k]**2*S3[c1,i,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + (3*S1[c1,i]**2*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[j]*mx02[k]*mx04[i])/2 \
                            + 4*S2[c1,i,j]**2*S2[c1,i,k]*S3[c1,i,i,k]*mx02[j]*mx02[k]*mx05[i] \
                            + (10*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,j,k]**2*mx02[k]*mx04[i]*mx04[j])/3 \
                            + 12*S2[c1,i,j]**2*S2[c1,i,k]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + (2*S1[c1,i]*S3[c1,i,j,j]**2*S3[c1,i,k,k]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + 2*S1[c1,i]*S3[c1,i,k,k]*S1[c1,j]**2*mx02[i]*mx02[j]*mx02[k] \
                            + 6*S1[c1,i]*S3[c1,i,j,k]**2*S1[c1,j]*mx03[i]*mx02[k]*mx03[j] \
                            + 3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,k]**2*mx02[k]*mx03[j]*mx06[i] \
                            + 2*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,k]**2*mx02[k]*mx03[j]*mx06[i] \
                            + 3*S2[c1,i,k]**2*S3[c1,i,i,j]*S3[c1,i,j,j]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]**2*mx02[k]*mx04[j]*mx05[i] \
                            + 6*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,k]**2*mx02[k]*mx04[j]*mx05[i] \
                            + (3*S2[c1,i,j]*S3[c1,i,i,k]**2*S3[c1,i,j,j]*mx02[k]*mx03[j]*mx06[i])/2 \
                            + 3*S2[c1,i,j]**2*S3[c1,i,i,j]*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 6*S2[c1,i,j]**2*S3[c1,i,i,k]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + (3*S2[c1,i,i]*S3[c1,i,j,j]**2*S3[c1,i,k,k]*mx02[k]*mx04[j]*mx05[i])/4 \
                            + (3*S2[c1,i,j]*S3[c1,i,i,j]**2*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx06[i])/2 \
                            + (3*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,i,j,j]**2*mx02[k]*mx04[j]*mx05[i])/2 \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]**2*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + (3*S2[c1,i,j]**2*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (10*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]**2*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (2*S2[c1,i,k]*S3[c1,i,j,j]**2*S3[c1,i,k,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + S1[c1,i]*S3[c1,i,i,i]*S2[c1,j,k]**2*mx02[j]*mx02[k]*mx04[i] \
                            + 2*S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,k]**2*mx03[i]*mx02[k]*mx03[j] \
                            + 3*S2[c1,i,i]*S3[c1,i,k,k]*S1[c1,j]**2*mx02[j]*mx03[i]*mx02[k] \
                            + 6*S2[c1,i,k]*S3[c1,i,i,k]*S1[c1,j]**2*mx02[j]*mx03[i]*mx02[k] \
                            + 2*S1[c1,i]*S2[c1,i,k]**2*S3[c1,j,j,j]*mx03[i]*mx02[k]*mx03[j] \
                            + 6*S2[c1,i,i]*S3[c1,i,j,k]**2*S1[c1,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S1[c1,i]**2*S2[c1,i,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k] \
                            + 6*S1[c1,i]**2*S3[c1,i,j,k]*S2[c1,j,k]*mx02[j]*mx03[i]*mx02[k] \
                            + 3*S1[c1,i]**2*S3[c1,i,k,k]*S2[c1,j,j]*mx02[j]*mx03[i]*mx02[k] \
                            + 4*S2[c1,i,k]**2*S3[c1,i,j,j]*S1[c1,j]*mx03[i]*mx02[k]*mx03[j] \
                            + 6*S1[c1,i]*S2[c1,i,j]**2*S3[c1,j,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 2*S1[c1,i]*S3[c1,i,j,k]**2*S2[c1,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 4*S2[c1,i,k]*S3[c1,i,j,k]**2*S1[c1,j]*mx03[i]*mx03[j]*mx03[k] \
                            + S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,k]**2*mx02[k]*mx03[j]*mx07[i] \
                            + S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]**2*mx02[k]*mx04[j]*mx06[i] \
                            + (3*S3[c1,i,i,j]*S3[c1,i,i,k]**2*S3[c1,i,j,j]*mx02[k]*mx03[j]*mx07[i])/4 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S3[c1,i,k,k]*mx02[k]*mx04[j]*mx06[i])/4 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,j,k]**2*mx02[k]*mx05[i]*mx05[j])/2 \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,i,k]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx07[i])/2 \
                            + (5*S3[c1,i,i,j]*S3[c1,i,j,j]**2*S3[c1,i,k,k]*mx02[k]*mx05[i]*mx05[j])/16 \
                            + 3*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]**2*mx03[k]*mx04[j]*mx05[i] \
                            + (11*S3[c1,i,i,k]*S3[c1,i,j,j]**2*S3[c1,i,j,k]*mx02[k]*mx05[i]*mx05[j])/16 \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx04[j]*mx06[i])/4 \
                            + (3*S3[c1,i,i,k]*S3[c1,i,j,j]**2*S3[c1,i,k,k]*mx03[k]*mx04[j]*mx05[i])/4 \
                            + (S3[c1,i,j,j]*S3[c1,i,j,k]**2*S3[c1,i,k,k]*mx04[i]*mx04[j]*mx04[k])/4 \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,k]**2*mx02[j]*mx02[k]*mx05[i])/2 \
                            + 3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,k]**2*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,k]**2*mx02[k]*mx03[j]*mx04[i] \
                            + S3[c1,i,i,i]*S3[c1,i,k,k]*S1[c1,j]**2*mx02[j]*mx02[k]*mx04[i] \
                            + (S2[c1,i,k]**2*S3[c1,i,i,i]*S2[c1,j,j]*mx02[j]*mx02[k]*mx05[i])/3 \
                            + (S1[c1,i]*S3[c1,i,i,i]*S3[c1,j,j,k]**2*mx02[k]*mx04[i]*mx04[j])/3 \
                            + S2[c1,i,i]*S2[c1,i,k]**2*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S3[c1,i,i,i]*S3[c1,i,j,k]**2*S1[c1,j]*mx02[k]*mx03[j]*mx05[i] \
                            + (S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]**2*mx03[i]*mx02[k]*mx03[j])/2 \
                            + (3*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]**2*mx03[i]*mx02[k]*mx03[j])/2 \
                            + 2*S1[c1,i]**2*S3[c1,i,i,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx04[i] \
                            + S1[c1,i]**2*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S2[c1,i,k]**2*S3[c1,i,i,j]*S2[c1,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + (S1[c1,i]*S3[c1,i,i,k]**2*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,k]**2*mx03[i]*mx03[j]*mx03[k] \
                            + 3*S2[c1,i,i]*S2[c1,i,j]**2*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + (3*S2[c1,i,i]*S3[c1,i,j,k]**2*S2[c1,j,j]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,k]**2*mx03[i]*mx03[j]*mx03[k] \
                            + (3*S2[c1,i,j]*S3[c1,i,i,k]**2*S2[c1,j,j]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + (8*S3[c1,i,i,j]*S3[c1,i,j,k]**2*S1[c1,j]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + (5*S1[c1,i]**2*S3[c1,i,j,j]*S3[c1,j,k,k]*mx03[i]*mx02[k]*mx03[j])/2 \
                            + (9*S1[c1,i]**2*S3[c1,i,j,k]*S3[c1,j,j,k]*mx03[i]*mx02[k]*mx03[j])/2 \
                            + (S1[c1,i]**2*S3[c1,i,k,k]*S3[c1,j,j,j]*mx03[i]*mx02[k]*mx03[j])/2 \
                            + (3*S2[c1,i,i]**2*S2[c1,i,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + (3*S2[c1,i,i]**2*S3[c1,i,j,k]*S2[c1,j,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + (3*S2[c1,i,i]**2*S3[c1,i,k,k]*S2[c1,j,j]*mx02[j]*mx02[k]*mx05[i])/4 \
                            + 6*S2[c1,i,j]**2*S3[c1,i,i,k]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + (3*S3[c1,i,i,k]**2*S3[c1,i,j,j]*S1[c1,j]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + (7*S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx04[k])/6 \
                            + (3*S1[c1,i]*S3[c1,i,i,j]**2*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx04[k])/8 \
                            + S2[c1,i,j]*S2[c1,i,k]**2*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]**2*S2[c1,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + S2[c1,i,k]**2*S3[c1,i,j,k]*S2[c1,j,j]*mx03[i]*mx03[j]*mx03[k] \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,k,k]*S1[c1,j]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + (S1[c1,i]*S3[c1,i,k,k]**2*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx04[k])/6 \
                            + (9*S2[c1,i,j]*S3[c1,i,k,k]**2*S2[c1,j,j]*mx03[i]*mx03[j]*mx04[k])/8 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]**2*S1[c1,j]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + 2*S2[c1,i,j]**2*S2[c1,i,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 2*S2[c1,i,j]**2*S3[c1,i,k,k]*S2[c1,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 4*S1[c1,i]*S3[c1,i,j,k]**2*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + 6*S2[c1,i,k]*S3[c1,i,j,k]**2*S2[c1,j,k]*mx03[i]*mx03[j]*mx04[k] \
                            + 2*S3[c1,i,j,k]**2*S3[c1,i,k,k]*S1[c1,j]*mx03[i]*mx03[j]*mx04[k] \
                            + S1[c1,i]*S3[c1,i,k,k]*S3[c1,j,k,k]**2*mx02[i]*mx02[j]*mx06[k] \
                            + 4*S1[c1,i]**2*S1[c1,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx02[k] \
                            + S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,k]**2*mx02[k]*mx03[j]*mx05[i] \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,j,j,k]**2*mx02[k]*mx04[j]*mx05[i])/4 \
                            + S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,k]**2*mx02[k]*mx04[i]*mx04[j] \
                            + (S2[c1,i,k]**2*S3[c1,i,i,i]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx05[i])/3 \
                            + (S2[c1,i,i]*S3[c1,i,i,k]**2*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx06[i])/4 \
                            + (S3[c1,i,i,i]*S3[c1,i,k,k]*S2[c1,j,j]**2*mx02[k]*mx04[i]*mx04[j])/4 \
                            + S3[c1,i,i,i]*S3[c1,i,j,k]**2*S2[c1,j,j]*mx02[k]*mx04[j]*mx05[i] \
                            + (3*S3[c1,i,i,j]*S3[c1,i,i,k]**2*S2[c1,j,j]*mx02[k]*mx03[j]*mx06[i])/4 \
                            + S2[c1,i,j]**2*S3[c1,i,i,i]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx05[i] \
                            + (S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,j,j,k]**2*mx04[i]*mx03[k]*mx04[j])/2 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]**2*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx06[i])/4 \
                            + (S2[c1,i,j]*S3[c1,i,i,k]**2*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + (S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,j,j,k]**2*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (3*S2[c1,i,i]**2*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx05[i])/4 \
                            + (3*S2[c1,i,i]**2*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + (S2[c1,i,i]**2*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx05[i])/4 \
                            + (3*S2[c1,i,j]**2*S3[c1,i,i,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (3*S2[c1,i,j]**2*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,i,k]*S2[c1,j,k]*mx02[k]*mx03[j]*mx06[i])/2 \
                            + (3*S3[c1,i,i,k]**2*S3[c1,i,j,j]*S2[c1,j,j]*mx02[k]*mx04[j]*mx05[i])/4 \
                            + 2*S2[c1,i,i]*S3[c1,i,j,k]**2*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j] \
                            + (3*S2[c1,i,j]*S3[c1,i,i,j]**2*S3[c1,j,k,k]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + (3*S2[c1,i,k]*S3[c1,i,i,j]**2*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + 3*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,k]**2*mx03[i]*mx03[j]*mx04[k] \
                            + (8*S3[c1,i,i,j]*S3[c1,i,j,k]**2*S2[c1,j,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + 6*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,k]**2*mx03[i]*mx03[j]*mx04[k] \
                            + S3[c1,i,i,k]*S3[c1,i,j,k]**2*S2[c1,j,j]*mx04[i]*mx03[k]*mx04[j] \
                            + 3*S3[c1,i,i,j]**2*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx04[j]*mx05[i])/4 \
                            + (5*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx05[k])/4 \
                            + S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx05[k] \
                            + (7*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (S2[c1,i,j]*S3[c1,i,k,k]**2*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx05[k])/2 \
                            + (S2[c1,i,k]*S3[c1,i,k,k]**2*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]**2*S2[c1,j,k]*mx03[i]*mx03[j]*mx05[k])/3 \
                            + (S3[c1,i,j,k]*S3[c1,i,k,k]**2*S2[c1,j,j]*mx03[i]*mx03[j]*mx05[k])/4 \
                            + (3*S2[c1,i,j]**2*S3[c1,i,k,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + 4*S2[c1,i,k]*S3[c1,i,j,k]**2*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx05[k] \
                            + 2*S3[c1,i,j,k]**2*S3[c1,i,k,k]*S2[c1,j,k]*mx03[i]*mx03[j]*mx05[k] \
                            + S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,k,k]**2*mx02[i]*mx02[j]*mx07[k] \
                            + 6*S2[c1,i,j]**2*S3[c1,i,i,k]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + (3*S1[c1,i]*S3[c1,i,i,j]**2*S2[c1,k,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]**2*S1[c1,k]*mx02[j]*mx02[k]*mx05[i] \
                            + (3*S2[c1,i,i]**2*S1[c1,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx04[i])/2 \
                            + 6*S2[c1,i,j]**2*S3[c1,i,j,k]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + (3*S3[c1,i,i,k]**2*S1[c1,j]*S2[c1,j,j]*mx02[k]*mx03[j]*mx04[i])/2 \
                            - (3*S2[c1,i,i]*S1[c1,j]*S3[c1,j,k,k]**2*mx02[i]*mx03[j]*mx04[k])/2 \
                            + (7*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,k]**2*mx02[k]*mx05[i]*mx05[j])/48 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]**2*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx07[i])/12 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,k]**2*mx03[k]*mx04[j]*mx05[i])/4 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]**2*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx07[i])/4 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,k]**2*S3[c1,j,j,j]*mx02[k]*mx05[i]*mx05[j])/6 \
                            + (S3[c1,i,i,j]*S3[c1,i,i,k]**2*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx06[i])/4 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]**2*mx04[i]*mx04[j]*mx04[k])/12 \
                            + (7*S3[c1,i,i,i]*S3[c1,i,j,j]**2*S3[c1,j,k,k]*mx02[k]*mx05[i]*mx05[j])/48 \
                            + S3[c1,i,i,i]*S3[c1,i,j,k]**2*S3[c1,j,j,k]*mx03[k]*mx04[j]*mx05[i] \
                            + (S3[c1,i,i,i]**2*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx07[i])/12 \
                            + (S3[c1,i,i,i]**2*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx07[i])/6 \
                            + (S3[c1,i,i,i]**2*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx07[i])/36 \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx06[i])/4 \
                            + (5*S3[c1,i,i,k]**2*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[k]*mx05[i]*mx05[j])/48 \
                            + (S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,j,j,k]**2*mx04[i]*mx04[j]*mx04[k])/24 \
                            + (7*S3[c1,i,i,j]**2*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx05[i]*mx05[j])/16 \
                            + (13*S3[c1,i,i,j]**2*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx05[i]*mx05[j])/16 \
                            + (5*S3[c1,i,i,j]**2*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx05[i]*mx05[j])/48 \
                            + (3*S3[c1,i,i,k]**2*S3[c1,i,j,j]*S3[c1,j,j,k]*mx03[k]*mx04[j]*mx05[i])/4 \
                            + (S3[c1,i,i,k]**2*S3[c1,i,j,k]*S3[c1,j,j,j]*mx03[k]*mx04[j]*mx05[i])/2 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,j,k]**2*S3[c1,j,k,k]*mx04[i]*mx04[j]*mx04[k])/4 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,k]**2*S3[c1,j,j,k]*mx04[i]*mx04[j]*mx04[k])/2 \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,j,k]*S3[c1,j,k,k]*mx03[k]*mx04[j]*mx05[i])/2 \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,k,k]*S3[c1,j,j,k]*mx03[k]*mx04[j]*mx05[i])/4 \
                            + (7*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx06[k])/16 \
                            + (15*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx06[k])/16 \
                            + (5*S3[c1,i,j,j]*S3[c1,i,k,k]**2*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx06[k])/16 \
                            + (9*S3[c1,i,j,k]*S3[c1,i,k,k]**2*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx06[k])/16 \
                            + (3*S3[c1,i,j,k]**2*S3[c1,i,k,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx06[k])/2 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,k]**2*mx03[i]*mx02[k]*mx03[j])/2 \
                            + (S3[c1,i,i,i]*S1[c1,j]**2*S3[c1,j,k,k]*mx03[i]*mx02[k]*mx03[j])/2 \
                            + (2*S2[c1,i,j]**2*S3[c1,i,i,i]*S2[c1,k,k]*mx02[j]*mx02[k]*mx05[i])/3 \
                            + 3*S2[c1,i,j]**2*S3[c1,i,i,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + (3*S2[c1,i,j]*S3[c1,i,i,j]**2*S2[c1,k,k]*mx02[k]*mx03[j]*mx05[i])/2 \
                            - (S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx04[k])/6 \
                            + (2*S3[c1,i,i,k]*S3[c1,i,j,j]**2*S1[c1,k]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + (3*S2[c1,i,i]**2*S2[c1,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx04[i])/4 \
                            + (3*S2[c1,i,i]**2*S2[c1,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx04[i])/2 \
                            + 3*S3[c1,i,i,j]**2*S3[c1,i,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx05[i] \
                            + (S3[c1,i,i,k]**2*S1[c1,j]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx04[j])/6 \
                            - 3*S2[c1,i,i]*S2[c1,j,k]**2*S3[c1,j,k,k]*mx02[i]*mx03[j]*mx04[k] \
                            + S2[c1,i,j]**2*S3[c1,i,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (5*S3[c1,i,i,j]**2*S1[c1,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx04[j])/6 \
                            + (S1[c1,i]*S3[c1,i,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx06[k])/9 \
                            - (3*S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,k,k]**2*mx02[i]*mx03[j]*mx05[k])/2 \
                            + 3*S2[c1,i,j]*S3[c1,i,j,k]**2*S2[c1,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (S3[c1,i,k,k]**2*S1[c1,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S2[c1,k,k]*mx02[k]*mx04[j]*mx05[i])/4 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]**2*mx03[i]*mx03[j]*mx04[k])/8 \
                            + (S3[c1,i,i,i]**2*S2[c1,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx06[i])/12 \
                            + (S3[c1,i,i,i]**2*S2[c1,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx06[i])/6 \
                            + (3*S3[c1,i,i,j]**2*S3[c1,i,j,j]*S2[c1,k,k]*mx02[k]*mx04[j]*mx05[i])/4 \
                            + (S2[c1,i,i]*S3[c1,i,j,j]**2*S3[c1,k,k,k]*mx04[i]*mx03[k]*mx04[j])/6 \
                            - (S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,j]**2*S2[c1,k,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (S3[c1,i,i,k]**2*S2[c1,j,j]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j])/4 \
                            + (S3[c1,i,i,k]**2*S2[c1,j,k]*S3[c1,j,j,j]*mx04[i]*mx03[k]*mx04[j])/6 \
                            - (3*S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]**2*mx02[i]*mx04[j]*mx05[k])/4 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx06[k])/4 \
                            + S2[c1,i,j]**2*S3[c1,i,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (5*S3[c1,i,i,j]**2*S2[c1,j,k]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/6 \
                            + (S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx07[k])/6 \
                            + S2[c1,i,j]*S3[c1,i,j,k]**2*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k] \
                            + (S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx07[k])/9 \
                            + (S3[c1,i,k,k]**2*S2[c1,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            + S3[c1,i,i,i]*S2[c1,j,j]*S1[c1,k]**2*mx02[j]*mx03[i]*mx02[k] \
                            + S3[c1,i,i,i]*S1[c1,j]**2*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k] \
                            - 3*S2[c1,i,i]*S3[c1,j,j,k]*S1[c1,k]**2*mx02[i]*mx02[j]*mx03[k] \
                            - 2*S3[c1,i,i,i]*S2[c1,j,k]**2*S1[c1,k]*mx02[j]*mx03[i]*mx03[k] \
                            - 2*S2[c1,i,i]*S1[c1,j]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k] \
                            + (3*S3[c1,i,i,j]**2*S1[c1,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i])/2 \
                            + 3*S3[c1,i,i,j]**2*S2[c1,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            - (3*S2[c1,i,i]*S3[c1,j,k,k]**2*S1[c1,k]*mx02[i]*mx02[j]*mx05[k])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]**2*S3[c1,k,k,k]*mx03[k]*mx04[j]*mx05[i])/12 \
                            + (S3[c1,i,i,i]**2*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[j]*mx06[i])/36 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]**2*mx03[i]*mx04[j]*mx05[k])/4 \
                            + (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,k,k,k]**2*mx03[i]*mx03[j]*mx06[k])/24 \
                            + (S3[c1,i,i,j]**2*S3[c1,i,j,j]*S3[c1,k,k,k]*mx03[k]*mx04[j]*mx05[i])/4 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,j]**2*S3[c1,k,k,k]*mx04[i]*mx04[j]*mx04[k])/36 \
                            + (S3[c1,i,i,k]**2*S3[c1,j,j,j]*S3[c1,j,k,k]*mx04[i]*mx04[j]*mx04[k])/24 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx08[k])/36 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,j]*S1[c1,k]**2*mx03[i]*mx02[k]*mx03[j])/3 \
                            - S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]**2*mx02[j]*mx03[i]*mx03[k] \
                            - (S3[c1,i,i,i]*S1[c1,j]**2*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[k])/3 \
                            - (S2[c1,i,i]*S3[c1,j,j,j]*S2[c1,k,k]**2*mx02[i]*mx03[j]*mx04[k])/4 \
                            - S3[c1,i,i,i]*S2[c1,j,k]**2*S2[c1,k,k]*mx02[j]*mx03[i]*mx04[k] \
                            + (S2[c1,i,i]**2*S3[c1,j,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i])/4 \
                            - (3*S2[c1,i,i]*S3[c1,j,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx05[k])/4 \
                            - (S3[c1,i,i,i]*S3[c1,j,k,k]**2*S1[c1,k]*mx02[j]*mx03[i]*mx05[k])/2 \
                            + (5*S3[c1,i,i,j]**2*S3[c1,j,j,k]*S1[c1,k]*mx02[k]*mx04[i]*mx04[j])/6 \
                            - S2[c1,i,i]*S2[c1,j,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx05[k] \
                            + (S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx06[k])/18 \
                            - (3*S2[c1,i,i]*S3[c1,j,k,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx06[k])/4 \
                            + S3[c1,i,i,k]*S3[c1,j,k,k]**2*S1[c1,k]*mx02[i]*mx02[j]*mx06[k] \
                            + (3*S3[c1,i,k,k]**2*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx06[k])/4 \
                            + (S3[c1,i,k,k]**2*S3[c1,j,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S2[c1,k,k]**2*mx02[j]*mx03[i]*mx05[k])/4 \
                            + (S3[c1,i,i,i]**2*S3[c1,j,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx06[i])/36 \
                            - (S3[c1,i,i,i]*S2[c1,j,k]**2*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx05[k])/3 \
                            - (S2[c1,i,i]*S3[c1,j,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx03[j]*mx06[k])/36 \
                            - (S3[c1,i,i,i]*S3[c1,j,k,k]**2*S2[c1,k,k]*mx02[j]*mx03[i]*mx06[k])/4 \
                            + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx06[k])/8 \
                            - (S3[c1,i,i,j]**2*S2[c1,j,j]*S3[c1,k,k,k]*mx04[i]*mx03[k]*mx04[j])/6 \
                            + (5*S3[c1,i,i,j]**2*S3[c1,j,j,k]*S2[c1,k,k]*mx04[i]*mx03[k]*mx04[j])/12 \
                            - (S2[c1,i,i]*S3[c1,j,j,k]**2*S3[c1,k,k,k]*mx02[i]*mx04[j]*mx05[k])/4 \
                            + (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx07[k])/18 \
                            + S3[c1,i,i,k]*S2[c1,j,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k] \
                            + (S3[c1,i,i,k]*S3[c1,j,k,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            + (S3[c1,i,k,k]**2*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx07[k])/4 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,k,k,k]**2*mx02[j]*mx03[i]*mx07[k])/36 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]**2*S3[c1,k,k,k]*mx03[i]*mx04[j]*mx05[k])/12 \
                            - (S3[c1,i,i,i]*S3[c1,j,k,k]**2*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx07[k])/12 \
                            + (S3[c1,i,i,j]**2*S3[c1,j,j,k]*S3[c1,k,k,k]*mx04[i]*mx04[j]*mx04[k])/18 \
                            + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx08[k])/18 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx08[k])/24 \
                            + (S3[c1,i,i,k]*S3[c1,j,k,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx08[k])/6 \
                            + (S3[c1,i,k,k]**2*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx08[k])/12 \
                            + S3[c1,i,j,k]**2*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k] \
                            + (S3[c1,i,j,k]**2*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            - S3[c1,i,j,k]**3*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k] \
                            - 6*S1[c1,i]*S3[c1,i,j,k]**2*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k] \
                            - (3*S2[c1,i,i]*S3[c1,i,k,k]*S2[c1,j,j]**2*mx02[j]**2*mx03[i]*mx02[k])/2 \
                            - 3*S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,j]**2*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S2[c1,i,k]**2*S3[c1,i,j,j]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S2[c1,i,j]**2*S3[c1,i,k,k]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k] \
                            + (3*S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,j]**2*mx02[i]*mx02[j]**2*mx03[k])/2 \
                            + (S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]**2*mx02[i]*mx03[j]**2*mx03[k])/6 \
                            - (3*S3[c1,i,j,j]*S3[c1,i,k,k]**2*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx04[k])/4 \
                            - 3*S3[c1,i,j,k]**2*S3[c1,i,k,k]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx04[k] \
                            - (S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]**2*mx02[k]*mx03[j]**2*mx04[i])/18 \
                            - S3[c1,i,i,j]*S3[c1,i,j,k]**2*S3[c1,j,j,j]*mx02[k]*mx03[j]**2*mx04[i] \
                            - S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,j,k]**2*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S2[c1,i,j]*S3[c1,i,j,k]**2*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j] \
                            - (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]**2*mx03[i]*mx02[k]**2*mx03[j])/8 \
                            - (S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx03[j]*mx03[k]**2)/3 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]**3)/4 \
                            - (3*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,k,k]**2*mx02[i]*mx02[j]**2*mx05[k])/4 \
                            - (3*S3[c1,i,k,k]**2*S2[c1,j,j]*S3[c1,j,j,k]*mx02[i]*mx02[j]**2*mx05[k])/4 \
                            - (3*S3[c1,i,i,j]*S1[c1,j]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]**2)/2 \
                            - (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,k,k,k]**2*mx03[i]*mx03[j]*mx03[k]**2)/24 \
                            - (S3[c1,i,i,j]**2*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]**2*mx04[i])/4 \
                            + (S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,k]**2*mx02[j]*mx03[i]*mx02[k]**2)/4 \
                            - (S2[c1,i,i]*S3[c1,j,j,j]**2*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]**2)/12 \
                            + S3[c1,i,i,i]*S2[c1,j,k]**2*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]**2 \
                            + (S2[c1,i,i]*S2[c1,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]**2)/12 \
                            - (S2[c1,i,i]*S2[c1,j,j]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx03[k])/4 \
                            - (S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]**2)/6 \
                            + (3*S3[c1,i,i,k]*S2[c1,j,j]**2*S2[c1,k,k]*mx02[i]*mx02[j]**2*mx03[k])/4 \
                            - 3*S3[c1,i,j,k]**2*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - (S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,k]**2*mx03[i]*mx02[k]**2*mx03[j])/12 \
                            + (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,k,k,k]**2*mx02[j]*mx03[i]*mx03[k]**2)/12 \
                            + (S3[c1,i,i,i]*S2[c1,j,j]**2*S3[c1,k,k,k]*mx02[j]**2*mx03[i]*mx03[k])/12 \
                            - (S2[c1,i,i]*S3[c1,j,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx03[j]*mx03[k]**2)/18 \
                            - (S2[c1,i,i]*S3[c1,j,j,j]**2*S3[c1,k,k,k]*mx02[i]*mx03[j]**2*mx03[k])/36 \
                            - (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx03[j]*mx03[k]**2)/6 \
                            + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]**3)/2 \
                            + (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]**3)/8 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,j]**2*S2[c1,k,k]*mx02[i]*mx03[j]**2*mx03[k])/12 \
                            - (S3[c1,i,i,k]*S2[c1,j,j]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx04[k])/4 \
                            - S3[c1,i,i,k]*S2[c1,j,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2 \
                            - S3[c1,i,j,k]**2*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx05[k] \
                            - 2*S3[c1,i,j,k]**2*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,k,k]**2*mx03[i]*mx03[j]*mx03[k]**2)/108 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,j]**2*S3[c1,k,k,k]*mx03[i]*mx03[j]**2*mx03[k])/108 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,j]**2*S3[c1,k,k,k]*mx02[i]*mx03[j]**2*mx04[k])/36 \
                            - S3[c1,i,j,k]**2*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2 \
                            - (3*S3[c1,i,j,k]**2*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx04[k])/2 \
                            - (S3[c1,i,j,k]**2*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]*mx05[k])/6 \
                            + (3*S2[c1,i,i]*S3[c1,i,j,k]**2*S2[c1,j,j]*mx02[i]**2*mx02[j]**2*mx02[k])/2 \
                            + (S3[c1,i,j,k]**2*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx03[k]**2)/12 \
                            - 3*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 12*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S1[c1,i]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S2[c1,i,i]*S2[c1,i,j]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S2[c1,i,i]*S2[c1,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 6*S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*S2[c1,j,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 6*S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,j,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 6*S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,j]*S2[c1,j,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 6*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k] \
                            - (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]**2*mx04[i])/2 \
                            - S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]**2*mx04[i] \
                            - (3*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j])/2 \
                            - 3*S2[c1,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - 6*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - 3*S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - 6*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,j]*S2[c1,j,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - 6*S2[c1,i,j]*S3[c1,i,k,k]*S1[c1,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            - 12*S2[c1,i,k]*S3[c1,i,j,k]*S1[c1,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            - (3*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]**2*mx03[i]*mx04[k])/2 \
                            - 3*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]**2*mx03[i]*mx04[k] \
                            - (3*S3[c1,i,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[j]**2*mx03[i]*mx04[k])/2 \
                            - 3*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]**2*mx05[k] \
                            - S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 6*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,j]*S1[c1,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - 3*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,j]*S1[c1,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            - (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]**2*mx04[i])/6 \
                            - (S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[k]*mx03[j]**2*mx04[i])/3 \
                            - (S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[k]*mx03[j]**2*mx04[i])/2 \
                            - (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k]**2)/18 \
                            + (S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[j]**2*mx03[i]*mx03[k])/2 \
                            + S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[j]**2*mx03[i]*mx03[k] \
                            - (3*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j])/2 \
                            - (3*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j])/2 \
                            - 6*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j] \
                            - (3*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j])/2 \
                            - 3*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j] \
                            - 3*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j] \
                            - 2*S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - S2[c1,i,k]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - 3*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - (3*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]**2*mx04[k])/2 \
                            - (S2[c1,i,j]*S2[c1,i,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/2 \
                            - (3*S2[c1,i,j]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/2 \
                            - 2*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2 \
                            - (S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/2 \
                            - 2*S3[c1,i,j,k]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2 \
                            - 6*S2[c1,i,j]*S3[c1,i,j,k]*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            - 6*S3[c1,i,i,k]*S1[c1,j]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            - S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[j]**2*mx03[i]*mx04[k] \
                            - (S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[j]**2*mx03[i]*mx04[k])/2 \
                            - 2*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            - S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            - S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            - 2*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            - (S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2)/3 \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx05[k])/2 \
                            - S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            - S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            - S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,k]*S1[c1,k]*mx02[j]**2*mx03[i]*mx02[k] \
                            + S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]**2 \
                            - (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k]**2)/18 \
                            - (2*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k]**2)/9 \
                            - (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k]**2)/18 \
                            - (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k]**2)/2 \
                            - (S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k]**2)/18 \
                            - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j])/2 \
                            - S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx02[k]**2*mx03[j] \
                            - S3[c1,i,i,j]*S2[c1,j,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx04[k] \
                            - (3*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]**2*mx04[k])/2 \
                            - (3*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]**2*mx04[k])/2 \
                            + (S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/2 \
                            - S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2 \
                            - S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2 \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/3 \
                            + 3*S2[c1,i,i]*S3[c1,j,j,k]*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            - 6*S3[c1,i,i,j]*S2[c1,j,k]*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            - 3*S3[c1,i,i,k]*S2[c1,j,j]*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]**2 \
                            + (S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[k]**2)/3 \
                            - (S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx05[k])/2 \
                            - S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            - (S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]**2*mx05[k])/2 \
                            - (S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2)/2 \
                            - S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            - S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2 \
                            + S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]**2 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k]**2)/18 \
                            + (S2[c1,i,i]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/2 \
                            - S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2 \
                            - (2*S3[c1,i,i,j]*S3[c1,j,k,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/3 \
                            - (S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/2 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]**2)/2 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[k]**2)/6 \
                            - (S3[c1,i,i,j]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2)/2 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]**2)/6 \
                            - 2*S1[c1,i]*S3[c1,i,j,k]**2*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - 3*S3[c1,i,i,j]*S3[c1,i,j,k]**2*S2[c1,j,j]*mx02[j]*mx02[k]*mx03[j]*mx04[i] \
                            - S2[c1,i,k]**2*S3[c1,i,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - S2[c1,i,j]**2*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]**2*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j]*mx04[k])/4 \
                            - S3[c1,i,j,k]**2*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx03[j]*mx04[k] \
                            - 3*S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - (S3[c1,i,i,i]*S2[c1,j,k]**2*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j])/3 \
                            - (S3[c1,i,i,k]**2*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx02[k]*mx03[j]*mx04[i])/4 \
                            - (3*S3[c1,i,i,j]**2*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx03[j]*mx04[i])/4 \
                            - 3*S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx04[k])/2 \
                            - (S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]*mx04[k])/3 \
                            - S2[c1,i,j]*S3[c1,i,j,k]**2*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k] \
                            - (S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,k,k]**2*mx02[i]*mx02[j]*mx03[j]*mx05[k])/4 \
                            - (S3[c1,i,k,k]**2*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx03[j]*mx05[k])/4 \
                            - (S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx03[j]*mx03[k]*mx04[k])/3 \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]*mx05[k])/18 \
                            + (3*S2[c1,i,i]*S3[c1,j,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            - 3*S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - (3*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            + (S2[c1,i,i]*S2[c1,j,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/3 \
                            - 3*S3[c1,i,i,k]*S2[c1,j,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            + (3*S2[c1,i,i]*S3[c1,j,k,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k])/4 \
                            - (S2[c1,i,k]**2*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/3 \
                            - 6*S3[c1,i,j,k]**2*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - (3*S3[c1,i,k,k]**2*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k])/4 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,k]*S2[c1,k,k]**2*mx02[j]*mx03[i]*mx02[k]*mx03[k])/2 \
                            + (S3[c1,i,i,i]*S2[c1,j,k]**2*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[k])/3 \
                            - (3*S3[c1,i,i,j]*S3[c1,j,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,j]*S2[c1,k,k]**2*mx02[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            + (S3[c1,i,i,i]*S3[c1,j,k,k]**2*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]*mx04[k])/4 \
                            - S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx04[k] \
                            - (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx04[k])/4 \
                            - S3[c1,i,j,k]**2*S3[c1,j,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k] \
                            + (S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]*mx04[k])/6 \
                            - (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]*mx04[k])/3 \
                            - (3*S3[c1,i,i,j]*S3[c1,j,k,k]**2*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k])/4 \
                            - (S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]*mx04[k])/6 \
                            + (S2[c1,i,i]*S3[c1,j,k,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k])/4 \
                            - (S3[c1,i,i,k]*S3[c1,j,k,k]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[k])/2 \
                            - (S3[c1,i,k,k]**2*S3[c1,j,j,j]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k])/4 \
                            - 3*S3[c1,i,j,k]**2*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k] \
                            - (S3[c1,i,k,k]**2*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k])/4 \
                            - (S3[c1,i,k,k]**2*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[k])/4 \
                            - 3*S3[c1,i,j,k]**2*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            + (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,k,k,k]**2*mx02[j]*mx03[i]*mx03[k]*mx04[k])/18 \
                            + (S3[c1,i,i,i]*S3[c1,j,k,k]**2*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[k]*mx04[k])/12 \
                            - (S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx03[j]*mx03[k]*mx04[k])/6 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,k,k]**2*mx02[i]*mx03[j]*mx03[k]*mx04[k])/18 \
                            - (S3[c1,i,j,k]**2*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[j]*mx05[k])/3 \
                            - (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]*mx05[k])/9 \
                            - (S3[c1,i,i,j]*S3[c1,j,k,k]**2*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]*mx04[k])/4 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx03[k]*mx05[k])/12 \
                            - (S3[c1,i,i,k]*S3[c1,j,k,k]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx05[k])/6 \
                            - (S3[c1,i,k,k]**2*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]*mx04[k])/12 \
                            - S3[c1,i,j,k]**2*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]*mx04[k] \
                            - (S3[c1,i,k,k]**2*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx05[k])/12 \
                            - (S2[c1,i,j]**2*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            - (S3[c1,i,j,k]**2*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[k])/2 \
                            - (S3[c1,i,j,k]**2*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k])/2 \
                            + (3*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[i]**2*mx02[j]**2*mx02[k])/4 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[i]**2*mx02[j]**2*mx02[k])/4 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[i]**2*mx02[j]**2*mx02[k])/4 \
                            + S3[c1,i,i,i]*S3[c1,i,j,k]**2*S2[c1,j,j]*mx02[i]*mx02[j]**2*mx03[i]*mx02[k] \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx03[k]**2)/36 \
                            + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx03[k]**2)/18 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,k]**2*mx02[i]*mx02[j]*mx02[k]*mx03[k]**2)/24 \
                            + (S3[c1,i,j,k]**2*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]**2*mx03[k])/2 \
                            + 12*S1[c1,i]*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,k]*mx02[j]*mx02[k]*mx04[i] \
                            + (3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + 3*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,k,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 3*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 9*S1[c1,i]*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 3*S1[c1,i]*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,i,j,j]*mx02[j]*mx02[k]*mx05[i] \
                            + 6*S2[c1,i,i]*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 6*S1[c1,i]*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 12*S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 12*S1[c1,i]*S2[c1,i,j]*S3[c1,i,k,k]*S1[c1,j]*mx02[j]*mx03[i]*mx02[k] \
                            + 24*S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,k]*S1[c1,j]*mx02[j]*mx03[i]*mx02[k] \
                            + 3*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 6*S1[c1,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 6*S2[c1,i,i]*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 12*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 6*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,i,j,j]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx04[i]*mx04[j] \
                            + 6*S1[c1,i]*S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 6*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[j]*mx02[k]*mx04[i] \
                            + 9*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S1[c1,i]*S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S1[c1,i]*S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,j]*mx02[j]*mx02[k]*mx04[i] \
                            + 6*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]*mx02[j]*mx02[k]*mx04[i] \
                            + 12*S1[c1,i]*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*mx02[j]*mx02[k]*mx04[i] \
                            + 6*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,k,k]*S1[c1,j]*mx02[j]*mx02[k]*mx04[i] \
                            + 6*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,k]*S1[c1,j]*mx02[j]*mx02[k]*mx04[i] \
                            + 12*S1[c1,i]*S2[c1,i,j]*S2[c1,i,k]*S3[c1,j,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 12*S1[c1,i]*S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 8*S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 6*S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*mx03[i]*mx02[k]*mx03[j] \
                            + 2*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,j]*mx03[i]*mx02[k]*mx03[j] \
                            + 12*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,k]*S1[c1,j]*mx03[i]*mx02[k]*mx03[j] \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx06[i])/2 \
                            + 3*S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx06[i] \
                            + 6*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*mx02[k]*mx03[j]*mx06[i] \
                            + 3*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx04[j]*mx05[i] \
                            + 6*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + 6*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + (7*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + 2*S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx05[i] \
                            + S1[c1,i]*S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[j]*mx02[k]*mx05[i] \
                            + (2*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx05[i])/3 \
                            + (4*S1[c1,i]*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,j,j,k]*mx02[j]*mx02[k]*mx05[i])/3 \
                            + 2*S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 3*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,i,k]*S2[c1,j,k]*mx02[j]*mx02[k]*mx05[i] \
                            + (9*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + (3*S2[c1,i,i]*S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + (3*S2[c1,i,i]*S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,j]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + 2*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,i]*S2[c1,j,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,j,k]*S1[c1,j]*mx02[j]*mx02[k]*mx05[i] \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,i,k]*S1[c1,j]*mx02[j]*mx02[k]*mx05[i] \
                            + 3*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S1[c1,i]*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + S1[c1,i]*S2[c1,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S1[c1,i]*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S1[c1,i]*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 12*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S1[c1,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S1[c1,i]*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,i]*S2[c1,i,j]*S2[c1,i,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 12*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,i]*S2[c1,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,i]*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 12*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 12*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 12*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,k]*S1[c1,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,i,j,j]*S1[c1,j]*mx02[k]*mx03[j]*mx04[i] \
                            + S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 2*S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (S1[c1,i]*S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx03[k])/3 \
                            + S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 4*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 2*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*mx03[i]*mx03[j]*mx03[k] \
                            + 2*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,j]*mx03[i]*mx03[j]*mx03[k] \
                            + 6*S1[c1,i]*S2[c1,i,i]*S1[c1,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k] \
                            + 12*S1[c1,i]*S3[c1,i,i,k]*S1[c1,j]*S2[c1,j,k]*mx02[j]*mx03[i]*mx02[k] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*mx02[k]*mx03[j]*mx07[i])/2 \
                            + S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx03[j]*mx07[i] \
                            + 3*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*mx02[k]*mx04[j]*mx06[i] \
                            + 3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*mx03[k]*mx04[j]*mx05[i] \
                            + S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 2*S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + (S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx05[i])/3 \
                            + 3*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,i]*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + S2[c1,i,i]*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx05[i] \
                            + 6*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + 3*S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx05[i] \
                            + 2*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 4*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx05[i] \
                            + 6*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,i,k]*S2[c1,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,i,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx05[i] \
                            + S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,j]*mx02[k]*mx03[j]*mx05[i] \
                            + 6*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*mx02[k]*mx03[j]*mx05[i] \
                            + 2*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx04[j] \
                            + (11*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + (4*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx04[j])/9 \
                            + (5*S1[c1,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + S1[c1,i]*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx04[j] \
                            + (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx04[j])/4 \
                            + (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx04[j])/4 \
                            + 3*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx04[i]*mx04[j] \
                            + (3*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx04[i]*mx04[j])/4 \
                            + (3*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + 9*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx04[i]*mx04[j] \
                            + (9*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx04[i]*mx04[j])/4 \
                            + (9*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (9*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,j,k]*mx02[k]*mx04[i]*mx04[j] \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[k]*mx04[i]*mx04[j] \
                            + (3*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,j]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,j]*mx02[k]*mx04[i]*mx04[j] \
                            + (7*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S1[c1,j]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + (5*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + 3*S1[c1,i]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx04[k] \
                            + 3*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (3*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + 9*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,k]*mx03[i]*mx03[j]*mx04[k] \
                            + 3*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,k]*mx03[i]*mx03[j]*mx04[k] \
                            + 3*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,j]*mx03[i]*mx03[j]*mx04[k] \
                            + 3*S1[c1,i]*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,k,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 2*S1[c1,i]*S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 12*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,k]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S2[c1,i,i]*S3[c1,i,i,k]*S1[c1,j]*S2[c1,j,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 6*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,i,j]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + S2[c1,i,k]*S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S1[c1,i]*S2[c1,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 6*S1[c1,i]*S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 4*S1[c1,i]*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 4*S1[c1,i]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 3*S1[c1,i]*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 7*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 6*S2[c1,i,i]*S2[c1,i,j]*S1[c1,j]*S3[c1,j,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 6*S2[c1,i,i]*S2[c1,i,k]*S1[c1,j]*S3[c1,j,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 6*S2[c1,i,i]*S3[c1,i,j,k]*S1[c1,j]*S2[c1,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 8*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,j]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 4*S2[c1,i,k]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,j,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 4*S1[c1,i]*S3[c1,i,j,k]*S1[c1,j]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k] \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx06[i])/2 \
                            + S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx06[i])/6 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx06[i])/2 \
                            + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx06[i] \
                            + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + (S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*mx02[k]*mx03[j]*mx06[i])/3 \
                            + 2*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx06[i])/2 \
                            + S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx06[i] \
                            + S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[k]*mx03[j]*mx06[i] \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + 3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + (S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx05[i] \
                            + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[j]*mx05[i] \
                            + 2*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + (S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx05[i])/3 \
                            + 3*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + (2*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx05[i])/3 \
                            + S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx05[i] \
                            + 2*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + 3*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + 3*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[k]*mx04[j]*mx05[i] \
                            + (13*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/6 \
                            + (5*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j])/6 \
                            + (S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx04[i]*mx03[k]*mx04[j])/2 \
                            + (11*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (4*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (5*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + 3*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j] \
                            + (S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + 2*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j] \
                            + (11*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (4*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx04[i]*mx03[k]*mx04[j])/9 \
                            + (5*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*mx04[i]*mx03[k]*mx04[j] \
                            + S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,k]*mx04[i]*mx03[k]*mx04[j] \
                            + (5*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,j]*mx04[i]*mx03[k]*mx04[j])/6 \
                            + (7*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,j,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + 3*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx05[k] \
                            + (5*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx05[k])/3 \
                            + 3*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx05[k] \
                            + S1[c1,i]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,k,k]*mx02[j]*mx02[k]*mx05[i] \
                            + (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,k,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx05[i])/3 \
                            + (9*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,k]*S1[c1,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S1[c1,k]*mx02[j]*mx02[k]*mx05[i])/2 \
                            + 2*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,k]*S1[c1,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 3*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,i,k]*S1[c1,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,k]*mx02[j]*mx02[k]*mx05[i] \
                            + S1[c1,i]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S1[c1,i]*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S1[c1,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S2[c1,i,i]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S2[c1,i,i]*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S2[c1,i,i]*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S2[c1,i,j]*S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 12*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,i,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 4*S3[c1,i,i,i]*S3[c1,i,j,k]*S1[c1,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + S3[c1,i,i,i]*S3[c1,i,k,k]*S1[c1,j]*S2[c1,j,j]*mx02[k]*mx03[j]*mx04[i] \
                            + 6*S3[c1,i,i,j]*S3[c1,i,i,k]*S1[c1,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx04[i] \
                            + S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + S1[c1,i]*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 2*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 2*S2[c1,i,i]*S2[c1,i,j]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + S2[c1,i,i]*S2[c1,i,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + S2[c1,i,i]*S2[c1,i,k]*S2[c1,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + S2[c1,i,i]*S3[c1,i,k,k]*S2[c1,j,j]*S2[c1,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 4*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (3*S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]*S2[c1,j,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + 3*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*S2[c1,j,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 6*S1[c1,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k] \
                            + 6*S2[c1,i,i]*S3[c1,i,j,k]*S1[c1,j]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k] \
                            + 6*S2[c1,i,j]*S3[c1,i,i,k]*S1[c1,j]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k] \
                            + 6*S2[c1,i,k]*S3[c1,i,i,j]*S1[c1,j]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k] \
                            - 2*S1[c1,i]*S2[c1,i,j]*S1[c1,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx07[i])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[j]*mx06[i])/2 \
                            + S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx06[i] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx06[i])/6 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx06[i])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx06[i])/3 \
                            + (13*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx02[k]*mx05[i]*mx05[j])/24 \
                            + (5*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx05[i]*mx05[j])/72 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,j,k]*mx02[k]*mx05[i]*mx05[j])/4 \
                            + (11*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*mx02[k]*mx05[i]*mx05[j])/24 \
                            + S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*mx03[k]*mx04[j]*mx05[i] \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx03[k]*mx04[j]*mx05[i])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx03[k]*mx04[j]*mx05[i])/3 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*mx03[k]*mx04[j]*mx05[i])/2 \
                            + 3*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*mx03[k]*mx04[j]*mx05[i] \
                            + (S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx03[k]*mx04[j]*mx05[i])/2 \
                            + (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*mx04[i]*mx04[j]*mx04[k])/4 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx04[i]*mx04[j]*mx04[k])/8 \
                            + (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*mx04[i]*mx04[j]*mx04[k])/8 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx04[i]*mx04[j]*mx04[k])/8 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx04[i]*mx04[j]*mx04[k])/24 \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx05[i] \
                            + S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,j,k,k]*mx02[k]*mx03[j]*mx05[i] \
                            + S3[c1,i,i,i]*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + S3[c1,i,i,i]*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,j,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 2*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx05[i] \
                            + 3*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,i,j,j]*S1[c1,k]*mx02[k]*mx03[j]*mx05[i] \
                            + (2*S1[c1,i]*S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx04[j])/9 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx04[j])/4 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/4 \
                            + (S2[c1,i,i]*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (3*S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx04[j])/4 \
                            + (3*S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (S2[c1,i,k]*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/2 \
                            + (S2[c1,i,k]*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + (5*S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx04[j])/9 \
                            + S3[c1,i,i,i]*S3[c1,i,j,k]*S1[c1,j]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j] \
                            + 2*S3[c1,i,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*S2[c1,j,k]*mx02[k]*mx04[i]*mx04[j] \
                            + (S3[c1,i,i,i]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx04[j])/9 \
                            + (4*S3[c1,i,i,j]*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + 3*S3[c1,i,i,j]*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,j,k]*mx02[k]*mx04[i]*mx04[j] \
                            + 3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,j,k]*S1[c1,k]*mx02[k]*mx04[i]*mx04[j] \
                            + 2*S1[c1,i]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (7*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/6 \
                            + 3*S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (3*S2[c1,i,i]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k])/4 \
                            + (3*S2[c1,i,i]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (9*S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (3*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + 3*S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (3*S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + 3*S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (4*S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + 3*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + S3[c1,i,i,k]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx04[k] \
                            + 3*S3[c1,i,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*S2[c1,j,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S1[c1,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + 3*S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx06[k] \
                            + (S1[c1,i]*S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,k]*mx02[j]*mx02[k]*mx04[i])/2 \
                            + 3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,k]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S2[c1,i,i]*S3[c1,i,i,k]*S2[c1,j,j]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + S2[c1,i,j]*S3[c1,i,i,i]*S1[c1,j]*S2[c1,k,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,k]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + S2[c1,i,k]*S3[c1,i,i,i]*S2[c1,j,j]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 2*S3[c1,i,i,i]*S3[c1,i,j,k]*S1[c1,j]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + 3*S3[c1,i,i,j]*S3[c1,i,i,k]*S1[c1,j]*S1[c1,k]*mx02[j]*mx02[k]*mx04[i] \
                            + S1[c1,i]*S2[c1,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 4*S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + S1[c1,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 6*S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,j,k]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 2*S2[c1,i,i]*S2[c1,i,k]*S3[c1,j,j,j]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            - S2[c1,i,i]*S3[c1,i,j,j]*S1[c1,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 4*S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,k]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 3*S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            - 4*S2[c1,i,j]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 4*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,k]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 2*S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,j]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 5*S3[c1,i,i,j]*S3[c1,i,j,k]*S1[c1,j]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + 2*S3[c1,i,i,k]*S3[c1,i,j,j]*S1[c1,j]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx06[i])/2 \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[j]*mx05[i])/6 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx05[i] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[k]*mx04[j]*mx05[i])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,j]*mx02[k]*mx04[j]*mx05[i])/3 \
                            + (7*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/6 \
                            + (S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (2*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (2*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/9 \
                            + (5*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,k]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/9 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/2 \
                            + S3[c1,i,i,i]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j] \
                            + (S3[c1,i,i,i]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j])/6 \
                            + (S3[c1,i,i,i]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,j,j,j]*mx04[i]*mx03[k]*mx04[j])/9 \
                            + (2*S3[c1,i,i,j]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (4*S3[c1,i,i,j]*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,k]*mx04[i]*mx03[k]*mx04[j])/3 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,k,k]*mx04[i]*mx03[k]*mx04[j])/2 \
                            + S2[c1,i,i]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx05[k] \
                            + (S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/3 \
                            + 2*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx05[k] \
                            + (7*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (4*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx05[k])/3 \
                            + 3*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx05[k] \
                            + (S3[c1,i,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx05[k])/2 \
                            + S3[c1,i,i,k]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx05[k] \
                            + (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,k]*mx02[j]*mx02[k]*mx05[i])/4 \
                            + (2*S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]*mx02[j]*mx02[k]*mx05[i])/3 \
                            + S3[c1,i,i,i]*S3[c1,i,i,j]*S1[c1,j]*S2[c1,k,k]*mx02[j]*mx02[k]*mx05[i] \
                            + 2*S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,k]*S1[c1,k]*mx02[j]*mx02[k]*mx05[i] \
                            + (S1[c1,i]*S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i])/3 \
                            + (3*S2[c1,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i])/2 \
                            + 3*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + S2[c1,i,j]*S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S2[c1,i,j]*S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + (2*S2[c1,i,k]*S3[c1,i,i,i]*S3[c1,j,j,j]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i])/3 \
                            + S3[c1,i,i,i]*S3[c1,i,j,j]*S1[c1,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 2*S3[c1,i,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + 3*S3[c1,i,i,j]*S3[c1,i,i,k]*S2[c1,j,j]*S1[c1,k]*mx02[k]*mx03[j]*mx04[i] \
                            + S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (S1[c1,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k])/3 \
                            + (S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            - S2[c1,i,i]*S3[c1,i,j,j]*S1[c1,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + (S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (S2[c1,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k])/6 \
                            - 2*S2[c1,i,j]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + 2*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,j,k]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (2*S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,j]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k])/3 \
                            + (S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + 3*S3[c1,i,i,j]*S3[c1,i,j,k]*S1[c1,j]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k] \
                            + 3*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k] \
                            + (3*S3[c1,i,i,k]*S3[c1,i,j,j]*S1[c1,j]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + (3*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + 2*S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,k]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k] \
                            - 6*S2[c1,i,i]*S1[c1,j]*S3[c1,j,k,k]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k] \
                            + (7*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[k]*mx05[i]*mx05[j])/72 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[k]*mx05[i]*mx05[j])/12 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[k]*mx04[j]*mx05[i])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx03[k]*mx04[j]*mx05[i])/6 \
                            + (7*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx04[i]*mx04[j]*mx04[k])/24 \
                            + (S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx04[i]*mx04[j]*mx04[k])/36 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx04[i]*mx04[j]*mx04[k])/8 \
                            + (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,k,k]*mx04[i]*mx04[j]*mx04[k])/6 \
                            + (3*S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx06[k])/4 \
                            + (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx06[k])/18 \
                            + (S2[c1,i,i]*S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx05[i])/6 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S2[c1,k,k]*mx02[k]*mx03[j]*mx05[i])/2 \
                            + S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S1[c1,k]*mx02[k]*mx03[j]*mx05[i] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S1[c1,k]*mx02[k]*mx03[j]*mx05[i])/3 \
                            + (5*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,k]*S1[c1,k]*mx02[k]*mx04[i]*mx04[j])/9 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,j]*S1[c1,k]*mx02[k]*mx04[i]*mx04[j])/3 \
                            + (4*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,j]*S1[c1,k]*mx02[k]*mx04[i]*mx04[j])/9 \
                            + (2*S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + (S1[c1,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/6 \
                            + (S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/4 \
                            + (S2[c1,i,i]*S3[c1,i,j,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (3*S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (3*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (3*S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/4 \
                            + (3*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (5*S3[c1,i,i,j]*S3[c1,i,j,k]*S1[c1,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/6 \
                            + 6*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S1[c1,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            + (4*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S1[c1,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,j]*S1[c1,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + 3*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S1[c1,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + 3*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx03[j]*mx04[k] \
                            + 3*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S1[c1,k]*mx03[i]*mx03[j]*mx04[k] \
                            + (S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S1[c1,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + (3*S1[c1,i]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (S1[c1,i]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (S2[c1,i,j]*S2[c1,i,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (3*S2[c1,i,j]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (3*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + 2*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k] \
                            + (S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (S3[c1,i,j,k]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + 6*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx06[k] \
                            + 3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,k]*S1[c1,k]*mx02[i]*mx02[j]*mx06[k] \
                            - S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j] \
                            + S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,j,k]*S1[c1,k]*mx03[i]*mx02[k]*mx03[j] \
                            - 2*S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,k]*S2[c1,k,k]*mx02[j]*mx03[i]*mx03[k] \
                            - 2*S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*S1[c1,k]*mx02[j]*mx03[i]*mx03[k] \
                            - 2*S2[c1,i,i]*S1[c1,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx04[k] \
                            - 3*S2[c1,i,i]*S1[c1,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k] \
                            - 6*S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S1[c1,k]*mx02[i]*mx02[j]*mx04[k] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[k]*mx04[j]*mx05[i])/6 \
                            + (S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx04[i]*mx03[k]*mx04[j])/9 \
                            - (S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,k,k,k]*mx04[i]*mx03[k]*mx04[j])/9 \
                            + (5*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,k]*S2[c1,k,k]*mx04[i]*mx03[k]*mx04[j])/18 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,j]*S2[c1,k,k]*mx04[i]*mx03[k]*mx04[j])/6 \
                            + (2*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,j]*S2[c1,k,k]*mx04[i]*mx03[k]*mx04[j])/9 \
                            + (5*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/12 \
                            + (3*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/4 \
                            + (S2[c1,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/12 \
                            + (2*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/3 \
                            + (S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/2 \
                            + (2*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/3 \
                            + (S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (5*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/12 \
                            + (2*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx05[k])/3 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/3 \
                            + (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/4 \
                            + (3*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx05[k])/2 \
                            + (S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S2[c1,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            + (3*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            + (S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            + (S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            + (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            - (S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx03[k])/3 \
                            + (S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + (S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*S1[c1,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k])/2 \
                            - (2*S3[c1,i,i,i]*S1[c1,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx04[k])/3 \
                            - S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[j]*mx03[i]*mx04[k] \
                            - 2*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S1[c1,k]*mx02[j]*mx03[i]*mx04[k] \
                            - S2[c1,i,i]*S1[c1,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx04[k] \
                            - S2[c1,i,i]*S2[c1,j,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx04[k] \
                            - (3*S2[c1,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx03[j]*mx04[k])/2 \
                            - 3*S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx03[j]*mx04[k] \
                            - 3*S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S1[c1,k]*mx02[i]*mx03[j]*mx04[k] \
                            - S2[c1,i,i]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx05[k] \
                            - 3*S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx05[k] \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[k]*mx04[j]*mx05[i])/18 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx04[i]*mx04[j]*mx04[k])/27 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx04[i]*mx04[j]*mx04[k])/54 \
                            + (S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx04[i]*mx04[j]*mx04[k])/54 \
                            + (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx06[k])/18 \
                            + (2*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx06[k])/9 \
                            + (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx06[k])/18 \
                            + (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx06[k])/2 \
                            + (S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx06[k])/18 \
                            + (S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx08[k])/2 \
                            - (S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/6 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S1[c1,k]*mx03[i]*mx03[j]*mx04[k])/3 \
                            + (S3[c1,i,i,j]*S3[c1,i,j,j]*S1[c1,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx04[k])/2 \
                            - (S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx05[k])/3 \
                            - S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[j]*mx03[i]*mx05[k] \
                            - (S2[c1,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx05[k])/2 \
                            - S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx05[k] \
                            - (3*S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx03[j]*mx05[k])/2 \
                            - (S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k] \
                            + (S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + 3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx06[k] \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/3 \
                            - (S2[c1,i,i]*S2[c1,j,j]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx04[k])/2 \
                            - 3*S2[c1,i,i]*S3[c1,j,j,k]*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx04[k] \
                            - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/12 \
                            - (S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx05[k])/6 \
                            + (S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx05[k])/4 \
                            - (S2[c1,i,i]*S3[c1,j,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx04[j]*mx05[k])/6 \
                            - (S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx06[k])/3 \
                            - (S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx06[k])/2 \
                            + (S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/2 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/6 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,j]*S1[c1,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx03[k])/3 \
                            - S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]*S2[c1,k,k]*mx02[j]*mx03[i]*mx04[k] \
                            - (S2[c1,i,i]*S3[c1,j,j,j]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx04[k])/3 \
                            - (S2[c1,i,i]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx05[k])/4 \
                            - S2[c1,i,i]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx05[k] \
                            - (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx04[j]*mx05[k])/18 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx06[k])/18 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx05[k])/3 \
                            - (S2[c1,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx05[k])/6 \
                            - (S2[c1,i,i]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k] \
                            + (2*S3[c1,i,i,j]*S3[c1,j,k,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/3 \
                            + (S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx06[k])/2 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx06[k])/6 \
                            + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/3 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx07[k])/4 \
                            - S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - 4*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[j]*mx02[k]*mx03[j]*mx04[i])/2 \
                            - 3*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,j,j]*mx02[j]*mx02[k]*mx03[j]*mx04[i] \
                            - S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - S1[c1,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - S2[c1,i,i]*S2[c1,i,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - 2*S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - (S2[c1,i,i]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j])/2 \
                            - 2*S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - 2*S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - 2*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            + S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j]*mx03[k] \
                            - (S3[c1,i,i,i]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]*mx02[k]*mx03[j]*mx04[i])/2 \
                            - S3[c1,i,i,i]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[j]*mx02[k]*mx03[j]*mx04[i] \
                            - (S3[c1,i,i,i]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,j,j]*mx02[j]*mx02[k]*mx03[j]*mx04[i])/6 \
                            - (3*S3[c1,i,i,j]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[j]*mx02[k]*mx03[j]*mx04[i])/2 \
                            - S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k] \
                            - 2*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k] \
                            - S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k] \
                            - 2*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[j]*mx04[k] \
                            - (S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            - (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/6 \
                            - 2*S2[c1,i,j]*S2[c1,i,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - (3*S2[c1,i,j]*S2[c1,i,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            - 2*S2[c1,i,j]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - (9*S2[c1,i,j]*S3[c1,i,k,k]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            - 4*S2[c1,i,k]*S3[c1,i,j,k]*S1[c1,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - 6*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - (3*S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            - 6*S3[c1,i,j,k]*S3[c1,i,k,k]*S1[c1,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - (S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx03[j]*mx04[k])/2 \
                            - S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx03[j]*mx04[k] \
                            - (S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[j]*mx03[i]*mx03[j]*mx04[k])/2 \
                            - S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[j]*mx05[k] \
                            - (S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j])/3 \
                            - 2*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,j,j]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,j,j]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j] \
                            - 6*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - 3*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - 3*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - 6*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - 3*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - 3*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - (3*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k])/2 \
                            - 6*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k] \
                            + (S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[j]*mx03[k])/6 \
                            + (S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[j]*mx03[k])/3 \
                            - (2*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k])/3 \
                            - (S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k])/3 \
                            - S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k] \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k])/2 \
                            - (S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            - (S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            - 2*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k] \
                            - (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/6 \
                            - (S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            - (2*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/3 \
                            - S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k] \
                            - (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/6 \
                            - S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k] \
                            - (3*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            - (S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/6 \
                            - (3*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k])/2 \
                            - 3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k] \
                            - (S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k])/2 \
                            - 2*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k] \
                            - (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[k])/2 \
                            - (S2[c1,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[j])/2 \
                            + (3*S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            - 2*S2[c1,i,j]*S3[c1,i,j,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - 2*S3[c1,i,i,k]*S1[c1,j]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - 3*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - (S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[j]*mx04[k])/3 \
                            - (S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[j]*mx04[k])/6 \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[j]*mx05[k])/6 \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]*mx04[k])/2 \
                            - S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]*mx04[k] \
                            - (S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx05[k])/2 \
                            - (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,j,k]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j])/3 \
                            - (S2[c1,i,i]*S2[c1,j,j]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[j]*mx03[k])/6 \
                            + (S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[j]*mx03[k])/2 \
                            + (S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[k])/3 \
                            + S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[k] \
                            - S2[c1,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - 3*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - (3*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            - 3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k] \
                            - S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k] \
                            - 3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k] \
                            - (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k])/3 \
                            - (S3[c1,i,i,j]*S3[c1,j,j,j]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k])/2 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[j]*mx04[k])/2 \
                            - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/6 \
                            - (S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/3 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/6 \
                            - (S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx03[i]*mx02[k]*mx03[j]*mx03[k])/4 \
                            - (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k])/2 \
                            - S3[c1,i,j,j]*S3[c1,i,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k] \
                            - S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k] \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[k])/6 \
                            - (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k])/6 \
                            + (S2[c1,i,i]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            + S2[c1,i,i]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - S3[c1,i,i,j]*S1[c1,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - 2*S3[c1,i,i,j]*S2[c1,j,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - 2*S3[c1,i,i,j]*S3[c1,j,k,k]*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - S3[c1,i,i,k]*S2[c1,j,j]*S1[c1,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k] \
                            - (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S1[c1,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[k])/2 \
                            - (S3[c1,i,i,j]*S3[c1,j,j,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[j]*mx05[k])/6 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[j]*mx05[k])/6 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx03[j]*mx03[k]*mx04[k])/2 \
                            + (S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[k])/3 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[k])/3 \
                            - (S2[c1,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k])/6 \
                            - (S3[c1,i,i,j]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[k]*mx03[j]*mx03[k])/2 \
                            + (S2[c1,i,i]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k])/2 \
                            - S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k] \
                            - (S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[k])/2 \
                            + (S3[c1,i,i,i]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx02[k]*mx04[k])/6 \
                            - (S3[c1,i,i,j]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k])/2 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,j]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[k]*mx03[j]*mx04[k])/6 \
                            - (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[k])/3 \
                            - (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k])/3 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[k])/4 \
                            - (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[k])/4 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,j,j]*mx02[i]*mx02[j]**2*mx03[i]*mx02[k])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]**2*mx03[i]*mx02[k])/2 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[i]*mx02[j]**2*mx03[i]*mx02[k])/2 \
                            + (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]**2*mx03[k])/6 \
                            + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]**2*mx03[k])/3 \
                            + (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx02[k]**2*mx03[k])/4 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,k]**2*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[i]*mx02[k]*mx03[j])/6 \
                            + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[i]*mx02[j]*mx03[i]*mx02[k]*mx03[j])/12 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[i]*mx02[k]*mx03[j])/12 \
                            + (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[i]*mx02[j]*mx03[i]*mx02[k]*mx03[j])/12
                            
                        for l in range(n):
                            if i!=j and i!=k and i!=l and j!=k and j!=l and k!=l:
                                mu4Y3ijkl[c1,i,j,k,l] =  + (S3[c1,i,i,i]*S3[c1,j,k,l]**3*mx03[i]*mx03[j]*mx03[k]*mx03[l])/12 \
                                + S3[c1,i,j,k]**3*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k] \
                                - (S3[c1,i,j,k]**3*S3[c1,l,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/12 \
                                + (S3[c1,i,j,j]**2*S3[c1,i,k,l]**2*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + (3*S3[c1,i,j,k]**2*S3[c1,i,j,l]**2*mx02[k]*mx04[i]*mx02[l]*mx04[j])/2 \
                                + S1[c1,i]**2*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (3*S3[c1,i,l,l]**2*S2[c1,j,k]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l])/4 \
                                + (S3[c1,i,i,j]**2*S3[c1,j,k,l]**2*mx02[k]*mx04[i]*mx02[l]*mx04[j])/2 \
                                + (S3[c1,i,i,k]**2*S3[c1,j,j,l]**2*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (S3[c1,i,i,l]**2*S3[c1,j,j,k]**2*mx02[k]*mx04[i]*mx02[l]*mx04[j])/8 \
                                + (3*S3[c1,i,l,l]**2*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + 3*S3[c1,i,j,l]**2*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (3*S3[c1,i,j,l]**2*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + (S3[c1,i,j,k]**2*S2[c1,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l])/4 \
                                + (S3[c1,i,j,k]**2*S3[c1,l,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx06[l])/36 \
                                - (S3[c1,i,j,k]**2*S2[c1,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2)/4 \
                                - (S3[c1,i,j,k]**2*S3[c1,l,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/36 \
                                + S1[c1,i]*S2[c1,i,i]*S3[c1,j,k,l]**2*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (17*S3[c1,i,j,j]*S3[c1,i,j,l]**2*S3[c1,i,k,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/24 \
                                + (13*S3[c1,i,j,j]*S3[c1,i,j,k]**2*S3[c1,i,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/24 \
                                + (S3[c1,i,j,j]**2*S3[c1,i,k,k]*S3[c1,i,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/8 \
                                + (3*S2[c1,i,i]*S3[c1,i,l,l]*S2[c1,j,k]**2*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 2*S2[c1,i,l]*S3[c1,i,i,l]*S2[c1,j,k]**2*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,l]**2*S3[c1,i,j,k]*S2[c1,j,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,j]*S2[c1,i,k]**2*S3[c1,j,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,k]**2*S3[c1,i,l,l]*S2[c1,j,j]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S2[c1,i,k]*S2[c1,i,l]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 2*S1[c1,i]*S3[c1,i,l,l]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (7*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + 2*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + 2*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (2*S2[c1,i,j]*S3[c1,i,k,l]**2*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,j,k]*S3[c1,i,k,l]**2*S2[c1,j,j]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (4*S2[c1,i,k]*S3[c1,i,j,l]**2*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S2[c1,i,k]*S3[c1,i,j,k]**2*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (10*S2[c1,i,l]*S3[c1,i,j,k]**2*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (4*S3[c1,i,j,k]**2*S3[c1,i,k,l]*S2[c1,j,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S3[c1,i,j,k]**2*S3[c1,i,l,l]*S2[c1,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + 8*S2[c1,i,k]*S3[c1,i,k,l]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 4*S2[c1,i,l]*S3[c1,i,k,k]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S2[c1,i,l]*S3[c1,i,l,l]*S3[c1,j,k,k]**2*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S2[c1,i,l]*S3[c1,i,l,l]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,k,l]**2*mx02[k]*mx04[i]*mx02[l]*mx04[j])/3 \
                                + (S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,j,j,l]**2*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (S3[c1,i,i,j]*S3[c1,i,k,l]**2*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/6 \
                                + (S3[c1,i,i,i]*S3[c1,i,l,l]*S3[c1,j,j,k]**2*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (31*S3[c1,i,i,j]*S3[c1,i,j,l]**2*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/24 \
                                + (3*S3[c1,i,i,k]*S3[c1,i,j,l]**2*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + (7*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,j,k]**2*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/24 \
                                + (15*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (3*S3[c1,i,i,l]*S3[c1,i,j,k]**2*S3[c1,j,j,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + (S3[c1,i,i,j]*S3[c1,i,k,l]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx03[k]*mx03[l])/2 \
                                + (7*S3[c1,i,i,j]*S3[c1,i,l,l]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx02[l]*mx04[k])/16 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx03[k]*mx03[l])/2 \
                                + (5*S3[c1,i,j,j]*S3[c1,i,k,l]**2*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx03[k]*mx03[l])/2 \
                                + (15*S3[c1,i,i,l]*S3[c1,i,j,l]*S3[c1,j,k,k]**2*mx03[i]*mx03[j]*mx02[l]*mx04[k])/16 \
                                + (9*S3[c1,i,j,k]*S3[c1,i,k,l]**2*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (S3[c1,i,k,k]*S3[c1,i,k,l]**2*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (5*S3[c1,i,j,j]*S3[c1,i,k,k]**2*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/16 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]**2*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (9*S3[c1,i,j,l]*S3[c1,i,k,k]**2*S3[c1,j,j,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/16 \
                                + (3*S3[c1,i,j,l]**2*S3[c1,i,k,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (S3[c1,i,k,k]**2*S3[c1,i,l,l]*S3[c1,j,j,j]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/16 \
                                + (3*S3[c1,i,j,k]**2*S3[c1,i,k,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + 6*S3[c1,i,j,k]**2*S3[c1,i,k,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k] \
                                + (3*S3[c1,i,j,k]**2*S3[c1,i,l,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (S3[c1,i,j,l]**2*S3[c1,i,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + 3*S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (S3[c1,i,j,k]**2*S3[c1,i,l,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (3*S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,k,k]**2*mx02[i]*mx02[j]*mx03[l]*mx05[k])/4 \
                                + S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,k,l]**2*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                - 3*S2[c1,i,j]*S3[c1,i,j,l]**2*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l] \
                                + (2*S1[c1,i]*S3[c1,i,j,j]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + 4*S2[c1,i,j]**2*S2[c1,i,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S2[c1,i,j]**2*S3[c1,i,k,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,j]**2*S3[c1,i,l,l]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S3[c1,i,k,l]**2*S1[c1,j]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (S3[c1,i,l,l]**2*S1[c1,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (2*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                - (2*S2[c1,i,j]*S3[c1,i,j,l]**2*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (S3[c1,i,j,k]*S3[c1,i,j,l]**2*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (3*S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (5*S2[c1,i,j]*S3[c1,i,j,k]**2*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,j,j]*S3[c1,i,l,l]*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (4*S3[c1,i,j,k]**2*S3[c1,i,j,l]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (2*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + 6*S2[c1,i,l]**2*S3[c1,j,k,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 4*S3[c1,i,k,l]**2*S2[c1,j,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S3[c1,i,k,l]**2*S2[c1,j,l]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,k,k]**2*S2[c1,j,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S3[c1,i,l,l]**2*S2[c1,j,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,l,l]**2*S2[c1,j,l]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + 3*S1[c1,i]*S3[c1,i,j,k]**2*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S2[c1,i,l]*S3[c1,i,j,k]**2*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,k,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx04[k])/8 \
                                + (3*S3[c1,i,i,i]*S3[c1,j,k,k]*S3[c1,j,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (5*S3[c1,i,i,l]**2*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (3*S3[c1,i,i,i]*S3[c1,j,k,k]**2*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/16 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,j]**2*S3[c1,k,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/12 \
                                + (S3[c1,i,i,k]**2*S3[c1,j,j,j]*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (S3[c1,i,j,j]**2*S3[c1,i,i,l]*S3[c1,k,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + S3[c1,i,j,k]*S3[c1,i,j,l]**2*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k] \
                                + (S3[c1,i,i,j]**2*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + (3*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + (S3[c1,i,j,k]*S3[c1,i,j,l]**2*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + 3*S3[c1,i,j,k]**2*S3[c1,i,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k] \
                                + (S3[c1,i,j,k]**2*S3[c1,i,j,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + 3*S3[c1,i,k,l]**2*S3[c1,j,k,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (3*S3[c1,i,k,k]**2*S3[c1,j,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/4 \
                                + (S3[c1,i,l,l]**2*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + (S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,l]**2*mx02[j]*mx03[i]*mx02[k]*mx02[l])/6 \
                                + (S2[c1,i,i]*S3[c1,j,j,j]*S2[c1,k,l]**2*mx02[i]*mx02[k]*mx03[j]*mx02[l])/2 \
                                + (3*S2[c1,i,i]*S2[c1,j,j]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l])/8 \
                                + (S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                - 3*S3[c1,i,i,l]*S2[c1,j,l]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                + S1[c1,i]**2*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (S1[c1,i]**2*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/2 \
                                + 3*S2[c1,i,i]*S3[c1,j,k,l]**2*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S3[c1,i,i,l]*S3[c1,j,k,l]**2*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + S3[c1,i,k,l]**2*S3[c1,j,j,l]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                - (3*S3[c1,i,l,l]**2*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/8 \
                                + (S3[c1,i,l,l]**2*S3[c1,j,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/6 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,l]**2*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                + (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,k,l,l]**2*mx02[j]*mx03[i]*mx02[k]*mx04[l])/4 \
                                + 2*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                - (3*S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx03[k]*mx04[l])/4 \
                                + (3*S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + (S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 2*S3[c1,i,i,j]*S3[c1,j,l,l]*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,l]*S2[c1,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                - S3[c1,i,i,l]*S2[c1,j,l]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + 3*S2[c1,i,i]*S3[c1,j,k,l]**2*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                - (S2[c1,i,i]*S3[c1,j,l,l]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/4 \
                                + (S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + 4*S3[c1,i,i,k]*S3[c1,j,k,l]**2*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S3[c1,i,i,l]*S3[c1,j,k,l]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S2[c1,i,l]**2*S3[c1,j,j,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + S2[c1,i,l]**2*S3[c1,j,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                - 3*S2[c1,i,i]*S3[c1,j,k,l]**2*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l] \
                                + (S2[c1,i,i]*S3[c1,j,l,l]**2*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/4 \
                                - (S3[c1,i,i,l]*S3[c1,j,l,l]**2*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + 2*S3[c1,i,k,l]**2*S3[c1,j,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,k,l]**2*S3[c1,j,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                - (S3[c1,i,l,l]**2*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/4 \
                                + (S3[c1,i,l,l]**2*S3[c1,j,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                - (S3[c1,i,l,l]**2*S3[c1,j,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/4 \
                                + 2*S2[c1,i,j]**2*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,k,l]**2*mx03[i]*mx03[j]*mx02[l]*mx04[k])/24 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,l,l]**2*mx03[i]*mx02[k]*mx03[j]*mx04[l])/24 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,k,l,l]**2*mx02[j]*mx03[i]*mx03[k]*mx04[l])/4 \
                                - (S3[c1,i,i,i]*S3[c1,j,l,l]**2*S3[c1,k,k,k]*mx02[j]*mx03[i]*mx03[k]*mx04[l])/12 \
                                + (S3[c1,i,i,j]**2*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/6 \
                                + (S3[c1,i,i,j]**2*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                - S3[c1,i,i,i]*S3[c1,j,k,l]**2*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx03[k]*mx04[l] \
                                + S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx05[k])/8 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,k,l]**2*mx02[i]*mx02[j]*mx03[l]*mx05[k])/8 \
                                + 2*S3[c1,i,i,k]*S3[c1,j,k,l]**2*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (2*S3[c1,i,i,l]*S3[c1,j,k,l]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/3 \
                                + (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx06[l])/4 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx06[l])/16 \
                                + (S3[c1,i,i,l]*S3[c1,j,k,k]**2*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                - (S3[c1,i,i,l]*S3[c1,j,l,l]**2*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx05[l])/6 \
                                + S3[c1,i,k,l]**2*S3[c1,j,j,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (S3[c1,i,k,l]**2*S3[c1,j,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/3 \
                                + (S3[c1,i,i,k]*S3[c1,j,l,l]**2*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + (S3[c1,i,i,l]*S3[c1,j,l,l]**2*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + (S3[c1,i,k,k]**2*S3[c1,j,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/4 \
                                - (S3[c1,i,l,l]**2*S3[c1,j,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx05[l])/12 \
                                + (S3[c1,i,l,l]**2*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + (S3[c1,i,l,l]**2*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/16 \
                                + (S3[c1,i,i,i]*S2[c1,j,k]**2*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                - (S2[c1,i,i]*S2[c1,j,k]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/3 \
                                + S3[c1,i,i,k]*S3[c1,j,k,l]**2*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 2*S3[c1,i,i,l]*S3[c1,j,k,l]**2*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + S3[c1,i,k,l]**2*S3[c1,j,j,k]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (7*S3[c1,i,j,l]**2*S1[c1,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,j,l]**2*S2[c1,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + S3[c1,i,j,l]**2*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l])/4 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx04[l])/8 \
                                + (S2[c1,i,i]*S3[c1,j,k,k]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + (S3[c1,i,i,l]*S2[c1,j,k]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (S3[c1,i,i,l]*S3[c1,j,k,k]**2*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S2[c1,i,i]*S3[c1,j,k,l]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + S3[c1,i,i,l]*S3[c1,j,k,l]**2*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (3*S2[c1,i,j]**2*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S3[c1,i,j,l]**2*S2[c1,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + S3[c1,i,j,l]**2*S2[c1,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,k,k]**2*S3[c1,j,j,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + 3*S3[c1,i,j,k]**2*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,j,l]**2*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + S2[c1,i,j]**2*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                - 3*S3[c1,i,j,l]**2*S2[c1,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx06[l])/36 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx06[l])/72 \
                                + (S3[c1,i,i,k]*S3[c1,j,k,k]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/6 \
                                + (S3[c1,i,j,l]**2*S3[c1,k,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + (S3[c1,i,k,k]**2*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/12 \
                                + (S3[c1,i,i,l]*S3[c1,j,k,l]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/3 \
                                + (3*S3[c1,i,j,k]**2*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                - (2*S2[c1,i,j]**2*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/3 \
                                - (S3[c1,i,j,l]**2*S3[c1,k,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (S3[c1,i,j,k]**2*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (9*S3[c1,i,j,k]**2*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (2*S3[c1,i,j,k]**2*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + S3[c1,i,j,l]**2*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                - (S2[c1,i,j]**2*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/12 \
                                + (S3[c1,i,j,k]**2*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,j,k]**2*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                - (S3[c1,i,j,l]**2*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/2 \
                                - (S3[c1,i,j,l]**2*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,j,l]**2*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,j,k]**2*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/6 \
                                - (S3[c1,i,j,l]**2*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx05[l])/6 \
                                + (S3[c1,i,j,l]**2*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/6 \
                                + (S3[c1,i,j,k]**2*S1[c1,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S3[c1,i,j,k]**2*S2[c1,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                - (7*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,k,l]**2*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/6 \
                                - (S2[c1,i,i]*S3[c1,i,k,l]**2*S3[c1,j,j,j]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/6 \
                                - (3*S2[c1,i,i]*S3[c1,i,j,l]**2*S3[c1,j,k,k]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/2 \
                                - (S3[c1,i,j,j]*S3[c1,i,k,l]**2*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/3 \
                                - (S2[c1,i,i]*S3[c1,i,j,k]**2*S3[c1,j,l,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/2 \
                                - (S3[c1,i,j,l]**2*S3[c1,i,k,k]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/2 \
                                - (S3[c1,i,j,k]**2*S3[c1,i,l,l]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/2 \
                                - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,l]**2*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/3 \
                                - (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2)/4 \
                                - (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2)/8 \
                                - (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/36 \
                                - (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,l]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/72 \
                                - (S3[c1,i,i,l]*S3[c1,j,k,l]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/3 \
                                + (S3[c1,i,j,l]**2*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[l]**2*mx03[k])/2 \
                                - (S3[c1,i,j,l]**2*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/6 \
                                + 2*S1[c1,i]*S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S2[c1,i,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 8*S1[c1,i]*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S1[c1,i]*S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S1[c1,i]*S3[c1,i,j,k]*S3[c1,i,k,l]*S2[c1,j,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S1[c1,i]*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S2[c1,i,j]*S2[c1,i,k]*S2[c1,i,l]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,k,l]*S2[c1,j,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,l,l]*S2[c1,j,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 8*S2[c1,i,j]*S2[c1,i,l]*S3[c1,i,k,l]*S2[c1,j,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S2[c1,i,k]*S2[c1,i,l]*S3[c1,i,j,l]*S2[c1,j,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,k]*S2[c1,i,l]*S3[c1,i,k,l]*S2[c1,j,j]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,i,l,l]*S1[c1,j]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,i,k,l]*S1[c1,j]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (5*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,i,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/2 \
                                + 2*S1[c1,i]*S2[c1,i,k]*S2[c1,j,l]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S2[c1,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 4*S2[c1,i,k]*S2[c1,i,l]*S1[c1,j]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,k]*S3[c1,i,k,l]*S1[c1,j]*S2[c1,j,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,l]*S3[c1,i,k,l]*S1[c1,j]*S2[c1,j,k]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (16*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (5*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + 2*S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (8*S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + 2*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (2*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (16*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (4*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (4*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (10*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,j,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + 2*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (2*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,i,k,k]*S2[c1,j,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (4*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,i,l,l]*S2[c1,j,j]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (S1[c1,i]*S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 3*S1[c1,i]*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S1[c1,i]*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S2[c1,i,i]*S2[c1,i,k]*S2[c1,j,k]*S3[c1,j,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 3*S2[c1,i,i]*S2[c1,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,i]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S2[c1,i,i]*S3[c1,i,k,l]*S2[c1,j,k]*S2[c1,j,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (S2[c1,i,i]*S3[c1,i,l,l]*S1[c1,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 6*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,j]*S2[c1,i,l]*S3[c1,i,j,k]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,j]*S2[c1,i,l]*S3[c1,i,j,l]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,i,l,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,k]*S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,k]*S3[c1,i,i,l]*S2[c1,j,k]*S2[c1,j,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,l]*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,l]*S3[c1,i,i,k]*S2[c1,j,k]*S2[c1,j,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,i,k,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,l]*S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,i,j,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/2 \
                                + (9*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (7*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/2 \
                                + (19*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (31*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (35*S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,j,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (25*S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/12 \
                                + (17*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (23*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (7*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,j,j,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (13*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (5*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,j,j,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + (7*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (5*S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (29*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (61*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,i,j,l]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (17*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,i,k,k]*S3[c1,j,j,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (13*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (7*S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,j,j,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + (7*S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (11*S3[c1,i,i,l]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (5*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (5*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/8 \
                                + 6*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k] \
                                + 6*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k] \
                                + (9*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (9*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/8 \
                                + (9*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                + S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l] \
                                - (3*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,l,l]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/2 \
                                + 3*S2[c1,i,k]*S2[c1,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (3*S2[c1,i,k]*S3[c1,i,k,k]*S2[c1,j,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 6*S2[c1,i,k]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (3*S2[c1,i,k]*S3[c1,i,l,l]*S2[c1,j,l]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,l]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 6*S2[c1,i,l]*S3[c1,i,k,k]*S2[c1,j,l]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 6*S2[c1,i,l]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 6*S2[c1,i,l]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (3*S2[c1,i,l]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 3*S3[c1,i,k,k]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S3[c1,i,k,k]*S3[c1,i,l,l]*S1[c1,j]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S3[c1,i,k,l]*S3[c1,i,l,l]*S1[c1,j]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 2*S1[c1,i]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (2*S1[c1,i]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (3*S2[c1,i,k]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 3*S2[c1,i,l]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 3*S2[c1,i,l]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 2*S3[c1,i,k,l]*S3[c1,i,l,l]*S1[c1,j]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 3*S3[c1,i,k,l]*S3[c1,i,l,l]*S2[c1,j,k]*S2[c1,j,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 2*S1[c1,i]*S2[c1,i,j]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S2[c1,i,j]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S2[c1,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S2[c1,i,k]*S2[c1,j,j]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S2[c1,i,l]*S2[c1,j,j]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S3[c1,i,j,k]*S1[c1,j]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S3[c1,i,j,l]*S1[c1,j]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S3[c1,i,j,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S3[c1,i,k,l]*S3[c1,j,j,l]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 6*S2[c1,i,j]*S2[c1,i,k]*S1[c1,j]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 3*S2[c1,i,j]*S2[c1,i,l]*S1[c1,j]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 3*S2[c1,i,j]*S2[c1,i,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,j]*S3[c1,i,k,l]*S1[c1,j]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S2[c1,i,j]*S3[c1,i,k,l]*S2[c1,j,l]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S2[c1,i,l]*S3[c1,i,k,l]*S2[c1,j,j]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (5*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (7*S2[c1,i,i]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S2[c1,i,i]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (S2[c1,i,i]*S3[c1,i,k,k]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/2 \
                                + (5*S2[c1,i,i]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S2[c1,i,i]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S2[c1,i,i]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/2 \
                                + (2*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                - (7*S2[c1,i,j]*S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                + (7*S2[c1,i,j]*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (8*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (3*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/2 \
                                + (7*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (5*S2[c1,i,k]*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S2[c1,i,k]*S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (7*S2[c1,i,l]*S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (5*S2[c1,i,l]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S2[c1,i,l]*S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (3*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/2 \
                                + (7*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                + S2[c1,i,l]*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + 2*S3[c1,i,i,j]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (4*S3[c1,i,i,j]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S3[c1,i,i,j]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (8*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + 2*S3[c1,i,i,k]*S3[c1,i,j,l]*S2[c1,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (4*S3[c1,i,i,k]*S3[c1,i,j,l]*S2[c1,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S3[c1,i,i,k]*S3[c1,i,k,l]*S2[c1,j,j]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (2*S3[c1,i,i,k]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,i,k]*S3[c1,i,l,l]*S2[c1,j,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,i,k]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,l]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (8*S3[c1,i,i,l]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (5*S3[c1,i,i,l]*S3[c1,i,j,k]*S2[c1,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (4*S3[c1,i,i,l]*S3[c1,i,j,l]*S2[c1,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,i,l]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/2 \
                                + (S3[c1,i,i,l]*S3[c1,i,k,k]*S2[c1,j,l]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S3[c1,i,i,l]*S3[c1,i,k,l]*S2[c1,j,j]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (2*S3[c1,i,i,l]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,j,j,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + 4*S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 4*S2[c1,i,k]*S3[c1,i,k,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 4*S2[c1,i,k]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S2[c1,i,l]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 8*S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S3[c1,i,k,k]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 4*S3[c1,i,k,k]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S3[c1,i,k,k]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,k,k]*S3[c1,i,l,l]*S2[c1,j,l]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S3[c1,i,k,l]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,j,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S2[c1,i,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + 2*S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (2*S2[c1,i,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + 2*S3[c1,i,k,l]*S3[c1,i,l,l]*S2[c1,j,l]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + S1[c1,i]*S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S1[c1,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 2*S1[c1,i]*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (9*S2[c1,i,i]*S2[c1,i,j]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S2[c1,i,i]*S2[c1,i,j]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + (3*S2[c1,i,i]*S2[c1,i,k]*S2[c1,j,j]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + (3*S2[c1,i,i]*S3[c1,i,j,k]*S1[c1,j]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + (3*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,l,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 3*S2[c1,i,i]*S3[c1,i,j,l]*S2[c1,j,k]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,i]*S3[c1,i,j,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S2[c1,i,i]*S3[c1,i,k,l]*S2[c1,j,j]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + (3*S2[c1,i,i]*S3[c1,i,l,l]*S2[c1,j,j]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/4 \
                                + S2[c1,i,i]*S3[c1,i,l,l]*S3[c1,j,j,k]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,j]*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 5*S2[c1,i,j]*S3[c1,i,i,l]*S2[c1,j,k]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,j]*S3[c1,i,i,l]*S2[c1,j,l]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,j]*S3[c1,i,i,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 6*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,i,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,k]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,i,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S2[c1,i,k]*S3[c1,i,i,l]*S2[c1,j,j]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 6*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,i,j,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,l]*S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,l]*S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S2[c1,i,l]*S3[c1,i,i,j]*S3[c1,j,k,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (S2[c1,i,l]*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 3*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (S2[c1,i,l]*S3[c1,i,i,l]*S2[c1,j,j]*S2[c1,k,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 2*S2[c1,i,l]*S3[c1,i,i,l]*S3[c1,j,j,k]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S3[c1,i,i,j]*S3[c1,i,k,l]*S1[c1,j]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S3[c1,i,i,j]*S3[c1,i,k,l]*S2[c1,j,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S3[c1,i,i,l]*S3[c1,i,j,k]*S1[c1,j]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S3[c1,i,i,l]*S3[c1,i,j,k]*S2[c1,j,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                - (3*S2[c1,i,j]*S2[c1,i,l]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/2 \
                                - (9*S2[c1,i,j]*S3[c1,i,l,l]*S2[c1,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/2 \
                                - 6*S2[c1,i,l]*S3[c1,i,j,l]*S2[c1,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                - (3*S2[c1,i,l]*S3[c1,i,l,l]*S2[c1,j,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/2 \
                                - 6*S3[c1,i,j,l]*S3[c1,i,l,l]*S1[c1,j]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/6 \
                                + (3*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (9*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (11*S3[c1,i,i,i]*S3[c1,i,j,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (7*S3[c1,i,i,i]*S3[c1,i,j,l]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (S3[c1,i,i,i]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/6 \
                                + (S3[c1,i,i,i]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,j,j,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + (S3[c1,i,i,i]*S3[c1,i,l,l]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/2 \
                                + S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,k,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j] \
                                + (17*S3[c1,i,i,j]*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/16 \
                                + (31*S3[c1,i,i,j]*S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/48 \
                                + (S3[c1,i,i,k]*S3[c1,i,i,l]*S3[c1,j,j,j]*S3[c1,j,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/4 \
                                + (3*S3[c1,i,i,k]*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,j,j,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/8 \
                                + (7*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/8 \
                                + (7*S3[c1,i,i,j]*S3[c1,i,k,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (15*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/8 \
                                + (15*S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (3*S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (3*S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + 3*S3[c1,i,i,k]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k] \
                                + (3*S3[c1,i,i,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/6 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/18 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/6 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/9 \
                                + (15*S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (3*S3[c1,i,i,l]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,l]*S3[c1,i,k,k]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (3*S3[c1,i,i,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,j,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                + (S3[c1,i,i,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (S3[c1,i,i,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (S3[c1,i,i,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/18 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/72 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/36 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/36 \
                                + (S3[c1,i,i,l]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (S3[c1,i,i,l]*S3[c1,i,k,k]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                + (S3[c1,i,i,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (3*S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + (3*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + (3*S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/4 \
                                - (3*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,l,l]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/2 \
                                - (3*S2[c1,i,j]*S3[c1,i,i,l]*S3[c1,j,j,l]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/2 \
                                - 6*S3[c1,i,i,j]*S3[c1,i,j,l]*S2[c1,j,l]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l] \
                                - (3*S3[c1,i,i,j]*S3[c1,i,l,l]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/2 \
                                - 3*S3[c1,i,j,j]*S3[c1,i,i,l]*S2[c1,j,l]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l] \
                                - 3*S3[c1,i,i,l]*S3[c1,i,j,l]*S2[c1,j,j]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l] \
                                + 6*S2[c1,i,j]*S2[c1,i,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 12*S2[c1,i,j]*S2[c1,i,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (3*S2[c1,i,j]*S2[c1,i,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 6*S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                - (3*S2[c1,i,j]*S3[c1,i,l,l]*S2[c1,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 3*S2[c1,i,k]*S2[c1,i,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S2[c1,i,k]*S2[c1,i,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S2[c1,i,k]*S3[c1,i,j,k]*S2[c1,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (9*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + 3*S2[c1,i,k]*S3[c1,i,j,l]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (9*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + 3*S2[c1,i,l]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 6*S2[c1,i,l]*S3[c1,i,j,k]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (27*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (9*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + 3*S2[c1,i,l]*S3[c1,i,j,l]*S2[c1,j,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (27*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,j,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (9*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,j,k,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (9*S2[c1,i,l]*S3[c1,i,k,k]*S3[c1,j,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (9*S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                - (S2[c1,i,l]*S3[c1,i,l,l]*S2[c1,j,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (14*S3[c1,i,j,k]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (9*S3[c1,i,j,k]*S3[c1,i,k,l]*S2[c1,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (5*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (8*S3[c1,i,j,k]*S3[c1,i,l,l]*S1[c1,j]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (7*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (10*S3[c1,i,j,l]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (9*S3[c1,i,j,l]*S3[c1,i,k,k]*S2[c1,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,l,l]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (22*S3[c1,i,j,l]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 8*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                - (2*S3[c1,i,j,l]*S3[c1,i,l,l]*S1[c1,j]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (5*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,l]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (11*S1[c1,i]*S3[c1,i,j,k]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/6 \
                                + 5*S1[c1,i]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (3*S1[c1,i]*S3[c1,i,j,l]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + S1[c1,i]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (5*S1[c1,i]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/6 \
                                + (S1[c1,i]*S3[c1,i,l,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 3*S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (3*S2[c1,i,j]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 3*S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                - (3*S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/4 \
                                + 3*S2[c1,i,l]*S3[c1,i,j,l]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (5*S3[c1,i,j,k]*S3[c1,i,l,l]*S1[c1,j]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/6 \
                                + 6*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,l,l]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + S3[c1,i,j,l]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 12*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (S3[c1,i,j,l]*S3[c1,i,l,l]*S1[c1,j]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 9*S3[c1,i,j,l]*S3[c1,i,l,l]*S2[c1,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                - 3*S3[c1,i,j,l]*S3[c1,i,l,l]*S2[c1,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 2*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 3*S3[c1,i,k,l]*S3[c1,i,l,l]*S2[c1,j,j]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 2*S1[c1,i]*S3[c1,i,j,j]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S1[c1,i]*S3[c1,i,j,k]*S3[c1,j,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S3[c1,i,k,l]*S3[c1,j,j,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,i]*S1[c1,j]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S2[c1,i,i]*S1[c1,j]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,i]*S1[c1,j]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S2[c1,i,i]*S2[c1,j,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + 3*S2[c1,i,j]*S2[c1,i,k]*S3[c1,j,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S2[c1,i,j]*S3[c1,i,k,l]*S2[c1,j,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S2[c1,i,k]*S3[c1,i,k,l]*S2[c1,j,j]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (3*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/4 \
                                + (4*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (5*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/36 \
                                + (5*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/4 \
                                + (11*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/12 \
                                + (13*S2[c1,i,i]*S3[c1,i,j,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/12 \
                                + (S2[c1,i,i]*S3[c1,i,j,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/4 \
                                + (S2[c1,i,i]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/12 \
                                + (S2[c1,i,i]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (S2[c1,i,i]*S3[c1,i,l,l]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/36 \
                                + (4*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (7*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                - (5*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                + (5*S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/2 \
                                + (5*S2[c1,i,j]*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                - (S2[c1,i,j]*S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (2*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S2[c1,i,k]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (S2[c1,i,k]*S3[c1,i,i,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + S2[c1,i,l]*S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (2*S2[c1,i,l]*S3[c1,i,i,j]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/9 \
                                + (S2[c1,i,l]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (S2[c1,i,l]*S3[c1,i,i,l]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                + (2*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/9 \
                                + (7*S3[c1,i,i,i]*S2[c1,j,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/9 \
                                + (7*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (7*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (8*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,l]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,l,l]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (5*S3[c1,i,i,j]*S3[c1,i,j,l]*S2[c1,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                - (31*S3[c1,i,i,j]*S3[c1,i,j,l]*S2[c1,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                + (4*S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,j,k,k]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,j,k,l]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/12 \
                                + (S3[c1,i,i,j]*S3[c1,i,k,l]*S2[c1,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (2*S3[c1,i,i,j]*S3[c1,i,k,l]*S3[c1,j,j,k]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                - (17*S3[c1,i,i,j]*S3[c1,i,l,l]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/36 \
                                + (S3[c1,i,i,j]*S3[c1,i,l,l]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,l]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (5*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/12 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,l]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,l]*S2[c1,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/4 \
                                + (2*S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,j,j,k]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,j,j]*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                - (8*S3[c1,i,j,j]*S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/9 \
                                + (2*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,j,k,k]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + (S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,j,k,l]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/2 \
                                + (5*S3[c1,i,i,l]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/12 \
                                + S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,j,k]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k] \
                                + (S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,j,l]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                - (11*S3[c1,i,i,l]*S3[c1,i,j,l]*S2[c1,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/12 \
                                + (S3[c1,i,i,l]*S3[c1,i,j,l]*S3[c1,j,j,k]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                + 3*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 6*S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 9*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (9*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (9*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 9*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S2[c1,i,k]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S2[c1,i,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S2[c1,i,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S2[c1,i,k]*S3[c1,i,l,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (9*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 9*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (9*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S2[c1,i,l]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S2[c1,i,l]*S3[c1,i,k,k]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S2[c1,i,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S3[c1,i,j,k]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S3[c1,i,j,k]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 12*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,k]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 6*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 6*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 6*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,j,l]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 2*S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                - (S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/2 \
                                - 2*S3[c1,i,j,l]*S3[c1,i,l,l]*S2[c1,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[l] \
                                + S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + 2*S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (3*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S2[c1,i,k]*S3[c1,i,l,l]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (11*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + 5*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (3*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (5*S2[c1,i,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (S2[c1,i,l]*S3[c1,i,l,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (5*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (S3[c1,i,j,l]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,j,l]*S3[c1,i,l,l]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + 2*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                - (3*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + 2*S1[c1,i]*S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S1[c1,i]*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (S2[c1,i,i]*S3[c1,i,j,j]*S1[c1,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + (3*S2[c1,i,i]*S3[c1,i,j,k]*S2[c1,j,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 4*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,j,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,i]*S3[c1,i,j,l]*S3[c1,j,k,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,i]*S3[c1,i,k,l]*S3[c1,j,j,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,j]*S3[c1,i,i,j]*S1[c1,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,j]*S3[c1,i,i,k]*S2[c1,j,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,j]*S3[c1,i,i,k]*S3[c1,j,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,j]*S3[c1,i,i,l]*S3[c1,j,k,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + S2[c1,i,k]*S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S2[c1,i,k]*S3[c1,i,i,j]*S3[c1,j,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (S2[c1,i,k]*S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 2*S2[c1,i,k]*S3[c1,i,i,l]*S3[c1,j,j,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 4*S2[c1,i,l]*S3[c1,i,i,j]*S3[c1,j,k,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 2*S2[c1,i,l]*S3[c1,i,i,k]*S3[c1,j,j,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (2*S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                + (2*S3[c1,i,i,i]*S2[c1,j,k]*S2[c1,j,l]*S2[c1,k,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                + (2*S3[c1,i,i,i]*S2[c1,j,l]*S3[c1,j,k,l]*S1[c1,k]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                + (3*S3[c1,i,i,j]*S3[c1,i,k,k]*S1[c1,j]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 3*S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S3[c1,i,i,j]*S3[c1,i,k,l]*S2[c1,j,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S3[c1,i,i,k]*S3[c1,i,j,k]*S1[c1,j]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 3*S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + 3*S3[c1,i,i,l]*S3[c1,i,j,k]*S2[c1,j,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                - (3*S2[c1,i,i]*S2[c1,j,l]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/2 \
                                - (2*S2[c1,i,j]*S2[c1,i,k]*S2[c1,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/3 \
                                - 3*S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/9 \
                                + (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/6 \
                                + (S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/18 \
                                + (S3[c1,i,i,i]*S3[c1,i,j,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/9 \
                                + (S3[c1,i,i,j]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/18 \
                                + (S3[c1,i,i,j]*S3[c1,i,i,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx02[k]*mx04[i]*mx02[l]*mx04[j])/6 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/6 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/18 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/6 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/9 \
                                + (2*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (2*S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (4*S3[c1,i,i,j]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (4*S3[c1,i,i,j]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/9 \
                                + (2*S3[c1,i,i,j]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/9 \
                                + (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/6 \
                                + (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/18 \
                                + (3*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (S3[c1,i,i,k]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/6 \
                                + (S3[c1,i,i,k]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (S3[c1,i,i,k]*S3[c1,i,l,l]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/18 \
                                + (5*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/6 \
                                + (5*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/9 \
                                + (3*S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (S3[c1,i,i,l]*S3[c1,i,j,l]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/2 \
                                + (S3[c1,i,i,l]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/6 \
                                + (S3[c1,i,i,l]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/9 \
                                + (S3[c1,i,i,i]*S3[c1,j,k,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/24 \
                                + (31*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/36 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/72 \
                                + (11*S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/36 \
                                + (5*S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/9 \
                                + (S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/9 \
                                + (2*S3[c1,i,i,j]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/9 \
                                + (S3[c1,i,i,j]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/9 \
                                + (S3[c1,i,i,j]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/18 \
                                + (29*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/72 \
                                + (5*S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/72 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                + (S3[c1,i,i,k]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/18 \
                                + (S3[c1,i,i,k]*S3[c1,i,l,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/72 \
                                + (5*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/36 \
                                + (19*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/72 \
                                - (S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,l,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/24 \
                                + (S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                + (S3[c1,i,i,l]*S3[c1,i,j,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                + (S3[c1,i,i,l]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/36 \
                                + (S3[c1,i,i,l]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/36 \
                                + 3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + 3*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + 6*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (3*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + 3*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + 3*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + 2*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + (S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/6 \
                                + (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/3 \
                                - (S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx05[l])/2 \
                                + (S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/2 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/4 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/2 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/4 \
                                - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,l,l]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/2 \
                                - S3[c1,i,i,i]*S2[c1,j,l]*S3[c1,j,j,l]*S2[c1,k,k]*mx03[i]*mx02[k]*mx03[j]*mx02[l] \
                                + 3*S2[c1,i,i]*S2[c1,j,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 6*S2[c1,i,i]*S2[c1,j,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (S2[c1,i,i]*S2[c1,j,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 3*S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S2[c1,i,j]*S3[c1,i,j,l]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + S2[c1,i,k]*S3[c1,i,j,j]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (9*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (9*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S2[c1,i,k]*S3[c1,i,k,l]*S3[c1,j,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + S2[c1,i,l]*S3[c1,i,j,j]*S2[c1,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S2[c1,i,l]*S3[c1,i,j,j]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (9*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,l]*S3[c1,i,k,k]*S3[c1,j,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (7*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (4*S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (9*S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (10*S3[c1,i,j,j]*S3[c1,i,k,l]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/9 \
                                + (7*S3[c1,i,j,j]*S3[c1,i,l,l]*S1[c1,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/9 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S2[c1,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (5*S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (11*S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                - (S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (9*S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,k,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S1[c1,k]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (10*S3[c1,i,j,k]*S3[c1,i,j,l]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (2*S3[c1,i,j,k]*S3[c1,i,k,k]*S1[c1,j]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,j,k]*S3[c1,i,k,l]*S2[c1,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 4*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (3*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,k]*S2[c1,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (5*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (7*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (4*S1[c1,i]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (2*S1[c1,i]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S1[c1,i]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (3*S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/4 \
                                + 3*S2[c1,i,j]*S3[c1,i,j,l]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + S2[c1,i,j]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + S2[c1,i,k]*S3[c1,i,j,l]*S2[c1,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + S2[c1,i,l]*S3[c1,i,j,k]*S2[c1,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 3*S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + 2*S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (3*S3[c1,i,i,l]*S2[c1,j,k]*S2[c1,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (9*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 6*S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                - (3*S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,l,l]*S1[c1,k]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (S3[c1,i,j,k]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (2*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + 3*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (8*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + 6*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 4*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (4*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (2*S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + S1[c1,i]*S3[c1,i,j,j]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S2[c1,i,i]*S2[c1,j,j]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                - 6*S2[c1,i,j]*S3[c1,i,j,k]*S1[c1,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S1[c1,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/3 \
                                + (7*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/36 \
                                + (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/3 \
                                - (5*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/36 \
                                + (5*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                + (S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (5*S3[c1,i,i,i]*S2[c1,j,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                - (5*S3[c1,i,i,i]*S2[c1,j,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                + (5*S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,l]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/9 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,l,l]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                + (2*S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,j,k,k]*S2[c1,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/9 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,j,k,l]*S2[c1,k,k]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/6 \
                                + (S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/4 \
                                + (S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/2 \
                                + 3*S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + S2[c1,i,i]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S2[c1,i,j]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 2*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S2[c1,i,k]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (2*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/3 \
                                + 3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 2*S3[c1,i,i,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S2[c1,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S2[c1,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/3 \
                                + (3*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 4*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S2[c1,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 6*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,j,k]*S3[c1,i,k,k]*S2[c1,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 3*S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 3*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (3*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + S3[c1,i,k,k]*S3[c1,i,k,l]*S3[c1,j,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                - (3*S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/2 \
                                - 3*S2[c1,i,i]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l] \
                                - S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[k]*mx04[l] \
                                + (3*S2[c1,i,i]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (S2[c1,i,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (S2[c1,i,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (4*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (2*S2[c1,i,l]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S2[c1,i,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + 2*S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,l,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (S3[c1,i,j,k]*S3[c1,i,k,l]*S2[c1,j,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (4*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + 2*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l] \
                                + (2*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,k,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (S2[c1,i,j]*S3[c1,i,i,j]*S2[c1,k,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 2*S2[c1,i,j]*S3[c1,i,i,j]*S3[c1,k,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (S3[c1,i,i,i]*S1[c1,j]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                + (2*S3[c1,i,i,i]*S2[c1,j,k]*S3[c1,j,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                + (2*S3[c1,i,i,i]*S2[c1,j,l]*S3[c1,j,k,k]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                + (9*S3[c1,i,i,j]*S3[c1,i,j,k]*S1[c1,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 9*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S3[c1,i,i,k]*S3[c1,i,j,j]*S1[c1,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/2 \
                                + 3*S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l] \
                                + (3*S2[c1,i,i]*S2[c1,j,j]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/2 \
                                - 2*S2[c1,i,j]*S3[c1,i,j,k]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                - 3*S2[c1,i,j]*S3[c1,i,j,l]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                - S3[c1,i,j,j]*S3[c1,i,l,l]*S2[c1,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                + (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (2*S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/9 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,j,k,k]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/3 \
                                + (2*S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,j,k,l]*S3[c1,k,k,k]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/9 \
                                + (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/4 \
                                + (11*S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/72 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/36 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,j,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/18 \
                                + (7*S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/72 \
                                + (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/16 \
                                - (5*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/24 \
                                - (S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/24 \
                                - (S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/12 \
                                - (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/8 \
                                - (S3[c1,i,i,i]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx03[k]*mx04[l])/2 \
                                - S3[c1,i,i,i]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[j]*mx03[i]*mx03[k]*mx04[l] \
                                + 2*S3[c1,i,i,k]*S3[c1,j,k,k]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + S3[c1,i,i,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (2*S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/3 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/3 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/6 \
                                + 2*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/3 \
                                + S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (S3[c1,i,j,k]*S3[c1,i,k,k]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/2 \
                                + (7*S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/8 \
                                + (S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/9 \
                                + (4*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/9 \
                                + (2*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/3 \
                                + (2*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/9 \
                                + (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/9 \
                                - (3*S3[c1,i,i,j]*S3[c1,i,j,j]*S2[c1,k,k]*S2[c1,l,l]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/4 \
                                + S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[l] \
                                + (3*S2[c1,i,i]*S2[c1,j,j]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (3*S2[c1,i,i]*S3[c1,j,j,k]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S2[c1,i,i]*S3[c1,j,j,l]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + 3*S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + 3*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (2*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,i,j]*S2[c1,j,l]*S2[c1,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,i,j]*S2[c1,j,l]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (43*S3[c1,i,i,j]*S3[c1,j,k,l]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/18 \
                                + (11*S3[c1,i,i,j]*S3[c1,j,l,l]*S1[c1,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/9 \
                                + (S3[c1,i,i,j]*S3[c1,j,l,l]*S2[c1,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,l]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (S3[c1,i,i,k]*S3[c1,j,k,k]*S3[c1,j,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,k]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/9 \
                                + (3*S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (2*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/9 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (8*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/9 \
                                - (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/9 \
                                + (3*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/4 \
                                + (4*S3[c1,i,i,l]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (7*S3[c1,i,i,l]*S3[c1,j,j,l]*S1[c1,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/6 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,l]*S2[c1,k,k]*S2[c1,k,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/8 \
                                + (5*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (8*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (S1[c1,i]*S3[c1,i,j,j]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S2[c1,i,i]*S2[c1,j,k]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + S2[c1,i,j]*S3[c1,i,j,k]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                - (S2[c1,i,j]*S3[c1,i,j,l]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 3*S3[c1,i,i,j]*S2[c1,j,l]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (2*S3[c1,i,i,j]*S3[c1,j,l,l]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (S3[c1,i,i,k]*S2[c1,j,k]*S2[c1,j,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + 2*S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (2*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (3*S3[c1,i,i,l]*S2[c1,j,j]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 3*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S1[c1,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (2*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S3[c1,i,j,k]*S3[c1,i,j,l]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + 6*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (8*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + 2*S2[c1,i,i]*S2[c1,j,j]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                - 6*S3[c1,i,i,j]*S2[c1,j,k]*S1[c1,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (2*S3[c1,i,i,j]*S3[c1,j,k,l]*S1[c1,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l])/3 \
                                - 3*S3[c1,i,i,k]*S2[c1,j,j]*S1[c1,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l] \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/36 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,l]*S3[c1,k,k,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/18 \
                                - (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,k,k,k]*S2[c1,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/4 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,l]*S3[c1,k,l,l]*mx03[i]*mx02[k]*mx03[j]*mx03[l])/6 \
                                + (S2[c1,i,i]*S3[c1,j,j,j]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx03[j]*mx03[k]*mx03[l])/4 \
                                + (3*S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/3 \
                                + (2*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/9 \
                                + S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/3 \
                                + 2*S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 2*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + 4*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + S3[c1,i,i,j]*S3[c1,j,l,l]*S2[c1,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (2*S3[c1,i,i,j]*S3[c1,j,l,l]*S2[c1,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/3 \
                                + (S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + 2*S3[c1,i,i,k]*S3[c1,j,k,k]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (S3[c1,i,j,j]*S3[c1,i,k,k]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/6 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/3 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,l]*S2[c1,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S2[c1,k,l]*S3[c1,k,k,k]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/2 \
                                + S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                - (3*S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/2 \
                                - S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l] \
                                - (S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/3 \
                                - (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/6 \
                                + (S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (2*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (2*S3[c1,i,i,j]*S3[c1,j,l,l]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                - (S3[c1,i,j,j]*S3[c1,i,l,l]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,i,l]*S2[c1,j,l]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S2[c1,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (4*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                - (S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/6 \
                                + (2*S3[c1,i,i,i]*S3[c1,j,j,k]*S1[c1,k]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                + (4*S3[c1,i,i,i]*S3[c1,j,j,k]*S2[c1,k,l]*S1[c1,l]*mx02[j]*mx03[i]*mx02[k]*mx02[l])/3 \
                                + (S2[c1,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[k]*mx03[j]*mx02[l])/4 \
                                + S2[c1,i,i]*S3[c1,j,j,j]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[k]*mx03[j]*mx02[l] \
                                - (S2[c1,i,i]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/12 \
                                + (3*S2[c1,i,i]*S2[c1,j,j]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/4 \
                                - S3[c1,i,i,j]*S1[c1,j]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                - 2*S3[c1,i,i,j]*S2[c1,j,k]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                - 3*S3[c1,i,i,j]*S2[c1,j,l]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                - 2*S3[c1,i,i,j]*S3[c1,j,l,l]*S2[c1,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                - S3[c1,i,i,k]*S2[c1,j,j]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l] \
                                - (3*S3[c1,i,i,l]*S2[c1,j,j]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/2 \
                                - (3*S3[c1,i,i,l]*S3[c1,j,j,l]*S2[c1,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l])/2 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx02[l]*mx04[k])/36 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/16 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/24 \
                                - (S3[c1,i,i,j]*S3[c1,i,j,j]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx03[i]*mx03[j]*mx03[k]*mx03[l])/48 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx03[k]*mx04[l])/2 \
                                - (S3[c1,i,i,i]*S3[c1,j,k,k]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[j]*mx03[i]*mx03[k]*mx04[l])/3 \
                                + S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k] \
                                + (2*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/3 \
                                + (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/3 \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/4 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/4 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/18 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,k,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/4 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/4 \
                                - (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx05[l])/18 \
                                + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/6 \
                                + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/9 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/18 \
                                + (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/9 \
                                + (4*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/9 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*S2[c1,l,l]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/8 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,k,l]*S1[c1,l]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/6 \
                                + (S3[c1,i,i,i]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[l])/6 \
                                + (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[l])/2 \
                                - (S2[c1,i,i]*S2[c1,j,j]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/24 \
                                + (3*S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                - (S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/6 \
                                + (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                - (2*S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (7*S3[c1,i,i,j]*S3[c1,j,k,k]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/18 \
                                + (3*S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (7*S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/9 \
                                + S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l] \
                                + (29*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/18 \
                                - (7*S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/18 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/6 \
                                + (9*S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/8 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/3 \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/8 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/6 \
                                - (S3[c1,i,i,l]*S2[c1,j,j]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/2 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/8 \
                                + (7*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/6 \
                                - (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*S1[c1,l]*mx02[i]*mx02[j]*mx03[k]*mx03[l])/6 \
                                + (S2[c1,i,i]*S2[c1,j,j]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/4 \
                                + (S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/6 \
                                + (3*S3[c1,i,i,j]*S2[c1,j,k]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (3*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                - (S3[c1,i,i,j]*S2[c1,j,l]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (S3[c1,i,i,j]*S3[c1,j,k,l]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + 4*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l] \
                                + (7*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (2*S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S3[c1,i,i,k]*S2[c1,j,j]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                - (S3[c1,i,i,l]*S2[c1,j,j]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/4 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/6 \
                                + (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                + (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/6 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/2 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,k,k]*S2[c1,l,l]*mx03[i]*mx03[j]*mx02[l]*mx03[k])/27 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S2[c1,k,k]*S3[c1,l,l,l]*mx03[i]*mx02[k]*mx03[j]*mx03[l])/36 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,k,l]*S2[c1,l,l]*mx03[i]*mx02[k]*mx03[j]*mx03[l])/12 \
                                + (S2[c1,i,i]*S3[c1,j,j,j]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx03[j]*mx03[k]*mx03[l])/108 \
                                + (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[j]*mx03[i]*mx02[k]*mx04[l])/6 \
                                + (S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/6 \
                                + (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/9 \
                                + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/3 \
                                + S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k] \
                                + (2*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/3 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + (3*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[l]*mx04[k])/4 \
                                - (S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/2 \
                                - (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/6 \
                                - (S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/3 \
                                - (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/3 \
                                - (S3[c1,i,i,l]*S2[c1,j,j]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/6 \
                                - (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx03[k]*mx04[l])/4 \
                                + (S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/4 \
                                + (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/2 \
                                + (7*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                - (S3[c1,i,i,j]*S3[c1,j,l,l]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/3 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/4 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/12 \
                                - (S3[c1,i,i,l]*S3[c1,j,j,l]*S2[c1,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/4 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/4 \
                                + (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx03[i]*mx02[k]*mx03[j]*mx04[l])/36 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,k]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[j]*mx03[i]*mx03[k]*mx04[l])/6 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[j]*mx03[i]*mx03[k]*mx04[l])/18 \
                                + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/9 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[l]*mx05[k])/12 \
                                - (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx05[l])/9 \
                                - (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx03[k]*mx05[l])/12 \
                                + (7*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/18 \
                                + (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/9 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/12 \
                                + (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/36 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx06[l])/12 \
                                + (S3[c1,i,i,j]*S3[c1,j,k,k]*S1[c1,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/3 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S1[c1,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx04[l])/6 \
                                + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/6 \
                                + (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx05[l])/12 \
                                - (S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,l,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/6 \
                                - (5*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/3 \
                                - (2*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/3 \
                                - 4*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l] \
                                - (5*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (3*S2[c1,i,i]*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (3*S2[c1,i,i]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,j,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (7*S2[c1,i,i]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (S2[c1,i,i]*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,j]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,i,l,l]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/6 \
                                - 2*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,j]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l] \
                                - (7*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,l,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (3*S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (5*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,l,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/9 \
                                - (43*S2[c1,i,i]*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,k,k,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/36 \
                                - (5*S2[c1,i,i]*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (3*S2[c1,i,i]*S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (S3[c1,i,i,j]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,l,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/12 \
                                - (4*S3[c1,i,i,j]*S3[c1,i,k,l]*S2[c1,j,j]*S3[c1,j,k,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/3 \
                                - (7*S3[c1,i,i,j]*S3[c1,i,l,l]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/12 \
                                - (S3[c1,i,i,k]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,j,l,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/4 \
                                - (5*S3[c1,i,i,k]*S3[c1,i,j,l]*S2[c1,j,j]*S3[c1,j,k,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/4 \
                                - (S3[c1,i,i,k]*S3[c1,i,k,l]*S2[c1,j,j]*S3[c1,j,j,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/4 \
                                - (S3[c1,i,i,k]*S3[c1,i,l,l]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/4 \
                                - (7*S3[c1,i,i,l]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,j,k,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/4 \
                                - (5*S3[c1,i,i,l]*S3[c1,i,j,l]*S2[c1,j,j]*S3[c1,j,k,k]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/4 \
                                - (S3[c1,i,i,l]*S3[c1,i,k,k]*S2[c1,j,j]*S3[c1,j,j,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/4 \
                                - (3*S3[c1,i,i,l]*S3[c1,i,k,l]*S2[c1,j,j]*S3[c1,j,j,k]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/4 \
                                - (7*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/18 \
                                - (11*S2[c1,i,i]*S3[c1,i,i,j]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/18 \
                                - (S2[c1,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/18 \
                                - (7*S2[c1,i,i]*S3[c1,i,i,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx02[i]**2*mx02[k]*mx03[j]*mx02[l])/36 \
                                - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/6 \
                                - (4*S3[c1,i,i,j]*S3[c1,i,j,k]*S2[c1,j,j]*S3[c1,k,l,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/9 \
                                - (29*S3[c1,i,i,j]*S3[c1,i,j,l]*S2[c1,j,j]*S3[c1,k,k,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/36 \
                                - (S3[c1,i,i,k]*S3[c1,i,j,j]*S2[c1,j,j]*S3[c1,k,l,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/9 \
                                - (7*S3[c1,i,j,j]*S3[c1,i,i,l]*S2[c1,j,j]*S3[c1,k,k,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/18 \
                                - 3*S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2 \
                                - 6*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2 \
                                - (S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/9 \
                                - (5*S3[c1,i,i,i]*S2[c1,j,j]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[j]**2*mx03[i]*mx02[k]*mx02[l])/36 \
                                - (S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/9 \
                                - (4*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/9 \
                                - (2*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/3 \
                                - (2*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/9 \
                                - (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/9 \
                                - (3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2)/2 \
                                - 2*S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2 \
                                - 3*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2 \
                                - 6*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[l]**2*mx03[k])/6 \
                                - (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/6 \
                                - (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/9 \
                                - (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/18 \
                                - (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/9 \
                                - (4*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/9 \
                                - (3*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2)/2 \
                                - 4*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2 \
                                - (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2)/2 \
                                - (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]**2)/2 \
                                + (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[l]**2*mx03[k])/3 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[l]**2*mx03[k])/4 \
                                - (7*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/18 \
                                - (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/9 \
                                - (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/12 \
                                - (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/36 \
                                - (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]**2)/12 \
                                - (7*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,k,l]**2*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/18 \
                                - (S3[c1,i,i,i]*S3[c1,i,k,l]**2*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/18 \
                                - (S3[c1,i,i,i]*S3[c1,i,j,l]**2*S3[c1,j,k,k]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/2 \
                                - (S3[c1,i,i,i]*S3[c1,i,j,k]**2*S3[c1,j,l,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/6 \
                                - (S3[c1,i,j,j]*S3[c1,i,k,l]**2*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/9 \
                                - (S3[c1,i,j,l]**2*S3[c1,i,k,k]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/6 \
                                - (S3[c1,i,j,k]**2*S3[c1,i,l,l]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/6 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,k,l]**2*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/9 \
                                - S3[c1,i,i,l]*S3[c1,j,k,l]**2*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l] \
                                - (S3[c1,i,j,l]**2*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/2 \
                                + (S3[c1,i,j,l]**2*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[l]*mx03[k]*mx03[l])/6 \
                                - (S3[c1,i,j,k]**2*S2[c1,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/6 \
                                - (S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,j,l,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/18 \
                                - (5*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/9 \
                                - (2*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,j,k,k]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/9 \
                                - (4*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/3 \
                                - (5*S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,j,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,i,i]*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,j,k]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (S3[c1,i,i,i]*S3[c1,i,j,l]*S3[c1,i,k,k]*S3[c1,j,j,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (7*S3[c1,i,i,i]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,j,k]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,i,i]*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,j]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/36 \
                                - (S3[c1,i,j,j]*S3[c1,i,k,k]*S3[c1,i,l,l]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/18 \
                                - (2*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,j,j]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/3 \
                                - (7*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/36 \
                                - (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,j,l,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,j,k,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (5*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,j,k]*S3[c1,k,l,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/27 \
                                - (43*S3[c1,i,i,i]*S3[c1,i,j,j]*S3[c1,i,j,l]*S3[c1,k,k,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/108 \
                                - (5*S3[c1,i,i,i]*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,j,k,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,i,i]*S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,j,k,k]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (S3[c1,i,i,j]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/36 \
                                - (4*S3[c1,i,i,j]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/9 \
                                - (7*S3[c1,i,i,j]*S3[c1,i,l,l]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/36 \
                                - (S3[c1,i,i,k]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,j,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (5*S3[c1,i,i,k]*S3[c1,i,j,l]*S3[c1,j,j,j]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,i,k]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,j,j,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,i,k]*S3[c1,i,l,l]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (7*S3[c1,i,i,l]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,j,k,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (5*S3[c1,i,i,l]*S3[c1,i,j,l]*S3[c1,j,j,j]*S3[c1,j,k,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,i,l]*S3[c1,i,k,k]*S3[c1,j,j,j]*S3[c1,j,j,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/12 \
                                - (S3[c1,i,i,l]*S3[c1,i,k,l]*S3[c1,j,j,j]*S3[c1,j,j,k]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/4 \
                                - (7*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/54 \
                                - (11*S3[c1,i,i,i]*S3[c1,i,i,j]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/54 \
                                - (S3[c1,i,i,i]*S3[c1,i,i,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/54 \
                                - (7*S3[c1,i,i,i]*S3[c1,i,i,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx02[i]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/108 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/18 \
                                - (4*S3[c1,i,i,j]*S3[c1,i,j,k]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/27 \
                                - (29*S3[c1,i,i,j]*S3[c1,i,j,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/108 \
                                - (S3[c1,i,i,k]*S3[c1,i,j,j]*S3[c1,j,j,j]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/27 \
                                - (7*S3[c1,i,j,j]*S3[c1,i,i,l]*S3[c1,j,j,j]*S3[c1,k,k,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/54 \
                                - (S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - S3[c1,i,j,k]*S3[c1,i,l,l]*S2[c1,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l] \
                                - (4*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - 2*S3[c1,i,j,l]*S3[c1,i,k,l]*S2[c1,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l] \
                                - 2*S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l] \
                                - (2*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - (S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/27 \
                                - (5*S3[c1,i,i,i]*S3[c1,j,j,j]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[j]*mx03[i]*mx02[k]*mx03[j]*mx02[l])/108 \
                                - (S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/2 \
                                - (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/2 \
                                - (2*S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/6 \
                                - S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l] \
                                - (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - 2*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l] \
                                - (4*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[l]*mx03[k]*mx03[l])/18 \
                                - (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/2 \
                                - (4*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - (7*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/6 \
                                - (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/3 \
                                - (S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/2 \
                                - (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/4 \
                                - (S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/2 \
                                - (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/12 \
                                - (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/4 \
                                + (S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[l]*mx03[k]*mx03[l])/9 \
                                + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[l]*mx03[k]*mx03[l])/12 \
                                - (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/6 \
                                - (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,l]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[l])/12 
                            for m in range(n):
                                if i!=j and i!=k and i!=l and i!=m and j!=k and j!=l and j!=m and k!=l and k!=m and l!=m:
                                    mu4Y3ijklm[c1,i,j,k,l,m] = (S3[c1,i,m,m]**2*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (3*S3[c1,i,j,m]**2*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,k]**2*S2[c1,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (S3[c1,i,j,k]**2*S3[c1,l,m,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (2*S2[c1,i,m]*S3[c1,i,m,m]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (3*S3[c1,i,l,l]*S3[c1,i,l,m]*S3[c1,j,k,m]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,l,m]*S3[c1,i,m,m]*S3[c1,j,k,l]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (4*S1[c1,i]*S3[c1,i,j,j]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (4*S2[c1,i,m]*S3[c1,i,j,j]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,l,m]**2*S3[c1,j,k,k]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (3*S3[c1,i,l,m]**2*S3[c1,j,k,l]*S3[c1,j,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (2*S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (2*S3[c1,i,i,j]*S2[c1,j,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (5*S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,i,k]*S3[c1,j,l,m]**2*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,k]*S3[c1,j,m,m]**2*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,i,l]*S3[c1,j,l,m]**2*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,i,l]*S3[c1,j,m,m]**2*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/24 \
                                    + (3*S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (3*S3[c1,i,i,m]*S3[c1,j,l,m]**2*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,l,m]**2*S3[c1,j,j,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (S3[c1,i,l,m]**2*S3[c1,j,j,l]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,l,m]**2*S3[c1,j,j,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,m,m]**2*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/24 \
                                    + (S3[c1,i,m,m]**2*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/48 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/4 \
                                    + (S2[c1,i,i]*S3[c1,j,k,m]**2*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (2*S3[c1,i,j,m]**2*S2[c1,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,m,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,m,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (11*S3[c1,i,i,l]*S3[c1,j,k,m]**2*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    - (S3[c1,i,i,m]*S3[c1,j,k,m]**2*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/12 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,m]**2*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (13*S3[c1,i,i,m]*S3[c1,j,k,l]**2*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,m]**2*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (9*S3[c1,i,j,m]**2*S3[c1,k,l,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (3*S3[c1,i,j,l]**2*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,j,m]**2*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S2[c1,i,i]*S3[c1,j,k,l]**2*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,l]**2*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,m]**2*S3[c1,k,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (S3[c1,i,j,m]**2*S3[c1,k,k,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,l]**2*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/12 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,l]**2*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (5*S3[c1,i,j,m]**2*S3[c1,k,k,l]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/12 \
                                    - (S3[c1,i,j,m]**2*S3[c1,k,k,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (S3[c1,i,j,l]**2*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/12 \
                                    + (S3[c1,i,j,m]**2*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,m]**2*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (2*S3[c1,i,j,k]**2*S1[c1,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,j,k]**2*S2[c1,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,l]**2*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (S3[c1,i,j,k]**2*S3[c1,l,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,j,k]**2*S3[c1,l,l,m]*S1[c1,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    - (S3[c1,i,j,k]**2*S2[c1,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,k]**2*S3[c1,l,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,j,k]**2*S3[c1,l,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/18 \
                                    + (2*S2[c1,i,k]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,k]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,l]*S3[c1,i,k,m]*S3[c1,j,k,l]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,l]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,j,k,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,m]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S2[c1,i,m]*S3[c1,i,k,m]*S3[c1,j,k,l]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S2[c1,i,m]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,j,k,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,k,l]*S3[c1,i,m,m]*S2[c1,j,m]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S3[c1,i,k,m]*S3[c1,i,l,m]*S2[c1,j,m]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,k,m]*S3[c1,i,m,m]*S2[c1,j,l]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,l,m]*S3[c1,i,m,m]*S2[c1,j,k]*S3[c1,j,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,k,m]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,k,l]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,k,l]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,k,l]*S3[c1,i,m,m]*S3[c1,j,k,m]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,k,m]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (3*S3[c1,i,k,m]*S3[c1,i,l,l]*S3[c1,j,k,m]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,k,m]*S3[c1,i,l,m]*S3[c1,j,k,m]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,k,m]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (S3[c1,i,l,l]*S3[c1,i,l,m]*S3[c1,j,k,k]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,l,l]*S3[c1,i,m,m]*S3[c1,j,k,k]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,l,l]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,j,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,l,m]*S3[c1,i,m,m]*S3[c1,j,k,k]*S3[c1,j,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,k,l]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,k,m]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,j,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,k,m]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,l,m]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,j,k,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S1[c1,i]*S3[c1,i,k,l]*S3[c1,j,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S1[c1,i]*S3[c1,i,k,m]*S3[c1,j,j,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S1[c1,i]*S3[c1,i,l,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S3[c1,i,j,k]*S3[c1,i,l,m]*S1[c1,j]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S3[c1,i,j,l]*S3[c1,i,k,m]*S1[c1,j]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S3[c1,i,j,m]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (2*S2[c1,i,j]*S3[c1,i,k,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,j]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,j]*S3[c1,i,l,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (5*S2[c1,i,j]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,j]*S3[c1,i,m,m]*S3[c1,j,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,j]*S3[c1,i,m,m]*S3[c1,j,l,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (3*S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S2[c1,i,k]*S3[c1,i,m,m]*S3[c1,j,j,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S2[c1,i,k]*S3[c1,i,m,m]*S3[c1,j,j,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (3*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (2*S2[c1,i,l]*S3[c1,i,j,m]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,l]*S3[c1,i,j,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S2[c1,i,l]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S2[c1,i,l]*S3[c1,i,m,m]*S3[c1,j,j,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (2*S2[c1,i,m]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,m]*S3[c1,i,j,l]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (13*S2[c1,i,m]*S3[c1,i,j,m]*S3[c1,j,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,m]*S3[c1,i,k,m]*S3[c1,j,j,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,m]*S3[c1,i,l,m]*S3[c1,j,j,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,m]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,m]*S3[c1,i,m,m]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,j,k]*S3[c1,i,m,m]*S2[c1,j,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,j,k]*S3[c1,i,m,m]*S2[c1,j,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,l]*S3[c1,i,m,m]*S2[c1,j,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,j,l]*S3[c1,i,m,m]*S2[c1,j,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (2*S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,k,l]*S2[c1,k,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,j,m]*S3[c1,i,k,m]*S2[c1,j,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,m]*S3[c1,i,l,m]*S2[c1,j,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,k,l]*S2[c1,k,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,m]*S3[c1,i,m,m]*S2[c1,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,m]*S3[c1,i,m,m]*S2[c1,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (2*S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,k,l]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (3*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (7*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (9*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,l,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (13*S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,l,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (7*S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,m,m]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (9*S3[c1,i,j,l]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (9*S3[c1,i,j,l]*S3[c1,i,l,m]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,j,l]*S3[c1,i,l,m]*S3[c1,j,l,m]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (5*S3[c1,i,j,l]*S3[c1,i,l,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,l,l]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (7*S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,l,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (5*S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (5*S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,l,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (11*S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,l,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (3*S3[c1,i,j,m]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,j,m]*S3[c1,i,l,l]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (5*S3[c1,i,j,m]*S3[c1,i,l,l]*S3[c1,j,l,m]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,j,m]*S3[c1,i,l,l]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (15*S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (15*S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (7*S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,l,l]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (9*S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (5*S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,l,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,k,l]*S3[c1,i,l,m]*S3[c1,j,j,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,k,l]*S3[c1,i,l,m]*S3[c1,j,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,k,l]*S3[c1,i,m,m]*S3[c1,j,j,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,k,l]*S3[c1,i,m,m]*S3[c1,j,j,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,k,m]*S3[c1,i,l,l]*S3[c1,j,j,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,k,m]*S3[c1,i,l,l]*S3[c1,j,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,k,m]*S3[c1,i,l,m]*S3[c1,j,j,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,k,m]*S3[c1,i,l,m]*S3[c1,j,j,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,k,m]*S3[c1,i,m,m]*S3[c1,j,j,l]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,l,l]*S3[c1,i,l,m]*S3[c1,j,j,k]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,l,l]*S3[c1,i,l,m]*S3[c1,j,j,m]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,l,l]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,l,l]*S3[c1,i,m,m]*S3[c1,j,j,l]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,l,l]*S3[c1,i,m,m]*S3[c1,j,j,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (5*S3[c1,i,l,m]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,l,m]*S3[c1,i,m,m]*S3[c1,j,j,l]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (3*S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (3*S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (3*S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (3*S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (5*S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,l,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,k,l]*S3[c1,i,m,m]*S3[c1,j,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,k,m]*S3[c1,i,l,m]*S3[c1,j,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,k,m]*S3[c1,i,m,m]*S3[c1,j,j,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,k,m]*S3[c1,i,m,m]*S3[c1,j,j,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,l,m]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,l,m]*S3[c1,i,m,m]*S3[c1,j,j,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (8*S1[c1,i]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (4*S1[c1,i]*S3[c1,i,j,k]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (4*S1[c1,i]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (2*S1[c1,i]*S3[c1,i,j,m]*S3[c1,j,k,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (2*S1[c1,i]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S1[c1,i]*S3[c1,i,k,m]*S3[c1,j,j,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (2*S3[c1,i,j,j]*S3[c1,i,l,m]*S1[c1,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,k,l]*S1[c1,j]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,j,k]*S3[c1,i,k,m]*S1[c1,j]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,i,m]*S1[c1,j]*S3[c1,j,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (2*S3[c1,i,j,l]*S3[c1,i,j,m]*S1[c1,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S2[c1,i,i]*S3[c1,j,k,l]*S3[c1,j,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S2[c1,i,i]*S3[c1,j,k,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,i]*S3[c1,j,k,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,i]*S3[c1,j,l,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,j]*S3[c1,i,j,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (5*S2[c1,i,j]*S3[c1,i,k,m]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,j]*S3[c1,i,l,m]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,j]*S3[c1,i,m,m]*S3[c1,j,k,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,k]*S3[c1,i,j,k]*S3[c1,j,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (5*S2[c1,i,k]*S3[c1,i,j,m]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,k]*S3[c1,i,l,m]*S3[c1,j,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,k]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (2*S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,l]*S3[c1,i,j,m]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,l]*S3[c1,i,k,m]*S3[c1,j,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S2[c1,i,m]*S3[c1,i,j,j]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (4*S2[c1,i,m]*S3[c1,i,j,k]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,m]*S3[c1,i,j,l]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,m]*S3[c1,i,j,m]*S3[c1,j,k,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,m]*S3[c1,i,k,l]*S3[c1,j,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,m]*S3[c1,i,k,m]*S3[c1,j,j,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,k]*S2[c1,j,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,j,j]*S3[c1,i,l,m]*S2[c1,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,j]*S3[c1,i,m,m]*S2[c1,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,j,j]*S3[c1,i,m,m]*S2[c1,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,l]*S2[c1,j,m]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,l]*S2[c1,j,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,m,m]*S2[c1,k,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,k]*S3[c1,i,k,l]*S2[c1,j,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,k]*S3[c1,i,k,m]*S2[c1,j,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,k]*S3[c1,i,k,m]*S2[c1,j,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (S3[c1,i,j,k]*S3[c1,i,k,m]*S3[c1,j,m,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,k]*S3[c1,i,l,m]*S2[c1,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,k]*S3[c1,i,m,m]*S2[c1,j,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (4*S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,k,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S2[c1,j,k]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S2[c1,j,l]*S3[c1,j,k,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S2[c1,j,l]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (5*S3[c1,i,i,m]*S2[c1,j,m]*S3[c1,j,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,i,m]*S2[c1,j,m]*S3[c1,j,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S2[c1,j,m]*S3[c1,j,l,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,l,m]*S2[c1,k,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,m,m]*S2[c1,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (2*S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,k,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (S3[c1,i,k,m]*S3[c1,i,m,m]*S3[c1,j,j,k]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,k]*S3[c1,j,l,l]*S3[c1,j,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,i,k]*S3[c1,j,l,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,i,k]*S3[c1,j,l,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,l,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (5*S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,m]*S3[c1,j,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (5*S3[c1,i,i,l]*S3[c1,j,k,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (3*S3[c1,i,i,l]*S3[c1,j,k,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,i,l]*S3[c1,j,l,l]*S3[c1,j,m,m]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,i,l]*S3[c1,j,l,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (17*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (11*S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (13*S3[c1,i,j,k]*S3[c1,i,k,m]*S3[c1,j,l,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (31*S3[c1,i,j,k]*S3[c1,i,k,m]*S3[c1,j,l,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    - (S3[c1,i,j,k]*S3[c1,i,k,m]*S3[c1,j,m,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (31*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (49*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (47*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (65*S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    - (11*S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,k,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (9*S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (5*S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (9*S3[c1,i,i,m]*S3[c1,j,k,m]*S3[c1,j,l,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (5*S3[c1,i,i,m]*S3[c1,j,k,m]*S3[c1,j,l,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (5*S3[c1,i,i,m]*S3[c1,j,l,l]*S3[c1,j,l,m]*S3[c1,k,k,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,i,m]*S3[c1,j,l,l]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/3 \
                                    + (13*S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/12 \
                                    + (11*S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/12 \
                                    + (17*S3[c1,i,j,l]*S3[c1,i,l,m]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (11*S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,k,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (13*S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/12 \
                                    + (11*S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/12 \
                                    + (5*S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,k,l]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/3 \
                                    - (S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,k,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/6 \
                                    + (13*S3[c1,i,j,m]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (31*S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,k,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    - (S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,k,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/18 \
                                    + (17*S3[c1,i,k,l]*S3[c1,i,l,m]*S3[c1,j,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (11*S3[c1,i,k,l]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (13*S3[c1,i,k,m]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (31*S3[c1,i,k,m]*S3[c1,i,l,m]*S3[c1,j,j,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    - (S3[c1,i,k,m]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (S3[c1,i,i,k]*S3[c1,j,l,m]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,m,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,m]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,m,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,j,k]*S3[c1,i,k,m]*S3[c1,j,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,j,k]*S3[c1,i,k,m]*S3[c1,j,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/3 \
                                    + (7*S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,j,k]*S3[c1,i,m,m]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/3 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (3*S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/3 \
                                    + (S3[c1,i,i,m]*S3[c1,j,l,m]*S3[c1,j,m,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,l]*S3[c1,i,m,m]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/3 \
                                    + (S3[c1,i,j,m]*S3[c1,i,m,m]*S3[c1,j,k,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,k,l]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,k,m]*S3[c1,i,l,m]*S3[c1,j,j,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,k,m]*S3[c1,i,m,m]*S3[c1,j,j,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (2*S1[c1,i]*S3[c1,i,j,j]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S1[c1,i]*S3[c1,i,j,j]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (4*S3[c1,i,i,j]*S3[c1,j,l,m]*S1[c1,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S3[c1,i,i,k]*S1[c1,j]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/6 \
                                    + (3*S3[c1,i,i,k]*S2[c1,j,k]*S3[c1,j,l,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (2*S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,m]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,m]*S2[c1,k,k]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (2*S3[c1,i,i,l]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (3*S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S1[c1,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/6 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,j,l]*S1[c1,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,j,k]*S3[c1,i,j,m]*S1[c1,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,m]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S3[c1,i,i,m]*S1[c1,j]*S3[c1,j,k,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (3*S3[c1,i,i,m]*S2[c1,j,k]*S3[c1,j,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,l]*S1[c1,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/6 \
                                    + (3*S3[c1,i,j,l]*S3[c1,i,j,m]*S2[c1,k,k]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,i]*S3[c1,j,k,k]*S3[c1,j,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S2[c1,i,i]*S3[c1,j,j,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S2[c1,i,i]*S3[c1,j,k,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (7*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (5*S2[c1,i,j]*S3[c1,i,j,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S2[c1,i,j]*S3[c1,i,j,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,l]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (5*S2[c1,i,m]*S3[c1,i,j,j]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (S2[c1,i,m]*S3[c1,i,j,j]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,i,j]*S2[c1,j,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (4*S3[c1,i,i,j]*S3[c1,j,l,m]*S2[c1,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,i,j]*S3[c1,j,m,m]*S2[c1,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (4*S3[c1,i,i,j]*S3[c1,j,m,m]*S2[c1,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (5*S3[c1,i,i,k]*S2[c1,j,m]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,k]*S2[c1,j,m]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,m,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (S3[c1,i,i,k]*S3[c1,j,k,m]*S3[c1,j,m,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (2*S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,m,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    - (S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,k,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,l]*S2[c1,j,m]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S2[c1,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,k]*S3[c1,i,j,m]*S2[c1,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,k]*S3[c1,i,j,m]*S2[c1,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (4*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,m,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (5*S3[c1,i,i,m]*S2[c1,j,k]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,m]*S2[c1,j,k]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,m]*S2[c1,j,l]*S3[c1,j,k,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S2[c1,j,m]*S3[c1,j,k,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,l]*S2[c1,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    - (S3[c1,i,i,m]*S3[c1,j,k,k]*S3[c1,j,m,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,m]*S2[c1,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,m]*S2[c1,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,k,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (7*S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/2 \
                                    + (9*S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,l,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (9*S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (3*S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,i,k]*S3[c1,j,k,m]*S3[c1,j,l,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (7*S3[c1,i,i,k]*S3[c1,j,k,m]*S3[c1,j,l,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    - (S3[c1,i,i,k]*S3[c1,j,k,m]*S3[c1,j,m,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/9 \
                                    + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (11*S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (5*S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/9 \
                                    - (S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,m,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (13*S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (11*S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (5*S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,k,l]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    - (S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,k,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (3*S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (17*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (11*S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (17*S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (53*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (5*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/18 \
                                    + (19*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/18 \
                                    + (139*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    - (11*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,m,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (11*S3[c1,i,j,k]*S3[c1,i,l,l]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (5*S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (13*S3[c1,i,i,m]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (31*S3[c1,i,i,m]*S3[c1,j,k,k]*S3[c1,j,l,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    - (S3[c1,i,i,m]*S3[c1,j,k,k]*S3[c1,j,m,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (9*S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,l,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (31*S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (13*S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (11*S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (S3[c1,i,j,l]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/6 \
                                    + (S3[c1,i,j,l]*S3[c1,i,l,l]*S3[c1,j,k,k]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/18 \
                                    + (S3[c1,i,k,l]*S3[c1,i,l,l]*S3[c1,j,j,k]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/2 \
                                    + (S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,m,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,i,k]*S3[c1,j,k,m]*S3[c1,j,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,i,k]*S3[c1,j,k,m]*S3[c1,j,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,m,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (5*S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,j,j]*S3[c1,i,m,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/24 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,m,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (3*S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,m,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/3 \
                                    + (17*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/3 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (3*S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,k]*S3[c1,j,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,i,m]*S3[c1,j,k,k]*S3[c1,j,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,l,l]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (3*S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/4 \
                                    + (2*S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (2*S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,i,j]*S1[c1,j]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/18 \
                                    + (3*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,l,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,l]*S1[c1,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,m]*S1[c1,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/6 \
                                    + (4*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,m]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/4 \
                                    + (3*S3[c1,i,i,k]*S3[c1,j,j,m]*S2[c1,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/4 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,m]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/6 \
                                    - (S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (2*S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,m]*S1[c1,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/4 \
                                    + (3*S3[c1,i,i,l]*S3[c1,j,j,m]*S2[c1,k,k]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/8 \
                                    - (3*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/2 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,m]*S1[c1,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/3 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,k]*S1[c1,k]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/18 \
                                    + (3*S3[c1,i,i,m]*S3[c1,j,j,k]*S2[c1,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/4 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*S1[c1,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/6 \
                                    + (3*S3[c1,i,i,m]*S3[c1,j,j,l]*S2[c1,k,k]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/8 \
                                    + (7*S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/4 \
                                    + (5*S2[c1,i,i]*S3[c1,j,j,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/12 \
                                    + (S2[c1,i,i]*S3[c1,j,j,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/4 \
                                    + (S2[c1,i,j]*S3[c1,i,j,k]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (2*S2[c1,i,k]*S3[c1,i,j,j]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (S2[c1,i,l]*S3[c1,i,j,j]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (2*S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (5*S3[c1,i,i,j]*S2[c1,j,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/18 \
                                    + (S3[c1,i,i,j]*S2[c1,j,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (5*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,m,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,m]*S2[c1,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,m]*S2[c1,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (4*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (7*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,m,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (2*S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    - (S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,k,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,k]*S2[c1,j,l]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    - (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,m,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/4 \
                                    + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (S3[c1,i,i,l]*S2[c1,j,k]*S3[c1,j,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,k,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,k]*S2[c1,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,k]*S2[c1,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    - (5*S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,m,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/12 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    - (S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,k,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/4 \
                                    + (S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (137*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (17*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (25*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (199*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    - (23*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,m,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (7*S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (55*S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (41*S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (17*S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,k,l]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    - (7*S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,k,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (5*S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/3 \
                                    - (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,m,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/16 \
                                    + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,l]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (29*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (7*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/36 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/48 \
                                    + (11*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (23*S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/72 \
                                    + (91*S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    - (5*S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,m,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (11*S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/48 \
                                    + (5*S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (5*S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,k,l]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    - (S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,k,m]*S3[c1,l,l,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/48 \
                                    + (5*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,m,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (13*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (7*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/24 \
                                    + (S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/6 \
                                    + (S3[c1,i,i,j]*S3[c1,j,m,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,m,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (5*S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (2*S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/27 \
                                    + (S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/27 \
                                    + (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,m,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/48 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,k,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/3 \
                                    + (2*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (19*S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/48 \
                                    + (5*S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,m,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/48 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/8 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,m]*S3[c1,k,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/16 \
                                    + (S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/9 \
                                    + (2*S3[c1,i,i,j]*S3[c1,j,k,k]*S1[c1,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (4*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,m]*S1[c1,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S1[c1,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    - (3*S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/8 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,m]*S1[c1,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/6 \
                                    - (3*S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/8 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,m]*S1[c1,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/6 \
                                    + (S2[c1,i,i]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S2[c1,i,i]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/12 \
                                    + (S3[c1,i,i,j]*S2[c1,j,k]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (S3[c1,i,i,j]*S2[c1,j,l]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/18 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    - (S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/2 \
                                    + (4*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (2*S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/9 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    - (S3[c1,i,i,k]*S3[c1,j,j,l]*S2[c1,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/4 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/4 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    - (S3[c1,i,i,l]*S3[c1,j,j,k]*S2[c1,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/4 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/4 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/12 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/12 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/4 \
                                    + (23*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (7*S3[c1,i,i,j]*S3[c1,j,l,l]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/8 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/24 \
                                    + (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/144 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m])/48 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/3 \
                                    + (4*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/27 \
                                    + (2*S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/27 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/18 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/12 \
                                    + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/36 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/18 \
                                    + (S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/36 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,m]*S1[c1,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/9 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,m]*S1[c1,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m])/18 \
                                    - (S3[c1,i,i,j]*S3[c1,j,k,k]*S2[c1,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/3 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    - (S3[c1,i,i,k]*S3[c1,j,j,k]*S2[c1,l,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/6 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m])/12 \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/18 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,m]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m])/36 \
                                    + 2*S2[c1,i,j]*S3[c1,i,j,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,i]*S3[c1,j,j,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,l,m]**2*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m] \
                                    - S3[c1,i,i,m]*S3[c1,j,k,m]**2*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,j,m]**2*S2[c1,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + (S3[c1,i,j,k]**2*S3[c1,l,l,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[m]*mx03[l])/6 \
                                    + S3[c1,i,k,l]*S3[c1,i,l,m]*S3[c1,j,k,m]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m] \
                                    + 2*S3[c1,i,k,m]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,j,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m] \
                                    + S1[c1,i]*S3[c1,i,j,k]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + S1[c1,i]*S3[c1,i,j,l]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + S1[c1,i]*S3[c1,i,j,m]*S3[c1,j,k,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,m,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 2*S2[c1,i,j]*S3[c1,i,k,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 2*S2[c1,i,j]*S3[c1,i,l,m]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 3*S2[c1,i,k]*S3[c1,i,j,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,k]*S3[c1,i,j,m]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,k]*S3[c1,i,l,m]*S3[c1,j,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 3*S2[c1,i,l]*S3[c1,i,j,m]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,l]*S3[c1,i,k,m]*S3[c1,j,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 3*S2[c1,i,m]*S3[c1,i,j,k]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,m]*S3[c1,i,j,k]*S3[c1,j,m,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 3*S2[c1,i,m]*S3[c1,i,j,l]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 2*S2[c1,i,m]*S3[c1,i,j,m]*S3[c1,j,k,m]*S3[c1,k,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,m]*S3[c1,i,j,m]*S3[c1,j,l,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,m]*S3[c1,i,k,l]*S3[c1,j,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,m]*S3[c1,i,k,m]*S3[c1,j,j,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,m]*S3[c1,i,l,m]*S3[c1,j,j,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,j,k]*S3[c1,i,l,m]*S2[c1,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,j,l]*S3[c1,i,k,m]*S2[c1,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,j,m]*S3[c1,i,k,l]*S2[c1,j,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,j,m]*S3[c1,i,k,m]*S2[c1,j,l]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,j,m]*S3[c1,i,l,m]*S2[c1,j,k]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 2*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m] \
                                    + S3[c1,i,j,m]*S3[c1,i,l,m]*S3[c1,j,l,m]*S3[c1,k,k,l]*mx02[i]*mx02[j]*mx02[k]*mx03[l]*mx03[m] \
                                    + 3*S3[c1,i,j,k]*S3[c1,i,l,m]*S2[c1,j,k]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + 3*S3[c1,i,j,l]*S3[c1,i,k,m]*S2[c1,j,k]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + 3*S3[c1,i,j,m]*S3[c1,i,k,l]*S2[c1,j,k]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + 2*S2[c1,i,i]*S3[c1,j,k,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,j]*S3[c1,i,j,l]*S3[c1,k,l,m]*S3[c1,k,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,j]*S3[c1,i,k,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,j]*S3[c1,i,k,m]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,k]*S3[c1,i,j,l]*S3[c1,j,k,m]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S2[c1,i,k]*S3[c1,i,j,m]*S3[c1,j,k,m]*S3[c1,l,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 2*S2[c1,i,m]*S3[c1,i,j,k]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,i,k]*S2[c1,j,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,i,l]*S2[c1,j,m]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,i,m]*S2[c1,j,k]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,i,m]*S2[c1,j,l]*S3[c1,j,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 2*S3[c1,i,j,l]*S3[c1,i,j,m]*S2[c1,k,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    - 2*S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,k,m]*S2[c1,l,l]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,i,m]*S3[c1,j,k,m]*S3[c1,j,l,m]*S3[c1,k,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m] \
                                    + S3[c1,i,j,m]*S3[c1,i,k,m]*S3[c1,j,k,l]*S3[c1,l,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx04[m] \
                                    + S3[c1,i,j,j]*S3[c1,i,k,l]*S2[c1,k,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + S3[c1,i,j,j]*S3[c1,i,k,m]*S2[c1,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + 3*S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + 3*S3[c1,i,j,k]*S3[c1,i,j,m]*S2[c1,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + 2*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + 2*S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,m]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + 2*S3[c1,i,i,j]*S3[c1,j,k,m]*S2[c1,k,l]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + S3[c1,i,i,j]*S3[c1,j,l,m]*S2[c1,k,k]*S2[c1,l,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    - S3[c1,i,j,k]*S3[c1,i,j,l]*S2[c1,k,l]*S3[c1,m,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    - S3[c1,i,i,j]*S3[c1,j,k,l]*S2[c1,k,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m] \
                                    + S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,m]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx03[m] \
                                    + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[m]*mx03[l])/6 \
                                    + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,l]*S2[c1,m,m]*mx02[i]*mx02[j]*mx02[k]*mx02[m]*mx03[l])/12
                                    
                                for o in range(n):
                                    if i!=j and i!=k and i!=l and i!=m and i!=o and j!=k and j!=l and j!=m and j!=o and k!=l and k!=m and k!=o and l!=m and l!=o and m!=o:
                                        mu4Y3ijklmo[c1,i,j,k,l,m,o] = (S3[c1,i,j,k]**2*S3[c1,l,m,o]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,m,o]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,m,o]**2*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/24 \
                                        + (S3[c1,i,i,m]*S3[c1,j,k,l]**2*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,o]*S3[c1,j,k,l]**2*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,j,k]**2*S3[c1,l,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,j,k]**2*S3[c1,l,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,j,k]*S3[c1,i,k,l]*S3[c1,j,m,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,j,k]*S3[c1,i,k,m]*S3[c1,j,l,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (2*S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,j,k]*S3[c1,i,k,o]*S3[c1,j,l,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (2*S3[c1,i,j,k]*S3[c1,i,l,o]*S3[c1,j,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (2*S3[c1,i,j,k]*S3[c1,i,m,o]*S3[c1,j,k,l]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,j,l]*S3[c1,i,k,o]*S3[c1,j,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,j,l]*S3[c1,i,m,o]*S3[c1,j,k,k]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,j,m]*S3[c1,i,k,o]*S3[c1,j,k,l]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,j,m]*S3[c1,i,l,o]*S3[c1,j,k,k]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,k,l]*S3[c1,i,j,o]*S3[c1,j,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,k,l]*S3[c1,i,m,o]*S3[c1,j,j,k]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,k,m]*S3[c1,i,j,o]*S3[c1,j,k,l]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,k,m]*S3[c1,i,l,o]*S3[c1,j,j,k]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,j,o]*S3[c1,i,l,m]*S3[c1,j,k,k]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,l,m]*S3[c1,i,k,o]*S3[c1,j,j,k]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,m,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,i,k]*S3[c1,j,k,m]*S3[c1,j,l,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,i,k]*S3[c1,j,l,m]*S3[c1,j,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,m,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,j,j]*S3[c1,i,k,o]*S3[c1,k,l,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,j,j]*S3[c1,i,l,o]*S3[c1,k,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,j,j]*S3[c1,i,m,o]*S3[c1,k,k,l]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,i,l]*S3[c1,j,k,k]*S3[c1,j,m,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,l]*S3[c1,j,k,m]*S3[c1,j,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (2*S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,m,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (2*S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (2*S3[c1,i,j,k]*S3[c1,i,j,o]*S3[c1,k,l,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,j,k]*S3[c1,i,l,m]*S3[c1,j,k,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,j,k]*S3[c1,i,l,o]*S3[c1,j,k,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,m]*S3[c1,j,k,k]*S3[c1,j,l,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,m]*S3[c1,j,k,l]*S3[c1,j,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,j,l]*S3[c1,i,k,m]*S3[c1,j,k,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,j,l]*S3[c1,i,j,o]*S3[c1,k,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,j,l]*S3[c1,i,k,o]*S3[c1,j,k,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,j,m]*S3[c1,i,k,l]*S3[c1,j,k,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,j,m]*S3[c1,i,j,o]*S3[c1,k,k,l]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,k,l]*S3[c1,i,j,o]*S3[c1,j,k,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,o]*S3[c1,j,k,k]*S3[c1,j,l,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,o]*S3[c1,j,k,l]*S3[c1,j,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (7*S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,m,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (7*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (7*S3[c1,i,i,j]*S3[c1,j,k,o]*S3[c1,k,l,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,i,j]*S3[c1,j,l,o]*S3[c1,k,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,j]*S3[c1,j,m,o]*S3[c1,k,k,l]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,m,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/24 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/24 \
                                        + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,k]*S3[c1,j,k,l]*S3[c1,j,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,o]*S3[c1,k,l,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/24 \
                                        + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,j,j]*S3[c1,i,k,l]*S3[c1,k,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,j,j]*S3[c1,i,k,m]*S3[c1,k,l,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/27 \
                                        + (S3[c1,i,j,j]*S3[c1,i,l,m]*S3[c1,k,k,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/54 \
                                        + (S3[c1,i,j,j]*S3[c1,i,k,o]*S3[c1,k,l,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/54 \
                                        + (S3[c1,i,j,j]*S3[c1,i,l,o]*S3[c1,k,k,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/108 \
                                        + (5*S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,m,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/48 \
                                        + (S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,k,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,i,l]*S3[c1,j,k,l]*S3[c1,j,k,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,i,l]*S3[c1,j,j,o]*S3[c1,k,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/48 \
                                        + (S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,j,k]*S3[c1,i,j,l]*S3[c1,k,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (S3[c1,i,j,k]*S3[c1,i,j,m]*S3[c1,k,l,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/9 \
                                        + (S3[c1,i,j,k]*S3[c1,i,j,o]*S3[c1,k,l,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (5*S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,k,o]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/48 \
                                        + (S3[c1,i,i,m]*S3[c1,j,j,o]*S3[c1,k,k,l]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/48 \
                                        + (S3[c1,i,j,l]*S3[c1,i,j,m]*S3[c1,k,k,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/18 \
                                        + (S3[c1,i,j,l]*S3[c1,i,j,o]*S3[c1,k,k,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (5*S3[c1,i,i,o]*S3[c1,j,j,k]*S3[c1,k,l,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,o]*S3[c1,j,j,l]*S3[c1,k,k,m]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/48 \
                                        + (S3[c1,i,i,o]*S3[c1,j,j,m]*S3[c1,k,k,l]*S3[c1,l,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/48 \
                                        + (S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/6 \
                                        + (S3[c1,i,i,j]*S3[c1,j,k,l]*S3[c1,k,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/12 \
                                        + (2*S3[c1,i,i,j]*S3[c1,j,k,m]*S3[c1,k,l,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/27 \
                                        + (S3[c1,i,i,j]*S3[c1,j,l,m]*S3[c1,k,k,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/27 \
                                        + (S3[c1,i,i,j]*S3[c1,j,k,o]*S3[c1,k,l,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/27 \
                                        + (S3[c1,i,i,j]*S3[c1,j,l,o]*S3[c1,k,k,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/54 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/24 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,l]*S3[c1,k,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/48 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,m]*S3[c1,k,l,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,o]*S3[c1,k,l,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/24 \
                                        + (S3[c1,i,i,l]*S3[c1,j,j,k]*S3[c1,k,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/48 \
                                        + (S3[c1,i,i,l]*S3[c1,j,j,m]*S3[c1,k,k,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,l]*S3[c1,j,j,o]*S3[c1,k,k,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/144 \
                                        + (S3[c1,i,i,m]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,i,m]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,o]*S3[c1,j,j,k]*S3[c1,k,l,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,o]*S3[c1,j,j,l]*S3[c1,k,k,l]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/144 \
                                        + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/36 \
                                        + (S3[c1,i,i,j]*S3[c1,j,k,k]*S3[c1,l,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,m]*S3[c1,m,o,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/72 \
                                        + (S3[c1,i,i,k]*S3[c1,j,j,k]*S3[c1,l,l,o]*S3[c1,m,m,o]*mx02[i]*mx02[j]*mx02[k]*mx02[l]*mx02[m]*mx02[o])/144
                                        
        for c1 in range(nPoints):
            fcmThird[c1] = fcmSecond[c1] + np.sum(mu4Y3i[c1,:]) + np.sum(mu4Y3ij[c1,:,:]) + np.sum(mu4Y3ijk[c1,:,:,:]) + np.sum(mu4Y3ijkl[c1,:,:,:,:]) + np.sum(mu4Y3ijklm[c1,:,:,:,:,:]) + np.sum(mu4Y3ijklmo[c1,:,:,:,:,:,:])
        
        self.fcmThird = fcmThird
        return fcmThird