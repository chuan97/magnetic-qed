import numpy as np

def global_(wz, wc, lam):
    def para(wz, wc, lam):
        innerroot = np.sqrt(wz**4 + wc**4 - 2*wz**2*wc**2 + 16*lam**2*wz*wc)
        return np.sqrt(0.5 * (wz**2 + wc**2 - innerroot)), np.sqrt(0.5 * (wz**2 + wc**2 + innerroot))
    
    def ferro(wz, wc, lam):
        mu = wz * wc / (4 * lam**2)
        g = lam * mu * np.sqrt(2 / (1 + mu))
        wztilde = wz * (1 + mu) / (2 * mu)
        eps = wz * (1 - mu) * (3 + mu) / (8 * mu * (1 + mu))
        
        innerroot = np.sqrt(wztilde**4 + wc**4 - 2*wztilde**2*wc**2 + 16*g**2*wztilde*wc + 4*(eps**2*wztilde**2 + eps*wztilde**3 - wc**2*eps*wztilde))
        return np.sqrt(0.5 * (wztilde**2 + wc**2 + 2*eps*wztilde - innerroot)), np.sqrt(0.5 * (wztilde**2 + wc**2 + 2*eps*wztilde + innerroot))
    
    if 4 * lam**2 < wz * wc:
        # paramagnetic phase
        return para(wz, wc, lam)
    else:
        # ferromagnetic phase
        return ferro(wz, wc, lam)