import numpy as np

def LLG(t, m, Beff, gammaLL, alpha):
    return -gammaLL * (np.cross(m, Beff) + alpha*np.cross(m, np.cross(m, Beff))) / (1 + alpha**2)

def cavity_real(t, xp, wc, kappa):
    M = np.array([[-kappa, -wc],
                 [wc, -kappa]])
    
    return np.dot(M, xp)

def LLG_explicit(t, mxp, Bext, Brms, wc, kappa, N, gammaLL, alpha):
    m = mxp[:-2]
    xp = mxp[-2:]
    
    Beff = Bext + Brms*xp[0]
    dm = -gammaLL * (np.cross(m, Beff) + alpha*np.cross(m, np.cross(m, Beff))) / (1 + alpha**2)
    
    M = np.array([[-kappa, -wc],
                 [wc, -kappa]])
    dxp = np.dot(M, xp)
    dxp[1] += -N * gammaLL * np.dot(Brms, m)
    
    return np.append(dm, dxp)

def LLG_memory(t, m, Bext, Brms, wc, kappa, N, gammaLL, alpha, xp0):
    dt = t - LLG_memory.last_t
    LLG_memory.S += np.exp(kappa * t) * np.sin(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    LLG_memory.C += np.exp(kappa * t) * np.cos(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    G = np.exp(-kappa * t) * (np.cos(wc * t)*(xp0[0] - gammaLL*N*LLG_memory.S) - np.sin(wc * t)*(xp0[1] - gammaLL*N*LLG_memory.C))
    Beff = Bext + Brms*G
    LLG_memory.last_t = t
    
    return -gammaLL * (np.cross(m, Beff) + alpha*np.cross(m, np.cross(m, Beff))) / (1 + alpha**2)

def f_alpha(ms, ts, Brms, wc, kappa, N, gammaLL, alpha0):
    dt = ts[1] - ts[0]
    I = 0
    alphas = np.empty([len(ts)], dtype=complex)
    alphas[0] = alpha0
    for i, t in enumerate(ts[1:], start=1):
        I += np.exp(kappa * t) * np.exp(1j * wc * t) * np.dot(Brms, ms[i]) * dt
        alphas[i] = np.exp(-kappa * t) * np.exp(-1j * wc * t) * (alpha0 + 1j*N/2*gammaLL*I)
        
    return alphas

