import numpy as np


def LLG(t, m, Beff, gammaLL, alpha):
    return (
        -gammaLL
        * (np.cross(m, Beff) + alpha * np.cross(m, np.cross(m, Beff)))
        / (1 + alpha**2)
    )


def cavity_real(t, xp, wc, kappa):
    M = np.array([[-kappa, -wc], [wc, -kappa]])

    return np.dot(M, xp)


def LLG_explicit(t, mxp, Bext, Brms, wc, kappa, N, gammaLL, alpha):
    m = mxp[:-2]
    xp = mxp[-2:]

    Beff = Bext + Brms * xp[0]
    dm = (
        -gammaLL
        * (np.cross(m, Beff) + alpha * np.cross(m, np.cross(m, Beff)))
        / (1 + alpha**2)
    )

    M = np.array([[-kappa, -wc], [wc, -kappa]])
    dxp = np.dot(M, xp)
    dxp[1] += -N * gammaLL * np.dot(Brms, m)

    return np.append(dm, dxp)


def LLG_memory(t, m, Bext, Brms, wc, kappa, N, gammaLL, alpha, xp0):
    dt = t - LLG_memory.last_t
    LLG_memory.last_t = t
    LLG_memory.S += np.exp(kappa * t) * np.sin(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    LLG_memory.C += np.exp(kappa * t) * np.cos(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    G = np.exp(-kappa * t) * (
        np.cos(wc * t) * (xp0[0] - gammaLL * N * LLG_memory.S)
        - np.sin(wc * t) * (xp0[1] - gammaLL * N * LLG_memory.C)
    )
    Beff = Bext + Brms * G

    return (
        -gammaLL
        * (np.cross(m, Beff) + alpha * np.cross(m, np.cross(m, Beff)))
        / (1 + alpha**2)
    )


def LLG_memory_3(t, m, Bext, Brms, wc, kappa, N, gammaLL, alpha, xp0):
    dt = t - LLG_memory_3.last_t

    # if undoing a previous step that had too big an error
    if dt < 0.0:
        # restore the memory to its state before the last full step
        LLG_memory_3.S = LLG_memory_3.cache_start_S
        LLG_memory_3.C = LLG_memory_3.cache_start_C

        # and recompute dt
        dt = t - LLG_memory_3.cache_start_t

    lastcall = not dt
    # if it is the 6th RK call (k7) which is evaluated for the same time as the 5th call (k6)
    # (the time being a full timestep after the previous step)
    if lastcall:
        # cache memory term in case the integrator decides to repeat the step
        LLG_memory_3.cache_start_S = LLG_memory_3.cache_final_S
        LLG_memory_3.cache_start_C = LLG_memory_3.cache_final_C
        LLG_memory_3.cache_start_t = LLG_memory_3.last_t
        LLG_memory_3.cache_final_S = LLG_memory_3.S
        LLG_memory_3.cache_final_C = LLG_memory_3.C

    LLG_memory_3.last_t = t
    LLG_memory_3.S += (
        np.exp(kappa * t) * np.sin(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    )
    LLG_memory_3.C += (
        np.exp(kappa * t) * np.cos(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    )
    G = np.exp(-kappa * t) * (
        np.cos(wc * t) * (xp0[0] - gammaLL * N * LLG_memory_3.S)
        - np.sin(wc * t) * (xp0[1] - gammaLL * N * LLG_memory_3.C)
    )
    Beff = Bext + Brms * G

    return (
        -gammaLL
        * (np.cross(m, Beff) + alpha * np.cross(m, np.cross(m, Beff)))
        / (1 + alpha**2)
    )


def LLG_memory_1(t, m, Bext, Brms, wc, kappa, N, gammaLL, alpha, xp0):
    dt = t - LLG_memory_1.last_t
    lastcall = not dt
    # if it is the 6th RK call (k7) which is evaluated for the same time as the 5th call (k6)
    # (the time being a full timestep after the previous step)
    # we have to recompute the memory term: subtract contributions from partial steps and add the contribution from the full step
    if lastcall:
        # subtract contributions from previous partial steps
        LLG_memory_1.last_t -= LLG_memory_1.dt
        LLG_memory_1.S -= LLG_memory_1.dS
        LLG_memory_1.C -= LLG_memory_1.dC

        # set accumulators to zero for future partial steps
        LLG_memory_1.dS = LLG_memory_1.dC = LLG_memory_1.dt = 0.0

        # make dt be equal to a full time step, difference between current time and time from last lastcall
        dt = t - LLG_memory_1.last_t
    LLG_memory_1.last_t = t

    # compute change in memory terms
    dS = np.exp(kappa * t) * np.sin(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    dC = np.exp(kappa * t) * np.cos(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    LLG_memory_1.S += dS
    LLG_memory_1.C += dC

    if not lastcall:
        # accumulate contributions from partial time steps since last lastcall
        LLG_memory_1.dS += dS
        LLG_memory_1.dC += dC
        LLG_memory_1.dt += dt

    G = np.exp(-kappa * t) * (
        np.cos(wc * t) * (xp0[0] - gammaLL * N * LLG_memory_1.S)
        - np.sin(wc * t) * (xp0[1] - gammaLL * N * LLG_memory_1.C)
    )
    Beff = Bext + Brms * G

    return (
        -gammaLL
        * (np.cross(m, Beff) + alpha * np.cross(m, np.cross(m, Beff)))
        / (1 + alpha**2)
    )


def LLG_memory_2(t, m, Bext, Brms, wc, kappa, N, gammaLL, alpha, xp0):
    dt = t - LLG_memory_2.last_t

    # if undoing a previous step that had too big an error
    if dt < 0.0:
        # restore the memory to its state before the last full step
        LLG_memory_2.S = LLG_memory_2.cachedS
        LLG_memory_2.C = LLG_memory_2.cachedC
        LLG_memory_2.last_t = LLG_memory_2.cachedlast_t

        # and recompute dt
        dt = t - LLG_memory_2.last_t

    lastcall = not dt
    # if it is the 6th RK call (k7) which is evaluated for the same time as the 5th call (k6)
    # (the time being a full timestep after the previous full step)
    # recompute the memory term: subtract contributions from partial steps and add the contribution from the full step
    if lastcall:
        # subtract contributions from previous partial steps
        LLG_memory_2.last_t -= LLG_memory_2.dt
        LLG_memory_2.S -= LLG_memory_2.dS
        LLG_memory_2.C -= LLG_memory_2.dC

        # set accumulators to zero for future partial steps
        LLG_memory_2.dS = LLG_memory_2.dC = LLG_memory_2.dt = 0.0

        # make dt be equal to a full time step, difference between current time and time from last lastcall
        dt = t - LLG_memory_2.last_t

        # cache restored memory in case the full step has to be redone if the error is too big
        LLG_memory_2.cachedlast_t = LLG_memory_2.last_t
        LLG_memory_2.cachedS = LLG_memory_2.S
        LLG_memory_2.cachedC = LLG_memory_2.C

    LLG_memory_2.last_t = t

    # compute change in memory terms
    dS = np.exp(kappa * t) * np.sin(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    dC = np.exp(kappa * t) * np.cos(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    LLG_memory_2.S += dS
    LLG_memory_2.C += dC

    if not lastcall:
        # accumulate contributions from partial time steps since last lastcall
        LLG_memory_2.dt += dt
        LLG_memory_2.dS += dS
        LLG_memory_2.dC += dC

    G = np.exp(-kappa * t) * (
        np.cos(wc * t) * (xp0[0] - gammaLL * N * LLG_memory_2.S)
        - np.sin(wc * t) * (xp0[1] - gammaLL * N * LLG_memory_2.C)
    )
    Beff = Bext + Brms * G

    return (
        -gammaLL
        * (np.cross(m, Beff) + alpha * np.cross(m, np.cross(m, Beff)))
        / (1 + alpha**2)
    )


def LLG_memory_2_log(t, m, Bext, Brms, wc, kappa, N, gammaLL, alpha, xp0, logfile):
    dt = t - LLG_memory_2_log.last_t

    # if undoing a previous step that had too big an error
    if dt < 0.0:
        # restore the memory to its state before the last full step
        LLG_memory_2_log.S = LLG_memory_2_log.cachedS
        LLG_memory_2_log.C = LLG_memory_2_log.cachedC
        LLG_memory_2_log.last_t = LLG_memory_2_log.cachedlast_t

        # and recompute dt
        dt = t - LLG_memory_2_log.last_t

        # log discard
        logfile.write("previous step discarded\n")

    lastcall = not dt
    # if it is the 6th RK call (k7) which is evaluated for the same time as the 5th call (k6)
    # (the time being a full timestep after the previous full step)
    # recompute the memory term: subtract contributions from partial steps and add the contribution from the full step
    if lastcall:
        # subtract contributions from previous partial steps
        LLG_memory_2_log.last_t -= LLG_memory_2_log.dt
        LLG_memory_2_log.S -= LLG_memory_2_log.dS
        LLG_memory_2_log.C -= LLG_memory_2_log.dC

        # set accumulators to zero for future partial steps
        LLG_memory_2_log.dS = LLG_memory_2_log.dC = LLG_memory_2_log.dt = 0.0

        # make dt be equal to a full time step, difference between current time and time from last lastcall
        dt = t - LLG_memory_2_log.last_t

        # cache restored memory in case the full step has to be redone if the error is too big
        LLG_memory_2_log.cachedlast_t = LLG_memory_2_log.last_t
        LLG_memory_2_log.cachedS = LLG_memory_2_log.S
        LLG_memory_2_log.cachedC = LLG_memory_2_log.C

        # log dt
        logfile.write(str(t) + ", " + str(dt) + "\n")

    LLG_memory_2_log.last_t = t

    # compute change in memory terms
    dS = np.exp(kappa * t) * np.sin(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    dC = np.exp(kappa * t) * np.cos(wc * t) * np.dot(Brms, m) * dt * (t > 0)
    LLG_memory_2_log.S += dS
    LLG_memory_2_log.C += dC

    if not lastcall:
        # accumulate contributions from partial time steps since last lastcall
        LLG_memory_2_log.dt += dt
        LLG_memory_2_log.dS += dS
        LLG_memory_2_log.dC += dC

    G = np.exp(-kappa * t) * (
        np.cos(wc * t) * (xp0[0] - gammaLL * N * LLG_memory_2_log.S)
        - np.sin(wc * t) * (xp0[1] - gammaLL * N * LLG_memory_2_log.C)
    )
    Beff = Bext + Brms * G

    return (
        -gammaLL
        * (np.cross(m, Beff) + alpha * np.cross(m, np.cross(m, Beff)))
        / (1 + alpha**2)
    )


def f_alpha(ms, ts, Brms, wc, kappa, N, gammaLL, alpha0):
    dt = ts[1] - ts[0]
    I = 0
    alphas = np.empty([len(ts)], dtype=complex)
    alphas[0] = alpha0
    for i, t in enumerate(ts[1:], start=1):
        I += np.exp(kappa * t) * np.exp(1j * wc * t) * np.dot(Brms, ms[i]) * dt
        alphas[i] = (
            np.exp(-kappa * t)
            * np.exp(-1j * wc * t)
            * (alpha0 + 1j * N / 2 * gammaLL * I)
        )

    return alphas
