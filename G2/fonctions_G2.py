

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



########################################################################## Construction de r(t)



############ Parties stochastiques

def mvts_browniens(T, dt, N, rho):

    # Mvts browniens indépendants
    dW1_ind = np.random.normal(0, np.sqrt(dt), N)
    dW2_ind = np.random.normal(0, np.sqrt(dt), N)

    # Mvts browniens corrélés
    dW1 = dW1_ind
    dW2 = rho * dW1_ind + np.sqrt(1 - rho**2) * dW2_ind

    # Trajectoires
    W1 = np.cumsum(dW1)
    W2 = np.cumsum(dW2)

    W1 = np.insert(W1, 0, 0) # initialiser à 0 OK ? 
    W2 = np.insert(W2, 0, 0)

    t = np.linspace(0, T, N+1)

    plt.figure(figsize=(10, 5))
    plt.plot(t, W1, label="W1(t)")
    plt.plot(t, W2, label="W2(t)")
    plt.legend()
    plt.title("Mouvements browniens corrélés")
    plt.xlabel("t")
    plt.show()

    return W1, W2




def parties_stochastiques(N, a, b, sigma, eta, W1, W2):

    x = np.zeros(N)
    y = np.zeros(N)

    for t in range(1, N):
        x[t] = x[t-1] * np.exp(-a) + W1[t-1] * sigma * np.sqrt((1 - np.exp(-2*a))/(2*a))
        y[t] = y[t-1]* np.exp(-b) + W2[t-1] * eta * np.sqrt((1 - np.exp(-2*b))/(2*b))

    return x, y



############ Partie déterministe


# Taux forward par la méthode NSS
def forward_0t(t, parametres):

    if np.isscalar(t):
        if t == 0.0: # taux spot initial ????????????????????????????????????????????????????????
            return 0.04  # MIS AU PIF !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    alpha1 = (1-np.exp(-t/parametres[4])) / (t/parametres[4])
    alpha2 = alpha1 - np.exp(-t/parametres[4])
    alpha3 = (1-np.exp(-t/parametres[5])) / (t/parametres[5]) - np.exp(-t/parametres[5])

    return parametres[0] + parametres[1] * alpha1 + parametres[2] * alpha2 + parametres[3] *alpha3




def NSS_optimisation(parametres, maturites, taux):

    def NSS_objectif(parametres, maturites, taux):

        return np.sum((forward_0t(maturites, parametres)-taux)**2)
    
    return minimize(NSS_objectif, x0=parametres, args = (maturites, taux), method="Nelder-Mead")




def partie_deterministe(t, sigma, eta, a, b, parametres_NSS):

    taux_forward = forward_0t(t, parametres_NSS)
    terme1 = -(sigma**2 / (2 * a**2)) * (1 - np.exp(-a * t))**2
    terme2 = -(eta**2 / (2 * b**2)) * (1 - np.exp(-b * t))**2
    terme3 = -(sigma * eta / (a * b)) * (1 - np.exp(-a * t)) * (1 - np.exp(-b * t))

    return taux_forward + terme1 + terme2 + terme3






########################################################################## Princing zero-coupon



def index(maturites, t):

    return np.searchsorted(maturites, t)


def V(t, T, sigma, eta, a, b, rho):

    variance1 = (sigma**2 / a**2) * (T - t + (2 / a) * np.exp(-a * (T - t)) - (1 / (2 * a)) * np.exp(-2 * a * (T - t)) - (3 / (2 * a)))
    variance2 = (eta**2 / b**2) * (T - t + (2 / b) * np.exp(-b * (T - t)) - (1 / (2 * b)) * np.exp(-2 * b * (T - t)) - (3 / (2 * b)))
    covariance = ((2 * rho * sigma * eta) / (a * b)) * (T - t + (1 - np.exp(-a * (T - t))) / a + (1 - np.exp(-b * (T - t))) / b - (np.exp(-(a + b) * (T - t)) - 1) / (a + b))
    
    return variance1 + variance2 + covariance


                                
def B(t, T, z):
    return (1 - np.exp(-z * (T - t))) / z



def P_0T(T, parametres, n=100):
    '''
    Calcul du prix zéro-coupon de maturité T, à l'instant t = 0 

    n = subdivision pour l'intégrale
    '''
    dt = T / n
    integrale = 0.0

    for i in range(n):
        t_i = i * dt
        t_i1 = (i + 1) * dt
        integrale += (forward_0t(t_i, parametres) + forward_0t(t_i1, parametres)) * dt / 2
        
    return np.exp(-integrale)



def A(t, T, sigma, eta, a, b, rho, parametres_NSS):

    V_tT = V(t, T, sigma, eta, a, b, rho)
    V_0T = V(0, T, sigma, eta, a, b, rho)
    V_0t = V(0, t, sigma, eta, a, b, rho)

    P_OT = P_0T(T, parametres_NSS)
    P_Ot = P_0T(t, parametres_NSS)

    return (P_OT / P_Ot) * np.exp(0.5 * (V_tT - V_0T + V_0t))




def prix_zero_coupon(t, T, a, b, sigma, eta, rho, x, y, parametres_NSS):

    A_tT = A(t, T, sigma, eta, a, b, rho, parametres_NSS)
    B_a = B(t, T, a)
    B_b = B(t, T, b)

    return A_tT * np.exp(-B_a * x - B_b * y)       


