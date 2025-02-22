

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

def forward_0t(t, parametres):

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
