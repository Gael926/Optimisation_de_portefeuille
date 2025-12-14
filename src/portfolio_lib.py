import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# Niveau 2, constraintes (Pymoo) 

class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, sigma, k_card=10, trans_cost=0.005, w_prev=None):
        """
        Optimisation Multi-Objectifs avec contraintes de Cardinalité et Coûts.
        Objectifs:
        1. Maximiser Rendement
        2. Minimiser Risque
        3. Minimiser Coûts (si w_prev est fourni)
        """
        self.mu = mu
        self.sigma = sigma
        self.k_card = k_card
        self.trans_cost = trans_cost
        self.w_prev = w_prev if w_prev is not None else np.zeros(len(mu))
        
        n_obj = 3 # Rendement, Risque, Transaction Cost
        
        super().__init__(n_var=len(mu),
                         n_obj=n_obj,
                         n_ieq_constr=0,
                         xl=0.0,
                         xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # Cardinalité (Top K), on garde les K plus grands poids, les autres à 0
        idx = np.argsort(x)
        x_clean = np.zeros_like(x)
        idx_top_k = idx[-self.k_card:]
        x_clean[idx_top_k] = x[idx_top_k]
        
        # Budget (Somme = 1)
        s = np.sum(x_clean)
        if s > 1e-6:
            w = x_clean / s
        else:
            w = x_clean # Should not happen often if xl=0, xu=1
            w[idx_top_k] = 1.0 / self.k_card
            
        # Objectifs
        
        # f1: Rendement (Minimiser -R)
        f1 = - (w @ self.mu)
        
        # f2: Risque (Minimiser Variance)
        f2 = w.T @ self.sigma @ w
        
        # f3: Coûts
        turnover = np.sum(np.abs(w - self.w_prev))
        f3 = turnover * self.trans_cost
        
        out["F"] = [f1, f2, f3]
        # On pourrait aussi retourner le "w" décodé si besoin, mais NSGA2 travaille sur les gènes "x"

def optimize_moo(mu, sigma, k_card, trans_cost, w_prev=None, pop_size=50, n_gen=50):
    
    # Lance l'optimisation NSGA-II, retourne les résultats (Front de Pareto)
    problem = PortfolioProblem(mu, sigma, k_card=k_card, trans_cost=trans_cost, w_prev=w_prev)
    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)
    
    res = minimize(problem, algorithm, termination, seed=1, verbose=False)
    
    # Post-traitement pour avoir les vrais poids normalisés pour chaque solution
    # Car res.X contient les variables de décision brutes
    
    final_weights = []
    for x in res.X:
        idx = np.argsort(x)
        w = np.zeros_like(x)
        idx_top = idx[-k_card:]
        w[idx_top] = x[idx_top]
        s = np.sum(w)
        if s > 0:
            w = w / s
        final_weights.append(w)
        
    return res, np.array(final_weights)
