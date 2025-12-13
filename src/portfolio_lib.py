
import numpy as np
import pandas as pd
import scipy.optimize as sco
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --- LEVEL 1: MARKOWITZ ---

def f1_rendement(w, mu):
    """Minimise -Rendement (donc maximise Rendement)"""
    return -w.T @ mu

def f2_risque(w, sigma):
    """Minimise le risque (Variance)"""
    return w.T @ sigma @ w

def get_rend_vol_sr(w, mu, sigma):
    """Retourne Rendement, Volatilité, Sharpe Ratio"""
    rend = w @ mu
    vol = np.sqrt(w.T @ sigma @ w)
    sr = rend / vol
    return rend, vol, sr

def neg_sharpe_ratio(w, mu, sigma):
    """Fonction à minimiser pour maximiser le Sharpe Ratio"""
    rend, vol, _ = get_rend_vol_sr(w, mu, sigma)
    return -rend / vol

def optimize_markowitz(mu, sigma):
    """
    Optimisation convexe classique (Level 1).
    Retourne:
    - w_sharpe: poids du portefeuille Tangent (Max Sharpe)
    - frontier: (vols, rends) pour la frontière efficiente
    """
    num_assets = len(mu)
    args = (mu, sigma)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # 1. Max Sharpe Ratio
    w0 = np.ones(num_assets) / num_assets
    res_sharpe = sco.minimize(neg_sharpe_ratio, w0, args=args,
                              method='SLSQP', bounds=bounds, constraints=constraints)
    w_sharpe = res_sharpe.x
    
    # 2. Frontière Efficiente
    # On balaie une plage de rendements cibles
    target_returns = np.linspace(mu.min(), mu.max(), 50)
    efficient_vols = []
    efficient_rends = []
    
    for target in target_returns:
        cons_loop = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: -f1_rendement(x, mu) - target}
        )
        res = sco.minimize(f2_risque, w0, args=(sigma,),
                           method='SLSQP', bounds=bounds, constraints=cons_loop)
        if res.success:
            efficient_vols.append(np.sqrt(res.fun))
            efficient_rends.append(target)
            
    return w_sharpe, (efficient_vols, efficient_rends)

# --- LEVEL 2: CONSTRAINTS (Pymoo) ---

class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, sigma, k_card=10, trans_cost=0.005, w_prev=None):
        """
        Optimisation Multi-Objectifs avec contraintes de Cardinalité et Coûts.
        Objectifs:
        1. Maximiser Rendement
        2. Minimiser Risque
        3. Minimiser Coûts (si w_prev est fourni)
        
        Note: Pour simplifier dans l'app, on peut se limiter à 2 objectifs si w_prev est None,
        ou fixer les coûts. Ici on suit la logique du notebook.
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
        # 1. Cardinalité (Top K)
        # On garde les K plus grands poids, les autres à 0
        idx = np.argsort(x)
        x_clean = np.zeros_like(x)
        idx_top_k = idx[-self.k_card:]
        x_clean[idx_top_k] = x[idx_top_k]
        
        # 2. Budget (Somme = 1)
        s = np.sum(x_clean)
        if s > 1e-6:
            w = x_clean / s
        else:
            w = x_clean # Should not happen often if xl=0, xu=1
            w[idx_top_k] = 1.0 / self.k_card
            
        # 3. Objectifs
        
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
    """
    Lance l'optimisation NSGA-II.
    Retourne les résultats (Front de Pareto).
    """
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

# --- LEVEL 3: ROBUSTNESS (Resampling) ---

def resampling_efficient_frontier(df_returns, n_simulations=50, n_samples=None):
    """
    Génère N frontières efficientes basées sur le ré-échantillonnage des rendements.
    """
    if n_samples is None:
        n_samples = len(df_returns)
        
    frontiers = []
    
    for _ in range(n_simulations):
        # Bootstrap resampling
        resampled_df = df_returns.sample(n=n_samples, replace=True)
        
        # Recalcul mu, sigma
        mu_sim = resampled_df.mean() * 252
        sigma_sim = resampled_df.cov() * 252
        
        # Optimisation rapide (juste la frontière)
        _, frontier_sim = optimize_markowitz(mu_sim, sigma_sim)
        frontiers.append(frontier_sim)
        
    return frontiers
