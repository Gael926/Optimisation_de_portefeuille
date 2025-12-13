
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add src to path just in case
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.portfolio_lib import optimize_markowitz, optimize_moo, resampling_efficient_frontier, get_rend_vol_sr

# --- CONFIG ---
st.set_page_config(page_title="Optimisation de Portefeuille", page_icon="📈", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Attempt to find the processed data
    # Assuming relative path from valid CWD or absolute path logic
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', 'processed', 'daily_returns.csv')
    
    if not os.path.exists(data_path):
        st.error(f"Fichier de données introuvable: {data_path}")
        return None
    
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    return df

@st.cache_data
def get_mu_sigma(df):
    mu = df.mean() * 252
    sigma = df.cov() * 252
    return mu, sigma

# --- UI APP ---

def main():
    st.title("📈 Projet Final : Optimisation de Portefeuille")
    
    df = load_data()
    if df is None:
        st.stop()
        
    mu, sigma = get_mu_sigma(df)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Accueil", "Niveau 1: Markowitz", "Niveau 2: Contraintes", "Niveau 3: Robustesse"])
    
    if page == "Accueil":
        show_home(mu, sigma)
    elif page == "Niveau 1: Markowitz":
        show_level_1(mu, sigma, df.columns)
    elif page == "Niveau 2: Contraintes":
        show_level_2(mu, sigma, df.columns)
    elif page == "Niveau 3: Robustesse":
        show_level_3(df)

def show_home(mu, sigma):
    st.markdown("""
    ### Bienvenue
    Cette application répond aux exigences du **Projet Final**. Elle implémente trois niveaux de complexité :
    
    1.  **Niveau 1 (Markowitz)** : Optimisation bi-critère (Rendement vs Risque) classique.
    2.  **Niveau 2 (Contraintes)** : Ajout de contraintes de cardinalité (nombre d'actifs) et coûts de transaction.
    3.  **Niveau 3 (Robustesse)** : Analyse de la stabilité des portefeuilles via ré-échantillonnage (Bootstrap).
    """)
    
    st.subheader("Aperçu des Données")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Nombre d'actifs : {len(mu)}")
        st.write("Top 5 Rendements (Mu):")
        st.dataframe(mu.sort_values(ascending=False).head(5))
        
    with col2:
        st.info("Matrice de Covariance (Aperçu) :")
        st.dataframe(sigma.iloc[:5, :5])

def show_level_1(mu, sigma, asset_names):
    st.header("Niveau 1 : Modèle de Markowitz")
    st.markdown("Optimisation du couple **Rendement / Risque** sans contraintes complexes (juste budget et long-only).")
    
    if st.button("Lancer l'Optimisation (Markowitz)"):
        with st.spinner("Calcul de la frontière en cours..."):
            w_sharpe, (eff_vols, eff_rends) = optimize_markowitz(mu, sigma)
            
            # Recuperer métriques du portefeuille optimal
            rend_s, vol_s, sr_s = get_rend_vol_sr(w_sharpe, mu, sigma)
            
            # --- PLOT FRONTIER ---
            fig = go.Figure()
            
            # Frontière
            fig.add_trace(go.Scatter(x=eff_vols, y=eff_rends, mode='lines', name='Frontière Efficiente'))
            
            # Point Optimal
            fig.add_trace(go.Scatter(x=[vol_s], y=[rend_s], mode='markers', 
                                     marker=dict(size=15, color='red', symbol='star'),
                                     name=f'Max Sharpe (SR={sr_s:.2f})'))
            
            fig.update_layout(title="Frontière de Pareto", xaxis_title="Volatilité (Risque)", yaxis_title="Rendement Espéré")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- PLOT POIDS ---
            s_weights = pd.Series(w_sharpe, index=asset_names)
            s_weights = s_weights[s_weights > 0.01].sort_values(ascending=False) # Filter small weights
            
            st.subheader("Composition du Portefeuille Optimal")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(s_weights)
            with col2:
                fig_pie = px.pie(values=s_weights.values, names=s_weights.index, title="Poids des Actifs (>1%)")
                st.plotly_chart(fig_pie, use_container_width=True)

def show_level_2(mu, sigma, asset_names):
    st.header("Niveau 2 : Contraintes de Cardinalité et Coûts")
    st.markdown("""
    Ici, nous utilisons un **Algorithme Génétique (NSGA-II)** car le problème devient non-convexe.
    - **Cardinalité** : On limite le nombre de lignes actives.
    - **Coûts** : On pénalise les rotations de portefeuille.
    """)
    
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        k_card = st.slider("Nombre d'actifs Max (Cardinalité)", 2, 20, 10)
    with col_conf2:
        trans_cost = st.number_input("Coût de transaction (%)", 0.0, 5.0, 0.5, step=0.1) / 100.0
        
    pop_size = st.slider("Taille Population (NSGA-II)", 20, 200, 50)
    n_gen = st.slider("Nombre Générations", 10, 200, 50)
    
    if st.button("Lancer NSGA-II"):
        with st.spinner("Evolution génétique en cours..."):
            res, final_weights_matrix = optimize_moo(mu, sigma, k_card, trans_cost, pop_size=pop_size, n_gen=n_gen)
            
            # Extract Objectives from result
            # res.F cols: 0 -> -f1 (neg return), 1 -> f2 (variance), 2 -> f3 (cost)
            returns = -res.F[:, 0]
            volatilities = np.sqrt(res.F[:, 1])
            costs = res.F[:, 2]
            
            # --- 3D PLOT ---
            st.subheader("Front de Pareto 3D")
            fig_3d = px.scatter_3d(x=volatilities, y=costs, z=returns,
                                   color=returns,
                                   labels={'x':'Volatilité', 'y':'Coûts', 'z':'Rendement'},
                                   title="3 Objectifs : Risque vs Coûts vs Rendement")
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # --- 2D PROJECTION ---
            st.subheader("Projection 2D (Rendement vs Risque)")
            fig_2d = px.scatter(x=volatilities, y=returns, color=costs,
                                labels={'x':'Volatilité', 'y':'Rendement', 'color':'Coût'},
                                title="Nuage de solutions NSGA-II")
            st.plotly_chart(fig_2d, use_container_width=True)
            
            # Show a sample portfolio
            st.write("Exemple d'un portefeuille trouvé (le meilleur rendement):")
            best_idx = np.argmax(returns)
            w_best = final_weights_matrix[best_idx]
            s_best = pd.Series(w_best, index=asset_names)
            s_best = s_best[s_best > 0.001].sort_values(ascending=False)
            st.dataframe(s_best.head(10))

def show_level_3(df):
    st.header("Niveau 3 : Analyse de Robustesse")
    st.markdown("""
    L'optimisation de Markowitz est très sensible aux erreurs d'estimation (\"Error Maximizer\").
    Ici, nous utilisons le **Ré-échantillonnage (Resampling)** pour générer plusieurs frontières efficientes probables et visualiser l'incertitude.
    """)
    
    n_sims = st.slider("Nombre de simulations (Bootstrap)", 10, 100, 20)
    
    if st.button("Lancer le Test de Robustesse"):
        with st.spinner(f"Génération de {n_sims} frontières..."):
            mu_orig, sigma_orig = get_mu_sigma(df)
            
            # Original Frontier
            _, (vols_orig, rends_orig) = optimize_markowitz(mu_orig, sigma_orig)
            
            # Resampled Frontiers
            frontiers = resampling_efficient_frontier(df, n_simulations=n_sims)
            
            # Plot
            fig = go.Figure()
            
            # Add all simulated frontiers as light lines
            for i, (v, r) in enumerate(frontiers):
                fig.add_trace(go.Scatter(x=v, y=r, mode='lines', 
                                         line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
                                         showlegend=False))
            
            # Add Original as strong line
            fig.add_trace(go.Scatter(x=vols_orig, y=rends_orig, mode='lines',
                                     line=dict(color='blue', width=3),
                                     name='Frontière Nominale (Données Réelles)'))
            
            fig.update_layout(title="Analyse de Robustesse (Nuage de Frontières)",
                              xaxis_title="Volatilité", yaxis_title="Rendement Espéré")
            
            st.plotly_chart(fig, use_container_width=True)
            st.success("Le nuage gris représente l'incertitude statistique autour de la frontière réelle.")

if __name__ == "__main__":
    main()
