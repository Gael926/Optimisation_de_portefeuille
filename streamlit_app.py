import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import json

# ====================================================================
# CONFIGURATION ET CHEMINS
# ====================================================================

# Ajout du répertoire courant au path pour les imports locaux
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assurez-vous que le répertoire parent de 'src' est bien dans le path si les fonctions sont dans 'src/portfolio_lib.py'
project_root = current_dir
if project_root not in sys.path:
    sys.path.append(project_root)


# L'import doit correspondre à votre structure de fichier
# Assurez-vous que 'portfolio_lib.py' est accessible et contient ces fonctions.
# L'import doit correspondre à votre structure de fichier
# Assurez-vous que 'portfolio_lib.py' est accessible et contient ces fonctions.
# from src.portfolio_lib import optimize_markowitz, optimize_moo, resampling_efficient_frontier, get_rend_vol_sr
# REMPLACEMENT TEMPORAIRE : On simule l'existence des fonctions pour que le code tourne
# Vous devez réactiver l'import réel et vous assurer que ces fonctions renvoient les structures de données attendues.
def optimize_moo(mu, sigma, k_card, trans_cost, pop_size=60, n_gen=60):
    # Simulation d'un objet 'res' et d'une matrice de poids pour le test
    N_solutions = 200
    returns = np.random.uniform(0.05, 0.15, N_solutions)
    volatilities_sq = (0.08 + 0.5 * returns + np.random.normal(0, 0.02, N_solutions))**2
    costs = np.random.uniform(0.0005, 0.005, N_solutions)
    
    # Structure de retour similaire à la librairie d'optimisation (e.g., pymoo)
    class MockRes:
        F = np.column_stack([-returns, volatilities_sq, costs]) # f1=-R, f2=Vol^2, f3=Cost
        
    res = MockRes()
    
    # Simulation de la matrice de poids AVEC K_CARD (Sparsity)
    n_assets = len(mu)
    final_weights_matrix = np.zeros((N_solutions, n_assets))
    
    for i in range(N_solutions):
        # Generate random weights
        raw_w = np.random.rand(n_assets)
        # Apply Top K constraint (Sparsity)
        # Identify indices of the top k_card elements
        idx = np.argpartition(raw_w, -k_card)[-k_card:]
        
        # Create sparse vector
        sparse_w = np.zeros(n_assets)
        sparse_w[idx] = raw_w[idx]
        
        # Normalize
        if sparse_w.sum() > 0:
            sparse_w /= sparse_w.sum()
            
        final_weights_matrix[i] = sparse_w
    
    return res, final_weights_matrix

def optimize_markowitz(mu, sigma, N_points=100):
    # Simulation du Front de Pareto 2D de Markowitz
    rets = np.linspace(mu.min(), mu.max(), N_points)
    vols = np.sqrt(0.05 + rets * 0.5) + np.random.normal(0, 0.01, N_points)
    
    return pd.DataFrame({'Rendement': rets, 'Volatilité': vols})




st.set_page_config(page_title="Optimisation de Portefeuille", page_icon="📈", layout="wide")

# ====================================================================
# FONCTIONS DE CHARGEMENT DE DONNÉES (CORRIGÉES POUR UTILISER LE CHEMIN)
# ====================================================================

@st.cache_data
def load_data():

    # Utilisation du chemin project_root pour plus de robustesse
    data_dir = os.path.join(project_root, 'data', 'processed')
    data_path = os.path.join(data_dir, 'daily_returns.csv')
    sector_path = os.path.join(data_dir, 'sector_map.json')
    names_path = os.path.join(data_dir, 'ticker_names.json')
    
    if not os.path.exists(data_path):
        st.error(f"Fichier de données introuvable: {data_path}")
        return None, None, None, None
    
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    sector_map = {}
    if os.path.exists(sector_path):
        with open(sector_path, 'r') as f:
            sector_map = json.load(f)
            
    ticker_names = {}
    if os.path.exists(names_path):
        with open(names_path, 'r') as f:
            ticker_names = json.load(f)
            
    # S'assurer que les actifs ont des noms (si ticker_names est vide)
    asset_names = df.columns.tolist()

    return df, sector_map, ticker_names, asset_names

@st.cache_data
def get_mu_sigma(df):
    mu = df.mean() * 252
    sigma = df.cov() * 252
    return mu, sigma

# ====================================================================
# LOGIQUE D'AFFICHAGE DU DEMONSTRATEUR
# ====================================================================

def show_demonstrator(mu, sigma, asset_names, sector_map, ticker_names):
    st.header("Démonstrateur Interactif")
    st.markdown("""
    Cette page permet d'explorer les solutions de compromis entre **Risque**, **Rendement** et **Coûts** (Niveau 2).
    """)
    
    # 1. Configuration et SIDEBAR (Contrainte Utilisateur)
    with st.sidebar:
        st.subheader("Paramètres Optimisation")
        k_card = st.slider("Cardinalité Max (K)", 5, 30, 15)
        trans_cost_pct = st.number_input("Frais de transaction (%)", 0.0, 5.0, 0.5, step=0.1)
        trans_cost = trans_cost_pct / 100.0
        st.markdown("---")
        st.subheader("Contrainte Utilisateur")
        
        # Estimate min/max return for slider
        min_ret, max_ret = mu.min(), mu.max()
        r_min = st.slider("Rendement Minimal ($r_{min}$)", float(min_ret), float(max_ret), float(min_ret), format="%.4f")
    
    # 2. Bouton de calcul
    if st.button("Calculer le Front de Pareto (Niveau 2)") or "moo_results" in st.session_state:
        
        # Rerunn if parameters change or if results are not in session state
        param_key = f"{k_card}-{trans_cost}-{mu.shape[0]}"
        if "moo_results" not in st.session_state or st.session_state.get("moo_params") != param_key:
             with st.spinner(f"Calcul des solutions (NSGA-II) avec K={k_card}..."):
                # Simulation de l'appel réel
                res, final_weights_matrix = optimize_moo(mu, sigma, k_card, trans_cost, pop_size=60, n_gen=60)
                # res, final_weights_matrix = optimize_moo(mu, sigma, k_card, trans_cost) # Utilise la fonction mock
                st.session_state["moo_results"] = (res, final_weights_matrix)
                st.session_state["moo_params"] = param_key
        
        res, final_weights_matrix = st.session_state["moo_results"]
        
        # Extract data
        returns = -res.F[:, 0]
        volatilities = np.sqrt(res.F[:, 1])
        costs = res.F[:, 2]
        
        st.info(f"Nombre de solutions trouvées (Global) : {len(returns)}")
        
        # --- Onglets pour séparer les niveaux ---
        tab_global, tab_selection = st.tabs(["Front de Pareto 3D (Global)", "Sélection et Analyse (r_min)"])

        with tab_global:
            # --- 3D PLOT (GLOBAL) : CORRECTION MAJEURE: UTILISATION DE go.Figure ---
            st.subheader("Front de Pareto 3D (Niveau 2)")

            fig_3d = go.Figure(data=[go.Scatter3d(
                x=volatilities,
                y=costs,
                z=returns,
                mode='markers',
                marker=dict(
                    size=5,
                    color=returns,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Rendement')
                ),
                hovertemplate=(
                    "<b>Rendement:</b> %{z:.4f}<br>" +
                    "<b>Risque (Volatilité):</b> %{x:.4f}<br>" +
                    "<b>Coût (f3):</b> %{y:.4f}<extra></extra>"
                )
            )])

            fig_3d.update_layout(
                title="3 Objectifs : Risque vs Coûts vs Rendement",
                scene=dict(
                    xaxis_title='Risque (Volatilité σ)',
                    yaxis_title='Coûts de Transaction (f3)',
                    zaxis_title='Rendement Espéré (E[R])'
                ),
                height=700,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with tab_selection:
            # 2. Filter by r_min
            valid_mask = returns >= r_min
            
            if not np.any(valid_mask):
                st.warning(f"Aucun portefeuille ne satisfait $r_{{min}} >= {r_min:.4f}$. Veuillez réduire la contrainte.")
                return

            # Indices of valid solutions
            valid_indices = np.where(valid_mask)[0]
            
            st.success(f"{len(valid_indices)} portefeuilles trouvés respectant $r_{{min}}$.")
            
            # Making a dataframe for the selector
            df_valid = pd.DataFrame({
                'Index': valid_indices,
                'Rendement': returns[valid_indices],
                'Volatilité': volatilities[valid_indices],
                'Coût': costs[valid_indices]
            })
            
            # 3. Selection & Analysis
            
            # Selection Slider: Sort by Volatility (Risk) to present a logical compromise
            df_valid_sorted = df_valid.sort_values(by="Volatilité")
            
            st.subheader("Sélection et Compromis")
            
            selected_idx_local = st.selectbox(
                "Choisir un portefeuille (classé par risque croissant) :", 
                df_valid_sorted.index, 
                format_func=lambda i: f"Volatilité: {df_valid_sorted.loc[i, 'Volatilité']:.4f} | Rendement: {df_valid_sorted.loc[i, 'Rendement']:.4f}"
            )
            
            # Get the global index back
            selected_global_idx = df_valid_sorted.loc[selected_idx_local, 'Index']
            
            # Retrieve Weights
            w_selected = final_weights_matrix[selected_global_idx]
            
            # 4. Affichage des Métriques et Composition
            s_selected = pd.Series(w_selected, index=asset_names)
            s_selected = s_selected[s_selected > 0.001] # Filter small weights
            formatted_weights = [f"{poids*100:.2f}%" for poids in s_selected.values]

            # Map Tickers to Names
            if ticker_names:
                names_list = [ticker_names.get(t, t) for t in s_selected.index]
                names_series = pd.Index(names_list)
            else:
                names_series = s_selected.index

            
            df_composition = pd.DataFrame({
                'Ticker': s_selected.index,
                'Nom': names_series,
                'Poids': s_selected.values
            }).sort_values(by='Poids', ascending=False)
            
            # Create a formatted copy for display
            df_display = df_composition.copy()
            df_display['Poids'] = df_display['Poids'].apply(lambda x: f"{x*100:.2f}%")
            
            col_metrics, col_info = st.columns([1, 1])
            
            with col_metrics:
                st.subheader("Métriques & Répartition")
                st.metric("Rendement Attendu", f"{df_valid_sorted.loc[selected_idx_local, 'Rendement']:.3%}")
                st.metric("Risque (Volatilité)", f"{df_valid_sorted.loc[selected_idx_local, 'Volatilité']:.3f}")
                st.metric("Coûts de Transaction", f"{df_valid_sorted.loc[selected_idx_local, 'Coût']:.4f}")

                # --- 5. Répartition Sectorielle (Moved here) ---
                if sector_map:
                    sector_series = s_selected.index.map(sector_map).fillna("Unknown")
                    # Group by sector
                    sector_weights = pd.DataFrame({'Weight': s_selected.values, 'Sector': sector_series})
                    sector_dist = sector_weights.groupby('Sector')['Weight'].sum().reset_index()
                    
                    st.markdown("#### Répartition Sectorielle")
                    fig_pie_sector = px.pie(sector_dist, values='Weight', names='Sector', height=300)
                    fig_pie_sector.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                    st.plotly_chart(fig_pie_sector, use_container_width=True)
                else:
                    st.warning("Aucune donnée sectorielle disponible.")

                st.markdown("#### Répartition par Actif")
                fig_pie_assets = px.pie(df_composition, values='Poids', names='Ticker', height=300)
                fig_pie_assets.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                st.plotly_chart(fig_pie_assets, use_container_width=True)

            with col_info:
                st.subheader("Composition du Portefeuille")
                st.dataframe(df_display, hide_index=True, use_container_width=True)


def main():
    st.title("Projet Final : Optimisation de Portefeuille")
    st.markdown("""
    Cette application présente l'optimisation Multi-Objectifs et un Démonstrateur interactif pour la sélection de portefeuille.
    """)
    
    # CHARGEMENT DES DONNÉES DE BASE
    df, sector_map, ticker_names, asset_names = load_data()
    if df is None:
        st.stop()
        
    mu, sigma = get_mu_sigma(df)
    
    # --- UI PRINCIPALE (ON PAGE) ---
    show_demonstrator(mu, sigma, asset_names, sector_map, ticker_names)

if __name__ == "__main__":
    main()