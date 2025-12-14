import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.append(project_root)

from src.portfolio_lib import optimize_moo

st.set_page_config(page_title="Optimisation de Portefeuille", page_icon="📈", layout="wide")

# Fonctions de chargement de données
@st.cache_data
def load_data():

    data_dir = os.path.join(project_root, 'data', 'processed')
    data_path = os.path.join(data_dir, 'daily_returns.csv')
    sector_path = os.path.join(data_dir, 'sector_map.json')
    names_path = os.path.join(data_dir, 'ticker_names.json')
    
    # Check for data files
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
            
    asset_names = df.columns.tolist()

    return df, sector_map, ticker_names, asset_names

@st.cache_data
def load_initial_weights(asset_names):
    # Calcule w_prev à partir du CSV pour garantir l'alignement
    weights_path = os.path.join(project_root, 'data', 'processed', 'optimal_weights_level1.csv')
    
    if not os.path.exists(weights_path):
        return None
        
    # Chargement
    df_w = pd.read_csv(weights_path, header=None, names=['Ticker', 'Weight'])
    
    # Création d'un dictionnaire (Ticker: Weight)
    w_dict = pd.Series(df_w.Weight.values, index=df_w.Ticker).to_dict()
    
    # Alignement avec asset_names
    w_prev = np.zeros(len(asset_names))
    for i, ticker in enumerate(asset_names):
        w_prev[i] = w_dict.get(ticker, 0.0)
        
    return w_prev

@st.cache_data
def get_mu_sigma(df):
    mu = df.mean() * 252
    sigma = df.cov() * 252
    return mu, sigma

# Logique d'affichage

def show_demonstrator(mu, sigma, asset_names, sector_map, ticker_names):
    st.header("Démonstrateur Interactif")
    st.markdown("""
    Cette page permet d'explorer les solutions de compromis entre **Risque**, **Rendement** et **Coûts** (Niveau 2).
    """)
    
    # Configuration et sidebar
    with st.sidebar:
        st.subheader("Paramètres Optimisation")
        k_card = st.slider("Cardinalité Max (K)", 5, 30, 15)
        trans_cost_pct = st.number_input("Frais de transaction (%)", 0.0, 5.0, 0.5, step=0.1)
        trans_cost = trans_cost_pct / 100.0
        st.markdown("---")
        st.subheader("Contrainte Utilisateur")
        
        # Estimate min/max return for slider
        # Estimate min/max return for slider
        min_ret, max_ret = mu.min(), mu.max()
        # Conversion en pourcentage pour l'affichage
        min_pct, max_pct = min_ret * 100.0, max_ret * 100.0
        
        r_min_pct = st.slider("Rendement Minimal ($r_{min}$)", 
                              min_value=float(min_pct), 
                              max_value=float(max_pct), 
                              value=float(min_pct),
                              step=0.5, # Pas de 0.5%
                              format="%.2f%%")
        r_min = r_min_pct / 100.0
    
    # Bouton de calcul
    if st.button("Calculer le Front de Pareto (Niveau 2)") or "moo_results" in st.session_state:
        
        # Rerunn if parameters change or if results are not in session state
        param_key = f"{k_card}-{trans_cost}-{mu.shape[0]}"
        if "moo_results" not in st.session_state or st.session_state.get("moo_params") != param_key:
             with st.spinner(f"Calcul des solutions (NSGA-II) avec K={k_card}..."):
                # Chargement du w_prev (Portefeuille initial)
                w_prev = load_initial_weights(asset_names)
                
                # Optimisation
                res, final_weights_matrix = optimize_moo(mu, sigma, k_card, trans_cost, w_prev=w_prev, pop_size=60, n_gen=60)
                st.session_state["moo_results"] = (res, final_weights_matrix)
                st.session_state["moo_params"] = param_key
        
        res, final_weights_matrix = st.session_state["moo_results"]
        
        # Extraction des données
        returns = -res.F[:, 0]
        volatilities = np.sqrt(res.F[:, 1])
        costs = res.F[:, 2]
        
        st.info(f"Nombre de solutions trouvées (Global) : {len(returns)}")
        
        # 3D Visualization
        st.markdown("---")
        st.subheader("Global View: Front de Pareto 3D")
        
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=volatilities,
            y=costs,
            z=returns,
            mode='markers',
            marker=dict(
                size=6,
                color=returns,
                colorscale='Viridis',
                opacity=0.9,
                colorbar=dict(title='Rendement', tickformat='.1%')
            ),
            hovertemplate=(
                "<b>Rendement:</b> %{z:.2%}<br>" +
                "<b>Risque (Volatilité):</b> %{x:.2%}<br>" +
                "<b>Coût (f3):</b> %{y:.2%}<extra></extra>"
            )
        )])

        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Risque (Volatilité σ)',
                yaxis_title='Coûts de Transaction (f3)',
                zaxis_title='Rendement Espéré (E[R])',
                xaxis=dict(tickformat='.1%'),
                yaxis=dict(tickformat='.2%'),
                zaxis=dict(tickformat='.1%')
            ),
            height=600, # Balanced height
            margin=dict(l=0, r=0, b=0, t=10) # Minimize margins
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        
        # Selection & Filtering
        st.markdown("---")
        st.subheader("Analyse Détaillée & Sélection")

        # Filter Logic (r_min)
        valid_mask = returns >= r_min
        
        if not np.any(valid_mask):
            st.error(f"⚠️ Aucun portefeuille ne satisfait la contrainte de rendement minimal : {r_min:.2%}")
            return

        valid_indices = np.where(valid_mask)[0]
        
        # Dataframe for Selection
        df_valid = pd.DataFrame({
            'Index': valid_indices,
            'Rendement': returns[valid_indices],
            'Volatilité': volatilities[valid_indices],
            'Coût': costs[valid_indices]
        })
        
        # Sort by Volatility for logical selection
        df_valid_sorted = df_valid.sort_values(by="Volatilité")

        # Layout: Warning/Success + Selector
        col_sel_text, col_sel_input = st.columns([1, 2])
        
        with col_sel_text:
            st.success(f"**{len(valid_indices)}** portefeuilles éligibles (sur {len(returns)})")
        
        with col_sel_input:
            selected_idx_local = st.selectbox(
                "Sélectionnez un portefeuille optimal (classé par risque) :", 
                df_valid_sorted.index, 
                format_func=lambda i: f"Risque: {df_valid_sorted.loc[i, 'Volatilité']:.2%} | Renta: {df_valid_sorted.loc[i, 'Rendement']:.2%} | Coût: {df_valid_sorted.loc[i, 'Coût']:.2%}"
            )

        # Retrieve selected solution
        selected_global_idx = df_valid_sorted.loc[selected_idx_local, 'Index']
        w_selected = final_weights_matrix[selected_global_idx]
        
        # Dashboard (Metrics | Charts | Table)
        
        st.markdown("### 📊 Tableau de Bord du Portefeuille Sélectionné")
        
        # Key Metrics (Cards)
        # Using columns for KPI cards
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Rendement Espéré", f"{df_valid_sorted.loc[selected_idx_local, 'Rendement']:.2%}", delta_color="normal")
        kpi2.metric("Risque (Volatilité)", f"{df_valid_sorted.loc[selected_idx_local, 'Volatilité']:.2%}", delta_color="inverse") # Low risk is good
        kpi3.metric("Coûts Transaction", f"{df_valid_sorted.loc[selected_idx_local, 'Coût']:.2%}", delta_color="inverse") # Low cost is good
        
        st.markdown("---")

        # Prepare composition data
        s_selected = pd.Series(w_selected, index=asset_names)
        s_selected = s_selected[s_selected > 0.001] # Filter noise
        
        # Prepare Display Dataframes
        if ticker_names:
            names_list = [ticker_names.get(t, t) for t in s_selected.index]
        else:
            names_list = s_selected.index
            
        df_display = pd.DataFrame({
            'Ticker': s_selected.index,
            'Nom': names_list,
            'Poids': s_selected.values
        }).sort_values(by='Poids', ascending=False)
        
        df_display_fmt = df_display.copy()
        df_display_fmt['Poids'] = df_display_fmt['Poids'].apply(lambda x: f"{x*100:.2f}%")

        # Row B: Charts & Details
        # On peut faire 3 colonnes : Secteur | Actif | Tableau
        col_charts_sect, col_charts_asset, col_table_right = st.columns([1, 1, 1.2])
        
        common_height = 400
        
        with col_charts_sect:
            if sector_map:
                st.markdown("**Répartition Sectorielle**")
                sector_series = s_selected.index.map(sector_map).fillna("Unknown")
                sector_weights = pd.DataFrame({'Weight': s_selected.values, 'Sector': sector_series})
                sector_dist = sector_weights.groupby('Sector')['Weight'].sum().reset_index()
                
                fig_pie_sector = px.pie(sector_dist, values='Weight', names='Sector', hole=0.3)
                fig_pie_sector.update_layout(
                    margin=dict(l=10, r=10, t=10, b=0), 
                    height=common_height, 
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_pie_sector, use_container_width=True)
            else:
                 st.info("Secteurs non disponibles")

        with col_charts_asset:
            st.markdown("**Répartition par Actif (Top 10)**")
            fig_pie_assets = px.pie(df_display.head(10), values='Poids', names='Ticker', hole=0.3)
            fig_pie_assets.update_layout(
                margin=dict(l=10, r=10, t=10, b=0), 
                height=common_height, 
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie_assets, use_container_width=True)

        with col_table_right:
            st.markdown("**Composition Détaillée**")
            st.dataframe(
                df_display_fmt, 
                column_config={
                    "Ticker": st.column_config.TextColumn("Symbole"),
                    "Nom": st.column_config.TextColumn("Nom de l'actif", width="medium"),
                    "Poids": st.column_config.TextColumn("Poids (%)"),
                },
                hide_index=True, 
                use_container_width=True,
                height=common_height 
            )


def main():
    st.title("Projet Final : Optimisation de Portefeuille")
    st.markdown("""
    Cette application présente l'optimisation Multi-Objectifs et un Démonstrateur interactif pour la sélection de portefeuille.
    """)
    
    # Chargement des données
    df, sector_map, ticker_names, asset_names = load_data()
    if df is None:
        st.stop()
        
    mu, sigma = get_mu_sigma(df)
    
    # UI principal
    show_demonstrator(mu, sigma, asset_names, sector_map, ticker_names)

if __name__ == "__main__":
    main()