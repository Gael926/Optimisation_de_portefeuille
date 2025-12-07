import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Construction des chemins absolus basés sur l'emplacement de ce fichier (src/data_prep.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
raw_data = os.path.join(project_root, 'data', 'raw')
processed_data = os.path.join(project_root, 'data', 'processed')

def all_sectors():
    all_files = glob.glob(os.path.join(raw_data, "*.csv"))
    
    if not all_files:
        print("Aucun fichier CSV trouvé !")
        return None, None

    df_list = []
    sector_map = {} # Pour garder en mémoire quel actif appartient à quel secteur

    for filename in all_files:
        # On récupère le nom du secteur
        sector_name = os.path.basename(filename).replace('.csv', '')
        
        # On lit les fichiers CSV en mettant la Date en index
        df_temp = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
        
        # Nettoyage immédiat des lignes vides (jours fériés) pour éviter les erreurs
        df_temp = df_temp.dropna(how='all')
        
        # On s'assure que ce sont des nombres
        df_temp = df_temp.apply(pd.to_numeric, errors='coerce')
        
        # On sauvegarde le secteur de chaque actif
        for asset in df_temp.columns:
            # On ignore les colonnes parasites 'Unnamed'
            if "Unnamed" not in asset:
                sector_map[asset] = sector_name
            
        df_list.append(df_temp)

    # On concatène tous les df
    full_price_data = pd.concat(df_list, axis=1)

    # On nettoie des données manquantes
    full_price_data = full_price_data.dropna(how='all')

    # Export du datafarame concat en csv
    output_csv = os.path.join(processed_data, 'all_sectors.csv')
    full_price_data.to_csv(output_csv)

    # Export sector map en json
    with open(os.path.join(processed_data, 'sector_map.json'), 'w') as f:
        json.dump(sector_map, f)

    print(f"Fichier all_sectors.csv sauvegardé")
    print(f"Nombre total de lignes : {len(full_price_data)}")
    
    return full_price_data

def all_sectors_with_sector():
    all_data = []
    
    files = [f for f in os.listdir(raw_data) if f.endswith('.csv')]
    
    if not files:
        print("Aucun fichier CSV trouvé !")
        return None

    for file in files:
        file_path = os.path.join(raw_data, file)
        
        # Lecture simple sans index pour le melt
        df = pd.read_csv(file_path)
        
        # Suppression des colonnes "Unnamed" qui cassaient le melt
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        df_melted = df.melt(id_vars=['Date'], var_name='Ticker', value_name='Close')
        
        # Ajout de la colonne Secteur
        df_melted['Sector'] = file.replace('.csv', '')
        
        # On force le type numérique et on supprime les lignes vides
        df_melted['Close'] = pd.to_numeric(df_melted['Close'], errors='coerce')
        df_melted = df_melted.dropna(subset=['Close'])
        
        all_data.append(df_melted)

    # Concatenation de tous les dataframes
    master_df = pd.concat(all_data, ignore_index=True)

    # Conversion en datetime
    master_df['Date'] = pd.to_datetime(master_df['Date'])

    # Tri par Date et Ticker
    master_df = master_df.sort_values(['Date', 'Ticker'])

    # Sauvegarde du fichier concaténé
    output_path = os.path.join(processed_data, 'all_sectors_with_sector.csv')
    master_df.to_csv(output_path, index=False)

    print(f"Fichier all_sectors_with_sector.csv sauvegardé")
    print(f"Nombre total de lignes : {len(master_df)}")
    
    return master_df