# Utiliser une image Python officielle et légère
FROM python:3.12-slim

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code de l'application
COPY . .

# Préparer les données (générer les CSV/JSON si nécessaire)
# Cela assure que l'image est autonome même si les notebooks n'ont pas tourné
RUN python src/prepare_data.py

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande de démarrage
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.serverAddress=localhost"]
