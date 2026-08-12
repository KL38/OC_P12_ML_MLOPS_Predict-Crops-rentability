# Image du front. Le contexte de build est la RACINE du dépôt, pas ce dossier :
#   docker build -f app/Dockerfile .
#
# pip et non uv, contrairement à l'API : le front a déjà un requirements.txt minimal,
# celui-là même que lit Streamlit Community Cloud. Une seule liste pour les deux cibles
# de déploiement vaut mieux que deux listes à garder d'accord.

FROM python:3.12-slim

WORKDIR /srv

# Couche séparée et copiée en premier : tant que les dépendances ne bougent pas, Docker
# réutilise cette couche et ne réinstalle rien quand seul le code de l'app change.
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le thème vit à la racine du dépôt, mais Streamlit le cherche dans le répertoire courant :
# dans l'image, il doit donc être à côté de app.py.
COPY .streamlit/ .streamlit/
COPY app/ .

RUN useradd --create-home --uid 1000 agritech && chown -R agritech /srv
USER agritech

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
