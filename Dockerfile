# Image de l'API. Multi-étage : l'environnement se construit dans une image jetable, et
# seul le résultat part dans l'image finale — ni uv, ni cache de build, ni fichiers de
# projet inutiles au service.

FROM python:3.12-slim AS constructeur

# Version épinglée plutôt que `:latest`, pour la même raison que setup-uv dans la CI :
# une image reproductible ne peut pas dépendre d'un tag mouvant.
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /bin/uv

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# --no-default-groups : seules les dépendances de [project] sont installées, c'est-à-dire
# le strict nécessaire pour servir le modèle. Sans ce drapeau, l'image embarquerait
# catboost, mlflow, jupyter et matplotlib.
#
# --locked : échoue si uv.lock ne correspond plus à pyproject.toml. Une image construite
# sur une résolution différente de celle testée en CI ne prouverait rien.
#
# Le montage bind évite de graver pyproject.toml et uv.lock dans une couche : ils ne
# servent qu'à cette commande.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups


FROM python:3.12-slim

COPY --from=constructeur /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Le contrat partagé avec le notebook, puis les deux artefacts que l'API charge au
# démarrage. Le modèle est versionné dans git précisément pour cette copie : sans lui,
# `lifespan` lève et le conteneur refuse de servir.
COPY src/ src/
COPY models/model_B.joblib models/domaine_validite.json models/

# Un service qui n'écrit rien n'a aucune raison de tourner en root.
RUN useradd --create-home --uid 1000 agritech && chown -R agritech /srv
USER agritech

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# `${PORT:-8000}` parce que Render et HF Spaces imposent leur port par variable
# d'environnement. `exec` pour qu'uvicorn devienne PID 1 et reçoive bien le SIGTERM
# d'arrêt au lieu de se faire tuer au bout du délai de grâce.
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
