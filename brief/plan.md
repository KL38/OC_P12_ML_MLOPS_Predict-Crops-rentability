# Plan de travail — P12 Agritech Answers

> **Version de travail unique.** La seule source d'exigences est `brief.md`. Les documents
> `plan_P12.md` et `methodo_etape2.md` sont archivés dans `OLD/` : ils contiennent des
> décisions périmées (filtre p5/p95, terme « regret », `Year` figée à 2013, intervalles
> MAPIE) qui ne correspondent plus à ce qui est implémenté. Ne pas s'y référer.

---

## Où on en est

| Étape du brief | État | Ce qui existe |
|---|---|---|
| **1** — Explorer, fusionner, préparer | ✅ | `notebooks/EDA CYPD.ipynb`, dataset consolidé, ACP |
| **2** — Construire et optimiser le modèle | ✅ | `notebooks/ML CYPD.ipynb`, 14 runs MLflow, registre `crop_yield_recommender` v1 `@champion` |
| **3** — API de prédiction | ✅ | `src/api.py` (3 routes), `Dockerfile`, `requirements.txt` |
| **4** — Interface Streamlit | ✅ | `app/app.py` (2 onglets), thème `.streamlit/config.toml`, `app/Dockerfile` |
| **5** — Tests et déploiement automatisés | 🟡 | `ci.yml` : `lint` · `tests` · `build` — **manque le README (doc + badges)** |

**Démo** : `docker compose up --build` → http://localhost:8501. Aucun secret, aucun compte
externe, aucune dépendance réseau.

**Modèle retenu** : HistGradientBoosting sur `log(rendement)`, protocole `GroupKFold` par pays.
CV R² 0,595 · test R² 0,627 · RMSE 4,94 t/ha · MAE 2,85 sur 22 pays jamais vus.

**Tests** : 26, dont 1 skip légitime (`df.csv` non versionné).

---

## Ce qui reste, dans l'ordre

### 1. ~~Valider le front~~ ✅

### 2. ~~Conteneurisation~~ ✅

`Dockerfile` multi-étage (uv épinglé, non-root, `${PORT:-8000}`, healthcheck),
`app/Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements.txt`.

`pyproject.toml` a été redécoupé pour l'occasion : `[project]` ne porte plus que les
6 dépendances du runtime de l'API, le reste vit dans les groupes `notebooks`, `app` et
`dev`, avec `default-groups` pour que le poste de travail ne change pas. Sans ce découpage
l'image embarquerait catboost (330 Mo), mlflow, jupyter et matplotlib.

### 3. ~~Livraison continue~~ — écartée

Le brief est explicite : le **Build est obligatoire**, le **Déploiement est *« optionnel
mais fortement recommandé »***. Décision prise : **pas de cloud**, la démo est le
`docker compose up` local. Le job `build` de `ci.yml` couvre donc l'exigence, sans registre,
sans compte externe et **sans un seul secret**.

Si le sujet revient : `ghcr.io` permettrait de publier l'image sans créer de compte ni
saisir de secret (le `GITHUB_TOKEN` d'Actions y suffit). Docker Hub imposerait un compte
et deux secrets.

Conséquence : le correctif `st.secrets` pour `API_URL` **n'a plus lieu d'être** — Streamlit
Community Cloud en était la seule justification.

### 4. README *(livrable noté, actuellement vide)*

Le brief exige explicitement : *« documentation incluant description des workflows,
triggers, **badges de statut** »*. À écrire :

- badge de statut de la CI ;
- schéma du pipeline (train → registre → image → déploiement) ;
- description des deux workflows, de leurs déclencheurs et de leur périmètre ;
- comment lancer en local (API, front, tests).

### 5. Déploiement

- **Front** → Streamlit Community Cloud. Entrypoint `app/app.py` ; `app/requirements.txt`
  est trouvé en premier, donc l'installation reste minimale. `API_URL` à passer en secret.
- **API** → Render ou HF Spaces depuis l'image Docker Hub.

### 6. Livrables de restitution

- **Rapport métier `.pdf`** — résultats, variables clés, recommandations agronomiques,
  captures MLflow.
- **Screenshots MLflow `.png`** — runs annotés. Trois méritent une note manuelle : le
  modèle retenu, le plus grand écart groupé/aléatoire, le run temporel.
- **Support de soutenance** — 15 min de présentation, 10 min de discussion.

---

## Décisions actées, à ne pas rejouer

| Sujet | Décision |
|---|---|
| Tri de `/recommend` | L'**API** trie par rendement décroissant (Étape 3 du brief) ; **Streamlit** reclasse par marge (objectif du mail de Gabriel). Les deux exigences sont satisfaites. |
| Prix et coûts | Hors de l'API. Table éditable côté front, un exploitant n'a pas les coûts de son voisin. Modifier un prix ne rappelle pas le modèle. |
| `Year` | Année réelle transmise, jamais figée. Un arbre sature sur sa dernière coupure : 2013 et 2026 donnent le même chiffre. Réserve affichée en permanence. |
| Domaine de validité | min/max observés sur le **train**, élargis de 10 % — multiplicatif pour la pluie, additif (10 % de l'amplitude) pour la température. Filtre large et assumé : écarter le manioc en Russie, pas coller au train. |
| Fiabilité par culture | **Pas affichée dans l'UI.** La réserve est portée par le rapport et la soutenance. |
| Validation des entrées | Pydantic refuse l'impossible physiquement (422) ; le domaine signale l'invraisemblable (`cultivable: false`). Deux mécanismes distincts. |
| Sérialisation | `joblib` + `ClippedExp` dans `src/preprocessing.py`. Une fonction de notebook serait picklée comme `__main__._clipped_exp` et casserait au déploiement seulement. |
| Modèle versionné | `models/model_B.joblib` est dans git (217 Ko) : l'image Docker le copie. Son absence est un échec de test, pas un skip. |
| Tests du front | **Aucun, volontairement.** Le brief ne demande des tests unitaires que pour l'API (Étape 5). `app/` est linté par la CI mais pas exécuté ; sa seule logique métier est le calcul de marge de `_classement`. |

---

## Reliquats de revue, non traités

- `temporel` défini deux fois dans `ML CYPD.ipynb`.
- Ancres de sommaire en double dans le notebook.
- Paramètre `features` de `run_cv`, ajouté pour la Phase B qui a été écartée.
- **Restart & Run All** du notebook avant remise.
- Champs en trop acceptés en silence par Pydantic (`extra="forbid"` si on veut durcir).
- Docstring de `/health` : elle annonce que l'app y lit sa réserve, or l'app porte son
  propre texte, plus long et orienté produit. À reformuler ou à faire converger.
