# Brief de mission — Prédiction de rendements & moteur de recommandation de culture

> **Contexte** — Vous êtes **Data Scientist Machine Learning junior** dans l'entreprise **Agritech Answers**, spécialisée dans l'optimisation agricole et l'innovation agrotechnologique.

![Logo Agritech Answers](logo%20agritech%20answers.png)

---

## 📨 Message Slack de Gabriel (Lead Data Scientist)

> Comme discuté, je voulais te donner des informations pour ta future mission. Nous avons besoin que tu travailles sur la **prédiction des rendements agricoles** et sur la mise en place d'un **moteur de recommandation de culture**. L'idée est de tirer parti des données sur les pratiques de culture, des données agroclimatiques et des historiques de rendement.
>
> Ton objectif ? **Modéliser la relation entre les espèces cultivées, les coûts de production et les profits** afin de maximiser les rendements et optimiser l'utilisation des ressources.
>
> Je t'envoie dans un mail cet après-midi les deux jeux de données avec lesquels tu travailleras, et je te donnerai les détails de la mission.

---

## ✉️ Mail de Gabriel — Détails de la mission

| | |
|---|---|
| **De** | Gabriel |
| **À** | Moi |
| **Objet** | Détails de la mission |
| **PJ** | voir dossier `/data` |

Hello !

Comme échangé ce matin par téléphone, voici les détails concernant ta mission à venir.

### Objectif

Développer une **application web simple et intuitive** pour aider nos clients agriculteurs à prendre de meilleures décisions. L'application doit remplir **deux fonctions complémentaires**, au sein d'une même interface :

- **Fonction de prédiction** — Permettre à un utilisateur de sélectionner une culture spécifique, de renseigner les conditions de sa parcelle (température, usage de pesticides, etc.) et d'obtenir une **estimation chiffrée du rendement attendu**. C'est notre moteur d'analyse de base.
- **Fonction de recommandation** — C'est là que réside notre plus grande valeur ajoutée. L'utilisateur renseigne **uniquement** les conditions de sa parcelle, et l'application lui recommande **la culture la plus rentable** en simulant le rendement pour toutes les cultures possibles et en affichant un classement.

### Architecture cible (moderne et découplée)

- **Back-end** — Une **API** (*Application Programming Interface*) qui contiendra notre modèle de Machine Learning entraîné. Elle exposera des *endpoints* (URLs spécifiques) pour la prédiction et la recommandation.
- **Front-end** — Une application web interactive développée avec **Streamlit**. Cette interface, simple et visuelle, permettra aux agriculteurs d'interroger notre API sans jamais voir une ligne de code.

Tu seras en charge de **l'ensemble du flux de travail**, depuis la préparation des données jusqu'à la mise en production d'un prototype fonctionnel.

### Attendus principaux

- **Pipeline MLOps & CI/CD robuste**, garantissant qualité, reproductibilité et évolutivité du système.
- **Fusion et analyse de deux jeux de données distincts** :
  - **Agriculture CropYield Dataset** — Données historiques de rendement selon les cultures et les régions. Une **analyse en composantes principales (ACP)** est attendue pour identifier les variables clés. *Merci de les expliciter clairement.*
  - **CropYield Prediction Dataset** — Données agronomiques et climatiques annuelles (usage de pesticides, fertilisants, température moyenne, etc.) utilisées pour **valider** l'analyse précédente.

> Tu es **libre de choisir la manière d'exploiter ces deux jeux de données** pour construire ton modèle, en fonction de ce qui te semble le plus pertinent d'un point de vue analytique et métier. L'important est de **justifier clairement tes choix** et d'optimiser la performance et l'interprétabilité du modèle.

### Livrables demandés

- **Notebooks ou scripts** contenant l'intégralité du pipeline d'entraînement des modèles (fusion, preprocessing, modélisation).
- **Un rapport métier synthétique** (`.pdf`), rédigé de façon claire et accessible à un public non technique, contenant :
  - Les résultats principaux et recommandations agronomiques ;
  - Les explications des variables clés ;
  - Des **captures d'écran de l'interface MLflow** montrant les résultats expérimentaux (suivi des runs, métriques, paramètres…).

Si tu rencontres des blocages, n'hésite pas à me contacter pour faire un point.

Bon courage pour cette mission. J'ai hâte de découvrir tes résultats !

*Gabriel — Lead Data Scientist*

---

# 🗺️ Mission guidée — Étapes

> Cette mission est guidée. Vous pouvez suivre les étapes ci-dessous.

## ÉTAPE 1 — Explorez, fusionnez et préparez les jeux de données

### Description

Vous commencez par explorer les deux jeux de données fournis (*CropYield Prediction Dataset* et *Agriculture CropYield Dataset*). L'objectif principal de cette étape est de **créer un dataset unifié** en enrichissant les données de rendement par culture (issues de *Agriculture CropYield Dataset*) avec les facteurs agronomiques et climatiques plus généraux (issus de *CropYield Prediction Dataset*). Vous devrez identifier les prétraitements nécessaires (nettoyage, transformation, encodage, etc.) pour chaque dataset avant de les combiner.

### Prérequis

- Avoir téléchargé les jeux de données.
- Avoir exploré les colonnes, types de données et valeurs manquantes de chaque dataset.
- Avoir identifié une ou plusieurs **clés de jointure** potentielles (ex. : pays, année) pour fusionner les deux datasets, et identifié des variables pouvant servir de **proxy** (irrigation / fertilisation).
- Avoir réalisé une première analyse exploratoire pour chaque source de données.

### Résultats attendus

- Un **notebook Jupyter** (`.ipynb`) détaillant l'analyse exploratoire de chaque dataset, la logique de fusion, les visualisations et les décisions de nettoyage.
- Un **dataset consolidé et nettoyé au format CSV**, prêt à être intégré dans le pipeline d'entraînement. Ce fichier unique sera la **source de vérité** pour les étapes suivantes.
- Un **résumé écrit** des choix réalisés (notamment sur la stratégie de fusion) et des variables clés identifiées, destiné à alimenter le rapport métier final.

### Check qualité

- Des visualisations claires des distributions et corrélations.
- Une justification documentée des choix de nettoyage.
- Un dataset nettoyé, sans erreurs ni colonnes redondantes.

### Recommandations

- Visualiser les distributions des variables et les corrélations.
- Identifier les variables clés pour la prédiction de rendement.
- Élaborer une stratégie de fusion pertinente. Une simple jointure peut nécessiter des agrégations ou des ajustements pour aligner les granularités (ex. : type de culture présent dans les 2 datasets).
- Bien documenter les choix de nettoyage et de transformation.

### Outils

- **Pandas** — *documentation*
- **Matplotlib, Seaborn** — *documentation*
- **NumPy, SciPy**

### Points de vigilance

- Ne pas négliger l'analyse des valeurs manquantes.
- Éviter d'éliminer trop d'observations sans justification.
- Vérifier la cohérence entre les deux jeux de données (formats, unités).

---

## ÉTAPE 2 — Construisez et optimisez le moteur de prédiction

### Description

Vous construirez le cœur de votre application : le **modèle de prédiction de rendement**. Vous utiliserez le jeu de données consolidé pour entraîner plusieurs modèles, comparer leurs performances et optimiser les hyperparamètres du meilleur d'entre eux. **MLflow** sera votre journal de bord pour suivre rigoureusement chaque expérimentation.

### Prérequis

- Avoir finalisé les datasets prétraités.
- Avoir choisi un ou plusieurs modèles adaptés à la tâche (régression, recommandation).
- Avoir installé et configuré MLflow.

### Résultats attendus

- Un **pipeline complet**, sous forme de notebook ou de script Python (preprocessing, optimisation, évaluation).
- Les **screenshots des expérimentations sous MLflow** (`.png`) prouvant votre démarche d'optimisation, à intégrer comme preuves visuelles dans le rapport métier.
- Un script ou notebook montrant le processus d'entraînement, d'optimisation et de sélection du modèle.
- Orientez vos résultats vers le **métier** (ici, l'agriculture) en expliquant les résultats clés, les variables les plus importantes et les recommandations optimisées pour l'agriculteur (par ex. quelles cultures privilégier, selon quelles conditions).

### Check qualité

- Des modèles comparés sur des **métriques claires** (RMSE, R², profitabilité).
- Des résultats **reproductibles** (seed fixé).
- Des expérimentations **annotées** dans MLflow.
- Les éléments à inclure dans le rapport métier final sont préparés, en mettant l'accent sur l'interprétation des variables importantes et les recommandations métier.
- Les résultats sont interprétés et expliqués dans un langage accessible à un non-spécialiste.
- Les recommandations finales sont alignées avec les objectifs métier (rendement, profitabilité).

### Recommandations

- Paramétrer les métriques adaptées (par ex. RMSE, R², précision économique).
- Tester plusieurs configurations d'hyperparamètres.
- Sauvegarder les expériences et annoter les résultats significatifs.

### Outils

- **Scikit-learn** — *documentation*
- **MLflow** — *documentation*

### Points de vigilance

- Ne pas oublier de **fixer les seeds** pour la reproductibilité.
- Vérifier que les logs MLflow sont complets et lisibles.
- Éviter les overfits en validant sur des **jeux de test séparés**.

---

## ÉTAPE 3 — Développez l'API de prédiction

### Description

Vous allez maintenant « servir » votre modèle via une **API** : créer un serveur web qui charge votre modèle entraîné et expose des fonctionnalités via des URLs. Nous vous recommandons d'utiliser **FastAPI** pour sa simplicité, ses performances et sa documentation automatique.

### Résultats attendus

- Un script Python (`main.py`) définissant votre application FastAPI, contenant **au minimum deux endpoints** :
  - `POST /predict` — Prend en entrée les données d'une **seule culture** et son contexte, et retourne la prédiction de rendement.
  - `POST /recommend` — Prend en entrée **uniquement le contexte** (température, etc.), et retourne la liste de **toutes les cultures possibles**, triées par rendement prédit décroissant.
- Un **Dockerfile** pour conteneuriser votre API, assurant sa portabilité et sa reproductibilité.
- Un fichier **`requirements.txt`** listant les dépendances Python de l'API.

### Outils

- **FastAPI** — *documentation*
- **Docker** — *documentation*

---

## ÉTAPE 4 — Construisez l'interface utilisateur avec Streamlit

### Description

C'est ici que le projet prend vie pour l'utilisateur final. Vous développerez une application web simple avec **Streamlit** qui servira de front-end. Cette application ne contiendra **aucune logique de Machine Learning** ; son seul rôle sera d'interroger votre API FastAPI et d'afficher les résultats de manière claire et interactive.

### Résultats attendus

- Un script Python (`app.py`) pour votre application Streamlit. L'interface devra permettre à l'utilisateur de :
  - Choisir entre le mode **« Prédiction »** et **« Recommandation »** ;
  - Utiliser des **sliders** et **champs de saisie** pour définir le contexte (température, pesticides, etc.) ;
  - Cliquer sur un **bouton** pour envoyer une requête à l'API ;
  - Visualiser les résultats : un **chiffre clair** pour la prédiction, un **graphique à barres** et un **tableau** pour la recommandation.
- Un fichier **`requirements.txt`** pour les dépendances de l'application Streamlit.

### Outils

- **Streamlit** — *documentation*
- **Requests** (pour interroger l'API)

---

## ÉTAPE 5 — Automatisez les tests et le déploiement

### Description

Vous mettrez en place un pipeline **CI/CD** (ex. : avec **GitHub Actions**) pour automatiser la chaîne de production de votre application. Le pipeline devra gérer l'ensemble de l'architecture (back-end et front-end).

### Résultats attendus

- Un **fichier de configuration de pipeline** (`.yaml`) qui automatise les tâches suivantes :
  - **Tests** — Lancement de tests unitaires sur les fonctions critiques de votre API.
  - **Build** — Construction de l'image Docker de l'API à chaque modification du code.
  - **Déploiement** *(optionnel mais fortement recommandé)* — Déploiement automatique de l'image Docker de l'API sur un registre (comme Docker Hub) et déploiement de l'application Streamlit (ex. : sur Streamlit Community Cloud).
- Une **documentation** expliquant le fonctionnement du pipeline CI/CD.

### Check qualité

- Tests unitaires automatisés pour les modules critiques.
- Workflow CI/CD complet : **test → build → déploiement**.
- Documentation incluant : description des workflows, triggers, badges de statut.

### Recommandations

- Commencer par définir les workflows dans un fichier YAML clair.
- Automatiser les tests unitaires et les tests d'intégration.
- Prévoir des notifications en cas d'échec du pipeline.
- Préparer des visuels simples (ex. : schéma du pipeline, badges de statut) pour illustrer le rapport métier.

### Outils

- **GitHub Actions** — *documentation*
- **GitLab CI** — *documentation*

### Points de vigilance

- Ne pas exposer de secrets (API keys, credentials) dans les fichiers.
- Vérifier que les étapes critiques échouent bien en cas d'erreur.
- Documenter les workflows pour qu'ils soient compréhensibles par un tiers.

---

# 📦 Récapitulatif des livrables

> **Activité : Mission — Concevez un système de recommandation avec intégration de données multi-sources**

| Livrable | Format | Contenu |
|---|---|---|
| **Screenshots MLflow** | `.png` | Résultats des expériences d'entraînement. |
| **Notebooks / scripts du pipeline d'entraînement** | `.ipynb` / `.py` | Étapes d'optimisation et d'entraînement des modèles. |
| **Pipeline MLOps complet** | `.py` / `.yaml` (+ Dockerfile si nécessaire) | Automatisation du prétraitement, entraînement, évaluation. |
| **Pipeline CI/CD automatisé** | `.yaml` (GitHub Actions, GitLab CI…) | Orchestration de l'intégration continue, des tests et du déploiement. |
| **Rapport synthétique métier** | `.pdf` | Résultats, variables clés et recommandations pour optimiser rendement et profit. |

---

# 🎤 Soutenance

> La soutenance ne porte que sur la mission.
> Durant la présentation orale, l'évaluateur interprétera le rôle de **Gabriel, Lead Data Scientist**. La soutenance dure **30 minutes**, structurée comme suit.

### 1. Présentation des livrables — *15 min* (avec support)

- Présentation du **pipeline d'entraînement** et des expérimentations **MLflow**.
- Présentation des **pipelines MLOps et CI/CD** : structure, étapes, automatisations mises en place.
- Explication des **résultats métiers** : quelles recommandations pour l'agriculture, quelles variables influentes, quels gains anticipés.

### 2. Discussion — *10 min*

L'évaluateur jouera le rôle de Gabriel et vous challengera sur les points suivants :

- **Vos choix techniques** — pourquoi ces modèles, pourquoi ces configurations ?
- **La robustesse de vos pipelines** — que se passe-t-il en cas d'échec ?
- **L'impact métier** — en quoi vos résultats sont-ils utiles pour les exploitants agricoles ?
- **Vos choix d'optimisation** — quels hyperparamètres ont été décisifs ?

### 3. Débrief — *5 min*

À la fin de la soutenance, l'évaluateur arrêtera de jouer le rôle de Gabriel pour vous permettre de débriefer ensemble.
