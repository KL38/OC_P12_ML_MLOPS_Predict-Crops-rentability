"""What the API and the notebook must agree on, checked automatically.

The centrepiece is `test_artefact_se_recharge_dans_un_processus_neuf`. The bug it guards
against passed every local check: a model whose inverse_func is a plain notebook function
pickles fine and reloads fine *in the notebook*, then raises AttributeError inside uvicorn,
because "__main__._clipped_exp" does not exist there. Only a fresh interpreter catches it —
which is precisely what CI gives us and a local run does not.

It runs on the shipped artifact itself rather than a look-alike built on the spot:
`models/model_B.joblib` is versioned, because the Docker image copies it in. Its absence
is a failure, never a skip.
"""

import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from src.preprocessing import CULTURES, NUM, ClippedExp, build_features

CHEMIN_MODELE = RACINE / "models" / "model_B.joblib"
CONTEXTE = {"pluie_mm": 1485, "temperature_c": 16.4, "pesticides_kg_ha": 0.17}


# --- ClippedExp ---------------------------------------------------------------------------


def test_bornes_appliquees_des_deux_cotes():
    borne = ClippedExp(np.log(0.058), np.log(49.6))
    assert borne(np.log(1e-9)) == pytest.approx(0.058)
    assert borne(np.log(1e9)) == pytest.approx(49.6)
    assert borne(np.log(5.0)) == pytest.approx(5.0)  # à l'intérieur : transparent


def test_bornes_voyagent_avec_lobjet():
    """Les bornes sont dans l'instance, pas dans des globales lues à la prédiction.

    pickle et non joblib : c'est le mécanisme réellement en jeu, et joblib n'expose pas
    de round-trip en mémoire.
    """
    borne = ClippedExp(np.log(1.0), np.log(10.0))
    recharge = pickle.loads(pickle.dumps(borne))
    assert recharge(np.log(100.0)) == pytest.approx(10.0)
    assert (recharge.log_min, recharge.log_max) == (borne.log_min, borne.log_max)


# --- build_features ------------------------------------------------------------------------


def test_toutes_les_categories_meme_pour_une_seule_culture():
    """Le piège qui fausse l'encodage sans lever d'erreur."""
    une = build_features(**CONTEXTE, cultures="Maize")
    assert len(une) == 1
    assert list(une["Item"].cat.categories) == CULTURES


def test_colonnes_alignees_sur_le_modele_entraine():
    """Référence indépendante : les noms de colonnes figés dans l'artefact au `fit`.

    Comparer `build_features` à `CAT + NUM` ne prouvait rien — la fonction se termine
    littéralement par `[CAT + NUM]`, elle se comparait à elle-même. Le modèle livré porte
    `feature_names_in_`, écrit à l'entraînement et indépendant de ce module : c'est ce
    décalage-là qui produit des prédictions fausses sans lever la moindre erreur.
    """
    modele = joblib.load(CHEMIN_MODELE)
    assert list(build_features(**CONTEXTE).columns) == list(modele.feature_names_in_)


def test_une_ligne_par_culture_et_meme_contexte():
    lignes = build_features(**CONTEXTE)
    assert len(lignes) == len(CULTURES)
    for colonne in NUM[1:]:  # tout sauf Year
        assert lignes[colonne].nunique() == 1


def test_culture_inconnue_rejetee():
    with pytest.raises(ValueError, match="inconnues"):
        build_features(**CONTEXTE, cultures=["Quinoa"])


def test_libelles_alignes_sur_le_jeu_dentrainement():
    donnees = Path(__file__).resolve().parents[1] / "notebooks" / "df.csv"
    if not donnees.exists():
        pytest.skip("df.csv absent (données non versionnées)")
    assert sorted(pd.read_csv(donnees)["Item"].unique()) == CULTURES


# --- portabilité de l'artefact --------------------------------------------------------------


def test_artefact_se_recharge_dans_un_processus_neuf():
    """Le test qui vaut tous les autres : charger le modèle livré depuis un interpréteur vierge.

    Il porte sur `models/model_B.joblib` lui-même et non sur une maquette de même forme :
    c'est cet artefact-là que l'image Docker embarque et qu'uvicorn chargera.
    """
    modele = joblib.load(CHEMIN_MODELE)
    attendu = modele.predict(build_features(**CONTEXTE))

    # La classe doit être référencée par son module importable, jamais par __main__.
    assert type(modele.transformer_.inverse_func).__module__ == "src.preprocessing"

    # Le sous-processus reconstruit ses entrées avec build_features : on vérifie du même
    # coup que la fonction que l'API appelle s'importe hors de tout contexte notebook.
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(RACINE)!r})
        import joblib
        from src.preprocessing import build_features
        modele = joblib.load({str(CHEMIN_MODELE)!r})
        rendements = modele.predict(build_features(**{CONTEXTE!r}))
        print(" ".join(f"{{v:.10f}}" for v in rendements))
    """)
    # check=False : l'échec est précisément ce qu'on veut inspecter, pas propager.
    sortie = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert sortie.returncode == 0, (
        f"chargement impossible hors du notebook :\n{sortie.stderr}"
    )
    assert np.allclose([float(v) for v in sortie.stdout.split()], attendu)


def test_modele_livre_predit_les_dix_cultures():
    """Un rendement par culture, tous strictement positifs.

    La portabilité est couverte par le test précédent, dans un processus neuf ; ici on
    regarde ce que l'artefact produit, pas la façon dont il se recharge.
    """
    predictions = joblib.load(CHEMIN_MODELE).predict(build_features(**CONTEXTE))
    assert len(predictions) == len(CULTURES)
    assert (predictions > 0).all(), "un rendement négatif ou nul n'a pas de sens"
