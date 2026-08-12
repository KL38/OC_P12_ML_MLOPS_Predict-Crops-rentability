"""What the API and the notebook must agree on, checked automatically.

The centrepiece is `test_artefact_se_recharge_dans_un_processus_neuf`. The bug it guards
against passed every local check: a model whose inverse_func is a plain notebook function
pickles fine and reloads fine *in the notebook*, then raises AttributeError inside uvicorn,
because "__main__._clipped_exp" does not exist there. Only a fresh interpreter catches it —
which is precisely what CI gives us and a local run does not.
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import CAT, CULTURES, NUM, ClippedExp, build_features

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


def test_colonnes_exactes_et_dans_lordre():
    assert list(build_features(**CONTEXTE).columns) == CAT + NUM


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


def test_artefact_se_recharge_dans_un_processus_neuf(tmp_path):
    """Le test qui vaut tous les autres : charger depuis un interpréteur vierge.

    On fabrique ici un modèle minuscule mais de même forme que celui servi — le point testé
    est la sérialisation d'`inverse_func`, pas la qualité de la prédiction.
    """
    X = np.random.default_rng(0).random((40, 3))
    modele = TransformedTargetRegressor(
        regressor=make_pipeline(DummyRegressor()),
        func=np.log,
        inverse_func=ClippedExp(np.log(0.058), np.log(49.6)),
    ).fit(X, 2 + X[:, 0] * 5)

    chemin = tmp_path / "modele.joblib"
    joblib.dump(modele, chemin)
    attendu = modele.predict(X[:3])

    # La classe doit être référencée par son module importable, jamais par __main__.
    assert type(modele.transformer_.inverse_func).__module__ == "src.preprocessing"

    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        import joblib, numpy as np
        modele = joblib.load({str(chemin)!r})
        print(" ".join(f"{{v:.10f}}" for v in modele.predict(np.load({str(tmp_path / "X.npy")!r}))))
    """)
    np.save(tmp_path / "X.npy", X[:3])
    # check=False : l'échec est précisément ce qu'on veut inspecter, pas propager.
    sortie = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert sortie.returncode == 0, (
        f"chargement impossible hors du notebook :\n{sortie.stderr}"
    )
    assert np.allclose([float(v) for v in sortie.stdout.split()], attendu)


def test_modele_livre_est_portable():
    """Le vrai artefact, quand il existe. Ignoré si le notebook n'a pas encore tourné."""
    chemin = Path(__file__).resolve().parents[1] / "models" / "model_B.joblib"
    if not chemin.exists():
        pytest.skip("models/model_B.joblib absent (notebook pas encore exécuté)")

    modele = joblib.load(chemin)
    assert type(modele.transformer_.inverse_func).__module__ == "src.preprocessing"

    predictions = modele.predict(build_features(**CONTEXTE))
    assert len(predictions) == len(CULTURES)
    assert (predictions > 0).all(), "un rendement négatif ou nul n'a pas de sens"
