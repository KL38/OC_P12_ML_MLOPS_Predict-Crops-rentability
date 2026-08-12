"""The API contract: what a caller is entitled to expect from the two endpoints.

Everything here runs against the real artifact, which ships with the repository because
the Docker image copies it in. Its absence is therefore a failure and not a reason to
skip: a skipped test passes in silence, and a model missing from the image is precisely
the kind of breakage that has to turn the run red rather than green.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import app
from src.preprocessing import ANNEE_REFERENCE, CULTURES

# A real context, taken from the training range: nothing here is out of domain.
CONTEXTE = {"pluie_mm": 1485, "temperature_c": 16.4, "pesticides_kg_ha": 0.17}

# 5 degrees and 700 mm: a temperate climate, chosen because the four tropical crops fall
# below their observed temperature floor there and the other six do not.
TEMPERE = {"pluie_mm": 700, "temperature_c": 5.0, "pesticides_kg_ha": 0.5}
TROPICALES = {"Cassava", "Plantains and others", "Sweet potatoes", "Yams"}


@pytest.fixture(scope="module")
def client():
    """The `with` block is what runs lifespan, and therefore what loads the model."""
    with TestClient(app) as connecte:
        yield connecte


# --- Ce que l'API refuse --------------------------------------------------------------


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("pluie_mm", -1),
        ("pesticides_kg_ha", -0.1),
        ("temperature_c", -300),  # sous le zéro absolu
        ("temperature_c", 100),  # au-delà du record terrestre
    ],
)
def test_valeurs_physiquement_impossibles_refusees(client, champ, valeur):
    """422 et non 500 : le champ fautif est nommé, rien n'est prédit."""
    reponse = client.post("/recommend", json={**CONTEXTE, champ: valeur})
    assert reponse.status_code == 422
    assert reponse.json()["detail"][0]["loc"][-1] == champ


def test_culture_inconnue_refusee(client):
    reponse = client.post("/predict", json={**CONTEXTE, "culture": "Quinoa"})
    assert reponse.status_code == 422


def test_champ_manquant_refuse(client):
    assert client.post("/predict", json=CONTEXTE).status_code == 422  # pas de culture


# --- Ce que l'API signale sans refuser -------------------------------------------------


def test_climat_tempere_ecarte_les_quatre_tropicales(client):
    """Le filtre de domaine, sur le cas qu'il existe pour attraper.

    À 5 °C aucune des quatre n'a jamais été observée ; le modèle, lui, rendrait un
    rendement plausible sans broncher.
    """
    cultures = client.post("/recommend", json=TEMPERE).json()["cultures"]
    ecartees = {c["culture"] for c in cultures if not c["cultivable"]}
    assert ecartees == TROPICALES
    # Insensible à la casse : ce qui est testé est que le motif nomme la BONNE variable,
    # pas la formulation exacte, qui reste libre de bouger.
    assert all(
        "température" in c["motif"].lower() for c in cultures if not c["cultivable"]
    )


def test_pluviometrie_hors_domaine_donne_son_propre_motif(client):
    """L'autre branche du filtre : 60 mm/an sous 20 °C, tempéré côté température."""
    aride = {"pluie_mm": 60, "temperature_c": 20.0, "pesticides_kg_ha": 0.5}
    cultures = client.post("/recommend", json=aride).json()["cultures"]
    manioc = next(c for c in cultures if c["culture"] == "Cassava")
    assert not manioc["cultivable"]
    assert "pluviométrie" in manioc["motif"].lower()


def test_toutes_les_cultures_sont_rendues_meme_ecartees(client):
    """Écarter n'est pas masquer : l'agriculteur doit voir pourquoi une culture manque."""
    cultures = client.post("/recommend", json=TEMPERE).json()["cultures"]
    assert {c["culture"] for c in cultures} == set(CULTURES)


def test_une_culture_dans_son_domaine_na_pas_de_motif(client):
    cultures = client.post("/recommend", json=CONTEXTE).json()["cultures"]
    assert all(c["cultivable"] and c["motif"] is None for c in cultures)


# --- Le classement ---------------------------------------------------------------------


def test_proposables_dabord_puis_rendement_decroissant(client):
    cultures = client.post("/recommend", json=TEMPERE).json()["cultures"]
    cles = [(not c["cultivable"], -c["rendement_t_ha"]) for c in cultures]
    assert cles == sorted(cles)


# --- Un seul modèle derrière les deux endpoints ----------------------------------------


def test_predict_et_recommend_donnent_la_meme_valeur(client):
    """Le test qui prouve qu'il n'y a qu'un modèle, et que l'encodage tient à une ligne.

    `Item` est une colonne catégorielle : construite depuis une seule valeur, ses
    catégories différeraient de celles de l'entraînement et la prédiction serait fausse
    sans qu'aucune erreur ne soit levée. C'est exactement ce que cette égalité interdit.
    """
    contexte = {**CONTEXTE, "annee": ANNEE_REFERENCE}
    classement = client.post("/recommend", json=contexte).json()["cultures"]

    for ligne in classement:
        seule = client.post(
            "/predict", json={**contexte, "culture": ligne["culture"]}
        ).json()
        assert seule["rendement_t_ha"] == ligne["rendement_t_ha"]
        assert seule["cultivable"] == ligne["cultivable"]


# --- L'année ----------------------------------------------------------------------------


def test_annee_par_defaut_est_lannee_courante(client):
    """Décision assumée : l'année réelle, pas 2013 figé, et une réserve à l'écran."""
    reponse = client.post("/recommend", json=CONTEXTE).json()
    assert reponse["annee"] == datetime.now(UTC).year


def test_annee_explicite_est_renvoyee(client):
    reponse = client.post(
        "/predict", json={**CONTEXTE, "culture": "Maize", "annee": 2000}
    )
    assert reponse.json()["annee"] == 2000


# --- Le service -------------------------------------------------------------------------


def test_health_porte_les_cultures_et_la_reserve(client):
    """L'app lit l'avertissement ici plutôt que d'en garder une copie qui divergerait."""
    corps = client.get("/health").json()
    assert corps["statut"] == "ok"
    assert corps["cultures"] == CULTURES
    assert corps["annee_max_entrainement"] == ANNEE_REFERENCE
    assert str(ANNEE_REFERENCE) in corps["avertissement"]


# --- Les valeurs servies ------------------------------------------------------------------


def test_les_rendements_servis_sont_plausibles(client):
    """Fourchette du ClippedExp embarqué : rien ne sort des rendements vus à l'entraînement."""
    cultures = client.post("/recommend", json=CONTEXTE).json()["cultures"]
    assert len(cultures) == len(CULTURES)
    assert all(0 < c["rendement_t_ha"] <= 49.6 for c in cultures)


def test_une_annee_posterieure_a_2013_ne_change_rien(client):
    """Ce que la réserve annonce, vérifié : l'arbre sature sur sa dernière coupure.

    Aucune coupure sur `Year` ne peut dépasser 2013, donc 2013 et 2026 tombent dans la
    même feuille. C'est pourquoi ne pas figer l'année ne trompe personne.
    """
    requete = {**CONTEXTE, "culture": "Maize"}
    en_2013 = client.post("/predict", json={**requete, "annee": 2013}).json()
    aujourdhui = client.post("/predict", json={**requete, "annee": 2026}).json()
    assert en_2013["rendement_t_ha"] == aujourdhui["rendement_t_ha"]
