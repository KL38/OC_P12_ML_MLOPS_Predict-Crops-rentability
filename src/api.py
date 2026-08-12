"""The service layer: two endpoints, one model.

`/recommend` is nothing but `/predict` called on the ten crops of a single context. No
second model, no second training -- literally the same object, loaded once at startup.

Two different things happen to a bad request, and the distinction is deliberate:

  - Pydantic REFUSES what cannot physically exist (negative rainfall, -300 degrees).
    Nothing is predicted, the caller gets a 422 naming the offending field.
  - The validity domain FLAGS what is agronomically implausible (cassava at 2 degrees).
    The prediction is still returned, with `cultivable: false` and the reason. A tree
    model never declines to predict: without that flag it would display a perfectly
    plausible yield where no observation has ever existed.

Prices deliberately do not pass through here. They are adjustable in the app -- a farmer
does not have his neighbour's costs -- so the API returns yields and Streamlit turns them
into margins. Editing a price must not call the model again.
"""

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.preprocessing import ANNEE_REFERENCE, CULTURES, build_features

RACINE = Path(__file__).resolve().parent.parent
CHEMIN_MODELE = RACINE / "models" / "model_B.joblib"
CHEMIN_DOMAINE = RACINE / "models" / "domaine_validite.json"

AVERTISSEMENT = (
    f"Les données d'entraînement s'arrêtent en {ANNEE_REFERENCE}, où les rendements "
    "progressaient alors d'environ 1,3 %/an de manière stable : les valeurs rendues sont donc"
    "potentiellement sous-estimées d'à peu près 20 % aujourd'hui. Par simplification, "
    "aucune correction de tendance n'a été appliquée."
)

# Loaded once at startup, never per request: joblib.load costs ~200 ms, which would
# dominate a prediction that takes about one.
RESSOURCES: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the artifacts at startup, release them at shutdown.

    Failing here is the intended behaviour when the model is missing: uvicorn refuses to
    serve rather than answering 500 on every call.
    """
    RESSOURCES["modele"] = joblib.load(CHEMIN_MODELE)
    RESSOURCES["domaine"] = json.loads(CHEMIN_DOMAINE.read_text(encoding="utf-8"))
    yield
    RESSOURCES.clear()


app = FastAPI(
    title="Agritech Answers — rendements agricoles",
    version="1.0.0",
    summary="Prédit un rendement, et classe dix cultures pour un même contexte climatique.",
    description=(
        "Un unique modèle HistGradientBoosting, entraîné sur 87 pays (1990-2013) et évalué "
        "sur 22 pays jamais vus : RMSE 4,94 t/ha · MAE 2,85 · R² 0,627.\n\n"
        f"**Réserve à afficher à l'utilisateur.** {AVERTISSEMENT}"
    ),
    lifespan=lifespan,
)


# --- Schemas ---------------------------------------------------------------------------

# Literal accepts a tuple and flattens it, so the ten crops become an enum in the OpenAPI
# schema -- a dropdown in /docs, and a 422 rather than a 500 on an unknown crop.
CultureConnue = Literal[tuple(CULTURES)]  # type: ignore[valid-type]


class Contexte(BaseModel):
    """What the farmer enters. One input per model feature, with no transformation.

    The bounds below are physical, not agronomic: they rule out what cannot exist, never
    what is merely rare. Rare is the validity domain's job, further down.
    """

    pluie_mm: float = Field(
        ge=0, description="Pluviométrie annuelle, en mm/an.", examples=[1485.0]
    )
    temperature_c: float = Field(
        ge=-90,
        le=60,
        description="Température moyenne annuelle, en °C.",
        examples=[16.4],
    )
    pesticides_kg_ha: float = Field(
        ge=0, description="Intensité de traitement, en kg/ha.", examples=[0.17]
    )
    annee: int | None = Field(
        default=None,
        ge=1900,
        le=2100,
        description=(
            "Année de la campagne. Par défaut l'année courante — elle n'est pas figée à "
            f"{ANNEE_REFERENCE}, mais toute année postérieure donne la même prédiction : "
            "un arbre sature sur sa dernière coupure."
        ),
    )


class RequetePredict(Contexte):
    culture: CultureConnue = Field(
        description="Culture à évaluer, parmi les dix connues du modèle.",
        examples=["Maize"],
    )


class Rendement(BaseModel):
    culture: str
    rendement_t_ha: float
    cultivable: bool = Field(
        description="Faux si le contexte demandé sort du domaine observé pour cette culture."
    )
    motif: str | None = Field(
        default=None, description="Ce qui écarte la culture, quand elle est écartée."
    )


class ReponsePredict(Rendement):
    annee: int


class ReponseRecommend(BaseModel):
    annee: int
    cultures: list[Rendement] = Field(
        description="Les dix cultures, proposables d'abord, puis par rendement décroissant."
    )


class Etat(BaseModel):
    statut: str
    cultures: list[str]
    annee_max_entrainement: int
    avertissement: str


# --- Domain logic ----------------------------------------------------------------------


def _annee(demandee: int | None) -> int:
    """The requested year, or the current one when nothing is asked.

    Not frozen to the last training year: the app states the gap in a disclaimer rather
    than hardcoding a date, which would suggest a correction that does not happen. UTC
    rather than server local time -- for a growing season the two differ only in the first
    hours of January, and an explicit timezone beats an implicit one.
    """
    return demandee or datetime.now(UTC).year


def _motif_exclusion(culture: str, pluie_mm: float, temperature_c: float) -> str | None:
    """Why this crop falls outside its observed domain, or None when it is inside.

    Same bounds and same test as the notebook cell that produced the file: min/max seen on
    the training rows, widened by 10 %. Deliberately loose -- the point is to rule out
    cassava in Russia, not to hug the training set.
    """
    bornes = RESSOURCES["domaine"][culture]
    if not bornes["temp_min"] <= temperature_c <= bornes["temp_max"]:
        return (
            f"Température hors du domaine observé pour cette culture, peu de chance d'obtenir un rendement "
            f"({bornes['temp_min']:.1f} à {bornes['temp_max']:.1f} °C)"
        )
    if not bornes["pluie_min"] <= pluie_mm <= bornes["pluie_max"]:
        return (
            f"Pluviométrie hors du domaine observé pour cette culture, peu de chance d'obtenir un rendement "
            f"({bornes['pluie_min']:.0f} à {bornes['pluie_max']:.0f} mm/an)"
        )
    return None


def _rendements(contexte: Contexte, cultures: list[str], annee: int) -> list[Rendement]:
    """Predict one row per crop, in a single call, and flag those out of domain.

    One code path for both endpoints: that is what makes /predict and /recommend agree by
    construction rather than by convention.
    """
    lignes = build_features(
        pluie_mm=contexte.pluie_mm,
        temperature_c=contexte.temperature_c,
        pesticides_kg_ha=contexte.pesticides_kg_ha,
        cultures=cultures,
        annee=annee,
    )
    predictions = RESSOURCES["modele"].predict(lignes)

    rendements = []
    for culture, prediction in zip(cultures, predictions, strict=True):
        motif = _motif_exclusion(culture, contexte.pluie_mm, contexte.temperature_c)
        rendements.append(
            Rendement(
                culture=culture,
                rendement_t_ha=round(float(prediction), 2),
                cultivable=motif is None,
                motif=motif,
            )
        )
    return rendements


# --- Endpoints -------------------------------------------------------------------------


@app.get("/health", summary="Le service est-il prêt, et sur quoi porte-t-il ?")
def health() -> Etat:
    """Startup probe for Docker, and the single source of the disclaimer for the app.

    The app reads the warning here instead of holding its own copy, so the two cannot
    drift apart.
    """
    return Etat(
        statut="ok",
        cultures=CULTURES,
        annee_max_entrainement=ANNEE_REFERENCE,
        avertissement=AVERTISSEMENT,
    )


@app.post("/predict", summary="Le rendement d'une culture, pour un contexte donné")
def predict(requete: RequetePredict) -> ReponsePredict:
    """One crop, one yield in t/ha.

    `cultivable` is returned but nothing is filtered: an explicit request deserves an
    answer, with its reservation attached.
    """
    annee = _annee(requete.annee)
    (rendement,) = _rendements(requete, [requete.culture], annee)
    return ReponsePredict(**rendement.model_dump(), annee=annee)


@app.post("/recommend", summary="Les dix cultures classées, pour un même contexte")
def recommend(contexte: Contexte) -> ReponseRecommend:
    """The ten crops of the same context, best first.

    All ten are returned, including those out of domain: the app displays them greyed out
    with their reason. Dropping them silently would leave the farmer wondering why a crop
    he grows is missing from the list.
    """
    annee = _annee(contexte.annee)
    rendements = _rendements(contexte, CULTURES, annee)

    # Proposable crops first, then by decreasing yield. The app re-sorts by margin: it
    # holds the prices, and editing one must not call the model again.
    rendements.sort(key=lambda r: (not r.cultivable, -r.rendement_t_ha))
    return ReponseRecommend(annee=annee, cultures=rendements)
