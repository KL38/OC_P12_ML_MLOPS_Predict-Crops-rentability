"""Shared between the training notebook and the API.

Two things live here, and both exist for the same reason: whatever the notebook uses to
build a model must be the very same object the API uses to serve it. A copy on each side
would drift, and the drift would only show up as wrong predictions in production.

`ClippedExp` also has to be here rather than in the notebook because of how pickle works:
a plain function is serialised BY REFERENCE ("look up __main__._clipped_exp"), and that
reference does not exist in a uvicorn process. Loading would raise AttributeError at
deployment, never during development. A class defined in an importable module is stored as
"src.preprocessing.ClippedExp", which resolves anywhere the repository is installed.
"""

import numpy as np
import pandas as pd

# The exact column names and order the model was trained on. `Item` is categorical, the
# rest numeric -- HistGradientBoosting reads the dtype directly (categorical_features
# defaults to 'from_dtype' in scikit-learn 1.9).
CAT = ["Item"]
NUM = ["Year", "average_rain_fall_mm_per_year", "avg_temp", "pesticides_kg_per_ha"]

# Frozen here on purpose. Built from a single row, a category column would only contain
# that one value, the one-hot encoding would shift, and the prediction would be wrong
# without any error being raised. This list is what keeps the encoding aligned.
CULTURES = [
    "Cassava",
    "Maize",
    "Plantains and others",
    "Potatoes",
    "Rice, paddy",
    "Sorghum",
    "Soybeans",
    "Sweet potatoes",
    "Wheat",
    "Yams",
]

# Data stops in 2013. A tree saturates on its last split, so any later year predicts
# exactly like 2013 -- the app passes the real current year and states the gap in a
# disclaimer rather than hardcoding a date that would suggest a treatment that does not
# happen. See decision 2 of the main plan.
ANNEE_REFERENCE = 2013


class ClippedExp:
    """Back to t/ha from log space, bounded by the yields observed during training.

    Callable object rather than a function so that pickling stores the bounds by value
    and the class by reference -- the artifact then loads in any process that can import
    this module, with no dependency on the notebook's __main__.

    Clipping happens in log space: identical to clipping after exp(), since exp is
    monotone, but one step earlier. It exists because exp() turns a mild linear
    extrapolation into an absurd number -- unbounded, Ridge reached 27 000 t/ha on a
    country it had never seen. On the tree model finally retained it never fires (0
    predictions out of 2 821 on the test set), so it is a guardrail, not a correction.
    """

    def __init__(self, log_min, log_max):
        self.log_min = float(log_min)
        self.log_max = float(log_max)

    def __call__(self, z):
        return np.exp(np.clip(z, self.log_min, self.log_max))

    def __repr__(self):
        return (
            f"ClippedExp({np.exp(self.log_min):.3f} – {np.exp(self.log_max):.1f} t/ha)"
        )


def build_features(
    pluie_mm, temperature_c, pesticides_kg_ha, cultures=None, annee=None
):
    """Build the exact frame the model expects, for one context and one or more crops.

    One row per crop, all sharing the same climate: that is what /recommend needs, and
    /predict is the same call with a single crop. Returning a frame in both cases keeps
    one code path.

    `Item` is built against the full CULTURES list so the categories match training even
    when a single crop is requested.
    """
    if cultures is None:
        cultures = CULTURES
    elif isinstance(cultures, str):
        cultures = [cultures]

    inconnues = set(cultures) - set(CULTURES)
    if inconnues:
        raise ValueError(f"cultures inconnues du modèle : {sorted(inconnues)}")

    return pd.DataFrame(
        {
            "Item": pd.Categorical(cultures, categories=CULTURES),
            "Year": annee if annee is not None else ANNEE_REFERENCE,
            "average_rain_fall_mm_per_year": float(pluie_mm),
            "avg_temp": float(temperature_c),
            "pesticides_kg_per_ha": float(pesticides_kg_ha),
        }
    )[CAT + NUM]
