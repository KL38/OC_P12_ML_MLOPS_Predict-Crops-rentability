"""Front Streamlit : aucune logique de machine learning, uniquement des appels à l'API.

Deux onglets, un par endpoint. Le contexte climatique est saisi une fois, au-dessus des
onglets, et les deux le partagent.

Les prix ne transitent jamais par l'API. Elle rend des rendements ; la marge se calcule
ici, à partir d'un tableau que l'utilisateur ajuste — un exploitant n'a pas les coûts de
son voisin. Corollaire tenu par `st.session_state` : modifier un prix reclasse l'affichage
sans relancer la moindre requête.

L'identité visuelle passe par `.streamlit/config.toml` et non par du CSS injecté : les
couleurs suivent alors les mises à jour de Streamlit au lieu de casser à la première
refonte de ses classes internes.
"""

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

DOSSIER = Path(__file__).resolve().parent
LOGO = str(DOSSIER / "logo.png")
CHEMIN_PRIX = DOSSIER / "prix_couts.csv"

# Variable d'environnement plutôt que constante : la même image sert en local
# (http://127.0.0.1:8000), derrière Docker Compose (http://api:8000) et en ligne.
URL_API = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Réserve affichée en permanence, en texte secondaire : un disclaimer doit être lisible
# sans crier plus fort que le contenu.
RESERVE = (
    "Le modèle s'appuie sur des données agricoles mondiales couvrant 1990-2013. Il décrit "
    "la relation entre climat, intrants et rendement telle qu'observée sur cette période — "
    "il ne projette pas les progrès agronomiques survenus depuis. À utiliser pour comparer "
    "des cultures entre elles, pas comme objectif chiffré absolu."
)

PORTEE = (
    "Profil agroclimatique régional de référence. Ces valeurs **situent votre région parmi "
    "les pays observés** — elles ne simulent pas l'effet d'une irrigation ou d'un changement "
    "de pratique sur une parcelle donnée."
)

# Réserve spécifique à la pluviométrie, placée en infobulle du curseur plutôt qu'à l'écran :
# elle ne concerne qu'une variable sur trois et alourdirait la page.
#
# Le fait mesuré derrière : dans la source, la pluviométrie ne prend qu'UNE valeur par pays
# (température et pesticides en prennent 21 et 23). C'est donc un identifiant de région, et
# l'importance par permutation le confirme — nulle sur des pays jamais vus, là où les deux
# autres variables gardent un apport réel.
RESERVE_PLUIE = (
    "Dans les données sources, la pluviométrie ne prend qu'**une seule valeur par pays** : "
    "elle identifie la région plutôt qu'elle ne mesure un apport d'eau. Déplacer ce curseur "
    "change donc le groupe de pays auquel votre région est comparée — ce n'est **pas** une "
    "simulation d'irrigation."
)

VERT = "#33673A"
ROUGE = "#B3261E"

st.set_page_config(page_title="Agritech Answers", page_icon=LOGO, layout="wide")

# Un seul endroit pour initialiser l'état. `version_prix` sert à réinitialiser l'éditeur :
# changer sa clé est la seule façon de le remettre à zéro, une clé stable conservant les
# modifications de l'utilisateur.
if "version_prix" not in st.session_state:
    st.session_state["version_prix"] = 0


# --- L'API -------------------------------------------------------------------------------


@st.cache_data(ttl=300, show_spinner=False)
def _sante() -> dict:
    """Métadonnées du service. En cache : elles ne bougent qu'au redéploiement de l'API."""
    reponse = requests.get(f"{URL_API}/health", timeout=5)
    reponse.raise_for_status()
    return reponse.json()


def _cultures_connues() -> list[str]:
    """Les dix cultures lues à l'API, jamais recopiées ici : une seule source de vérité."""
    try:
        return _sante()["cultures"]
    except requests.RequestException:
        st.error(
            f"API injoignable sur {URL_API} — liste des cultures indisponible.",
            icon=":material/cloud_off:",
        )
        return []


def _poster(route: str, corps: dict) -> dict | None:
    """Appelle l'API et rend None après avoir affiché l'erreur, jamais une exception.

    Le 422 est traité à part parce qu'il est le seul que l'utilisateur peut corriger
    lui-même : l'API nomme le champ fautif, autant le lui répéter.
    """
    try:
        reponse = requests.post(f"{URL_API}{route}", json=corps, timeout=10)
    except requests.RequestException:
        st.error(
            f"API injoignable sur {URL_API}. En local, lancez-la avec "
            "`uv run uvicorn src.api:app`.",
            icon=":material/cloud_off:",
        )
        return None

    if reponse.status_code == 422:
        faute = reponse.json()["detail"][0]
        champ = ".".join(str(p) for p in faute["loc"][1:])
        st.error(
            f"Saisie refusée — **{champ}** : {faute['msg']}", icon=":material/error:"
        )
        return None
    if not reponse.ok:
        st.error(f"L'API a répondu {reponse.status_code}.", icon=":material/error:")
        return None
    return reponse.json()


def _contexte() -> dict:
    """Le contexte saisi plus haut, au format attendu par les deux endpoints."""
    return {
        "pluie_mm": st.session_state["pluie"],
        "temperature_c": st.session_state["temperature"],
        "pesticides_kg_ha": st.session_state["pesticides"],
    }


# --- Prix et marges ------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _prix_par_defaut() -> pd.DataFrame:
    return pd.read_csv(CHEMIN_PRIX)


def _nom_fr(culture: str) -> str:
    return _prix_par_defaut().set_index("culture")["nom_fr"].get(culture, culture)


def _dollars(valeur: float) -> str:
    return f"{valeur:,.0f} $/ha".replace(",", " ")


def _couleur_statut(valeur: object) -> str:
    """Seule la couleur passe par le Styler ; tout le formatage reste en column_config.

    `object` et non `str` : le Styler passe ses cellules typées `Scalar`, qui couvre aussi
    les octets et les nombres.
    """
    return f"color: {VERT if valeur == 'Cultivable' else ROUGE}; font-weight: 600"


def _table_marges(reponse: dict, prix: pd.DataFrame) -> pd.DataFrame:
    """Croise les rendements de l'API avec les prix de l'utilisateur, puis classe.

    La jointure porte sur le libellé de culture et elle est interne : si le CSV et l'API
    divergeaient sur un nom, la culture disparaîtrait du classement sans qu'aucune erreur
    ne soit levée. Les dix libellés doivent rester identiques des deux côtés, virgule de
    « Rice, paddy » comprise.
    """
    table = pd.DataFrame(reponse["cultures"]).merge(prix, on="culture")
    table["marge_usd_par_ha"] = (
        table["rendement_t_ha"] * table["prix_usd_par_tonne"]
        - table["couts_usd_par_ha"]
    )
    table["statut"] = [
        "Cultivable" if ok else "Hors domaine" for ok in table["cultivable"]
    ]
    table["motif"] = table["motif"].fillna("")
    # Proposables d'abord, marge décroissante ensuite. Une culture hors domaine peut
    # afficher une marge flatteuse : elle n'a simplement jamais poussé dans ces conditions.
    return table.sort_values(
        ["cultivable", "marge_usd_par_ha"], ascending=[False, False]
    )


# --- En-tête -------------------------------------------------------------------------------

# Le logo porte déjà le nom de l'entreprise : un titre texte ferait doublon.
st.image(LOGO, width=280)
st.caption(RESERVE)

with st.container(border=True):
    st.subheader("Votre contexte")
    # Libellés volontairement régionaux et non parcellaires : les trois variables sont des
    # agrégats nationaux, les nommer autrement serait mentir sur ce que le modèle a vu.
    #
    # Valeurs par défaut = médianes du jeu d'entraînement (1032 mm, 20,4 °C, 0,92 kg/ha) :
    # l'app s'ouvre sur le profil que le modèle connaît le mieux. La médiane et non la
    # moyenne pour les pesticides, dont la distribution est très asymétrique (moyenne 2,46
    # pour une médiane de 0,92, maximum 60,57).
    #
    # Les bornes hautes relèvent de l'ergonomie, pas de la validation : l'API accepte toute
    # valeur positive. Le curseur pesticides s'arrête à 10 alors que les données montent à
    # 60,57, parce que 75 % des observations sont sous 2,89.
    colonne_pluie, colonne_temp, colonne_pest = st.columns(3)
    colonne_pluie.slider(
        "Pluviométrie annuelle de votre région (mm/an)",
        min_value=0,
        max_value=3600,
        value=1030,
        step=10,
        key="pluie",
        help=RESERVE_PLUIE,
    )
    colonne_temp.slider(
        "Température moyenne annuelle (°C)",
        min_value=-5.0,
        max_value=35.0,
        value=20.4,
        step=0.1,
        key="temperature",
    )
    colonne_pest.slider(
        "Intensité de traitement (kg/ha)",
        min_value=0.0,
        max_value=10.0,
        value=0.9,
        step=0.1,
        key="pesticides",
    )
    st.caption(PORTEE)


# --- Onglets -------------------------------------------------------------------------------

onglet_prediction, onglet_classement = st.tabs(
    [":material/agriculture: Prédiction", ":material/trending_up: Recommandation"]
)

with onglet_prediction:
    cultures = _cultures_connues()
    if cultures:
        choix, action = st.columns([3, 1], vertical_alignment="bottom")
        culture = choix.selectbox("Culture à évaluer", cultures)
        if action.button(
            "Estimer le rendement",
            type="primary",
            icon=":material/calculate:",
            width="stretch",
        ):
            reponse = _poster("/predict", {**_contexte(), "culture": culture})
            if reponse is not None:
                st.session_state["prediction"] = reponse

        prediction = st.session_state.get("prediction")
        if prediction is None:
            st.caption("Choisissez une culture, puis lancez l'estimation.")
        else:
            st.metric(
                f"Rendement estimé — {prediction['culture']}",
                f"{prediction['rendement_t_ha']:.2f} t/ha",
            )
            st.caption(_nom_fr(prediction["culture"]))
            # Rien n'est filtré ici : une demande explicite mérite une réponse, avec sa
            # réserve attachée.
            if not prediction["cultivable"]:
                st.warning(
                    f"{prediction['motif']}. Le chiffre ci-dessus est donc peu fiable.",
                    icon=":material/warning:",
                )

with onglet_classement:
    if st.button("Classer les cultures", type="primary", icon=":material/leaderboard:"):
        reponse = _poster("/recommend", _contexte())
        if reponse is not None:
            st.session_state["recommandation"] = reponse

    recommandation = st.session_state.get("recommandation")
    if recommandation is None:
        st.caption("Lancez le classement pour comparer les dix cultures.")
    else:
        # Le classement s'affiche AU-DESSUS du tableau de prix mais se calcule APRÈS lui :
        # l'éditeur doit s'exécuter d'abord pour livrer ses valeurs. Ce conteneur réserve
        # la place et se remplit ensuite.
        resultats = st.container()

        st.subheader("Vos prix et vos coûts")
        st.caption(
            "Valeurs indicatives, à ajuster à votre exploitation. Le rendement vient du "
            "modèle ; ces deux colonnes vous appartiennent."
        )

        # Centré et borné en largeur : étalé sur toute la page, un tableau de quatre
        # colonnes n'est plus qu'une ligne perdue dans du vide.
        with st.container(horizontal_alignment="center"), st.container(width=760):
            with st.container(horizontal=True, horizontal_alignment="right"):
                if st.button(
                    "Réinitialiser", icon=":material/refresh:", type="tertiary"
                ):
                    st.session_state["version_prix"] += 1

            prix = st.data_editor(
                _prix_par_defaut(),
                key=f"prix_{st.session_state['version_prix']}",
                hide_index=True,
                disabled=["culture", "nom_fr"],
                column_config={
                    "culture": "Culture",
                    "nom_fr": "Nom français",
                    "prix_usd_par_tonne": st.column_config.NumberColumn(
                        "Prix de vente ($/t)", min_value=0, step=5, format="%d"
                    ),
                    "couts_usd_par_ha": st.column_config.NumberColumn(
                        "Coûts ($/ha)", min_value=0, step=10, format="%d"
                    ),
                },
            )

        with resultats:
            table = _table_marges(recommandation, prix)
            proposables = table[table["cultivable"]]

            if proposables.empty:
                st.warning(
                    "Aucune des dix cultures n'a d'observation dans ce contexte "
                    "climatique.",
                    icon=":material/block:",
                )
            else:
                tete = proposables.iloc[0]
                colonne_culture, colonne_marge = st.columns(2)
                with colonne_culture:
                    st.metric("Culture la plus rentable", tete["culture"])
                    st.caption(tete["nom_fr"])
                colonne_marge.metric(
                    "Marge estimée", _dollars(tete["marge_usd_par_ha"])
                )
                st.bar_chart(
                    proposables,
                    x="culture",
                    y="marge_usd_par_ha",
                    horizontal=True,
                    x_label="Culture",
                    y_label="Marge ($/ha)",
                )

            st.dataframe(
                table[
                    [
                        "culture",
                        "nom_fr",
                        "rendement_t_ha",
                        "marge_usd_par_ha",
                        "statut",
                        "motif",
                    ]
                ].style.map(_couleur_statut, subset=["statut"]),
                hide_index=True,
                column_config={
                    "culture": "Culture",
                    "nom_fr": "Nom français",
                    "rendement_t_ha": st.column_config.NumberColumn(
                        "Rendement (t/ha)", format="%.2f"
                    ),
                    "marge_usd_par_ha": st.column_config.NumberColumn(
                        "Marge ($/ha)", format="%.0f"
                    ),
                    "statut": "Statut",
                    "motif": st.column_config.TextColumn(
                        "Pourquoi écartée", width="large"
                    ),
                },
            )
