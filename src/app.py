import pandas as pd
import streamlit as st

from config import DATA_DIR, RESULTS_DIR


def build_app() -> None:
    st.set_page_config(
        page_title="Prédiction du risque cyclonique aux Antilles",
        layout="wide",
    )

    st.title("Prédiction du risque cyclonique aux Antilles")

    st.markdown(
        """
        Ce projet est un Proof of Concept de machine learning visant à identifier les observations cycloniques
        présentant un risque élevé dans la zone des Antilles.

        Le modèle utilise des données issues de la base IBTrACS de la NOAA, qui recense les trajectoires,
        positions et intensités des cyclones tropicaux.
        """
    )

    st.header("Objectif business")

    st.markdown(
        """
        L'objectif est d'aider à repérer les situations météorologiques les plus dangereuses à partir
        de variables simples comme la position géographique, la saison, la pression atmosphérique
        et la vitesse du vent.

        Dans ce POC, une observation est considérée comme à haut risque lorsque la vitesse du vent
        atteint au moins 64 nœuds, seuil généralement associé à un ouragan.
        """
    )

    st.header("Dataset")

    data_path = DATA_DIR / "processed" / "ibtracs_antilles_model.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)

        st.write("Aperçu du dataset préparé :")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Nombre d'observations", len(df))

        with col2:
            st.metric("Nombre de variables", df.shape[1])

        with col3:
            st.metric("Observations à haut risque", int(df["high_risk"].sum()))

        st.subheader("Répartition du risque")
        st.bar_chart(df["high_risk"].value_counts().sort_index())

        st.subheader("Distribution de la vitesse du vent")
        st.line_chart(df["WMO_WIND"].reset_index(drop=True))

    else:
        st.warning("Le fichier de données préparé est introuvable.")

    st.header("Résultats des modèles")

    results_path = RESULTS_DIR / "model_metrics.csv"

    if results_path.exists():
        metrics = pd.read_csv(results_path)
        st.dataframe(metrics)
    else:
        st.info("Les résultats des modèles seront affichés ici après l'exécution de scripts/main.py.")

    st.header("Interprétation")

    st.markdown(
        """
        Les modèles comparés permettent d'évaluer la capacité à détecter les observations à haut risque.
        Les métriques principales sont l'accuracy, la precision, le recall et le F1-score.

        Le recall est particulièrement important dans ce type de sujet, car il mesure la capacité du modèle
        à identifier les cas réellement dangereux.
        """
    )
build_app()