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

        Le modèle utilise les données historiques IBTrACS de la NOAA, qui recensent les trajectoires,
        positions, vitesses de vent et pressions atmosphériques des cyclones tropicaux.
        """
    )

    st.divider()

    st.header("1. Objectif business")

    st.markdown(
        """
        Les Antilles sont exposées à des phénomènes cycloniques pouvant provoquer des dégâts humains,
        économiques et matériels importants.

        L'objectif du projet est de construire un premier modèle capable de repérer automatiquement
        les observations cycloniques à haut risque à partir de variables simples :

        - la position géographique ;
        - l'année de l'observation ;
        - la vitesse du vent ;
        - la pression atmosphérique.

        Dans ce POC, une observation est classée comme à haut risque lorsque la vitesse du vent
        atteint au moins 64 nœuds.
        """
    )

    st.divider()

    st.header("2. Dataset préparé")

    data_path = DATA_DIR / "processed" / "ibtracs_antilles_model.csv"

    if not data_path.exists():
        st.error("Le fichier de données préparé est introuvable.")
        return

    df = pd.read_csv(data_path)

    st.markdown("Aperçu des premières lignes du dataset utilisé par les modèles :")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Observations", len(df))

    with col2:
        st.metric("Variables", df.shape[1])

    with col3:
        st.metric("Observations à haut risque", int(df["high_risk"].sum()))

    with col4:
        risk_rate = round(df["high_risk"].mean() * 100, 2)
        st.metric("Taux haut risque", f"{risk_rate}%")

    st.divider()

    st.header("3. Analyse exploratoire")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Répartition du risque")

        risk_counts = df["high_risk"].value_counts().sort_index()
        risk_counts.index = ["Risque faible/modéré", "Risque élevé"]

        st.bar_chart(risk_counts)

        st.markdown(
            """
            Cette visualisation montre la proportion d'observations classées comme dangereuses
            ou non dangereuses dans le dataset filtré sur la zone des Antilles.
            """
        )

    with col_right:
        st.subheader("Distribution de la vitesse du vent")

        wind_data = df[["WMO_WIND"]].dropna().reset_index(drop=True)
        st.line_chart(wind_data)

        st.markdown(
            """
            La vitesse du vent est une variable centrale du projet, car elle sert à définir
            le niveau de risque cyclonique.
            """
        )

    st.subheader("Évolution du nombre d'observations par année")

    observations_by_year = df.groupby("SEASON").size()
    st.line_chart(observations_by_year)

    st.markdown(
        """
        Cette courbe permet d'observer l'évolution du nombre d'observations cycloniques
        disponibles dans la zone étudiée au fil du temps.
        """
    )

    st.divider()

    st.header("4. Résultats des modèles")

    results_path = RESULTS_DIR / "model_metrics.csv"

    if results_path.exists():
        metrics = pd.read_csv(results_path)

        st.markdown("Comparaison des modèles entraînés :")
        st.dataframe(metrics, use_container_width=True)

        if "f1" in metrics.columns:
            st.subheader("Comparaison du F1-score")

            model_column = "model_name" if "model_name" in metrics.columns else metrics.columns[0]
            f1_chart = metrics.set_index(model_column)[["f1"]]
            st.bar_chart(f1_chart)

            best_model = metrics.sort_values("f1", ascending=False).iloc[0]

            st.success(
                f"Le meilleur modèle selon le F1-score est : {best_model[model_column]} "
                f"avec un F1-score de {best_model['f1']:.3f}."
            )
    else:
        st.warning("Les métriques ne sont pas encore disponibles. Lancez python3 scripts/main.py.")

    st.divider()

    st.header("5. Interprétation")

    st.markdown(
        """
        Les résultats permettent de comparer les modèles sur leur capacité à détecter les observations
        cycloniques à haut risque.

        La Logistic Regression sert de modèle de référence : elle est simple, rapide et interprétable.

        Le Random Forest est plus flexible, car il peut mieux capturer des relations non linéaires
        entre la pression, le vent, la position géographique et le niveau de risque.

        Dans ce type de projet, le recall est particulièrement important : il mesure la capacité du modèle
        à repérer les situations réellement dangereuses.
        """
    )

    st.divider()

    st.header("6. Limites et améliorations possibles")

    st.markdown(
        """
        Ce POC reste volontairement simple. Pour construire un système plus complet, on pourrait ajouter :

        - des données de précipitations ;
        - des données de population exposée ;
        - des données d'altitude ;
        - des données historiques sur les dégâts ;
        - une prédiction temporelle à plusieurs jours.
        """
    )

    st.divider()

    st.header("7. Conclusion")

    st.markdown(
        """
        Ce projet montre qu'un modèle de machine learning simple peut servir de première base pour identifier
        des situations cycloniques dangereuses aux Antilles.

        Le résultat n'est pas un système d'alerte officiel, mais un Proof of Concept permettant de montrer
        comment des données météorologiques historiques peuvent être utilisées pour construire un modèle
        de classification du risque.
        """
    )


build_app()