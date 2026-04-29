# Prédiction du risque cyclonique aux Antilles

## Présentation du projet

Ce projet est un Proof of Concept de machine learning visant à prédire le risque cyclonique élevé dans la zone des Antilles à partir de données historiques de cyclones tropicaux.

L’objectif est de construire un pipeline complet allant de la préparation des données jusqu’à l’évaluation de modèles et la présentation des résultats dans une application Streamlit.

## Contexte

Les Antilles sont régulièrement exposées aux phénomènes cycloniques de l’Atlantique Nord. Ces événements peuvent entraîner des vents violents, des inondations, des dégâts matériels importants et des risques pour les populations.

Dans ce contexte, l’objectif du projet est d’identifier automatiquement les observations cycloniques les plus dangereuses à partir de variables météorologiques et géographiques simples.

Ce projet ne constitue pas un système d’alerte officiel. Il s’agit d’un POC destiné à démontrer comment un modèle de machine learning peut être intégré dans un workflow complet de prédiction du risque.

## Source des données

Les données utilisées proviennent de la base IBTrACS de la NOAA.

IBTrACS recense les trajectoires, positions, intensités, vitesses de vent et pressions atmosphériques des cyclones tropicaux à l’échelle mondiale.

Le fichier utilisé dans ce projet correspond au bassin Atlantique Nord :

`ibtracs.NA.list.v04r01.csv`

## Zone étudiée

Le projet se concentre sur la zone des Antilles.

La zone géographique a été approximée avec les coordonnées suivantes :

- latitude entre 10 et 25 ;
- longitude entre -75 et -55.

Ce filtre permet de conserver les observations cycloniques situées autour de la Caraïbe et des Antilles.

## Préparation des données

Les principales étapes de préparation sont :

1. chargement du fichier IBTrACS ;
2. suppression de la ligne contenant les unités ;
3. sélection des variables utiles ;
4. conversion des colonnes numériques ;
5. filtrage géographique sur la zone des Antilles ;
6. création de la variable cible `high_risk` ;
7. sauvegarde du dataset préparé dans `data/processed`.

Le fichier final utilisé par les modèles est :

`data/processed/ibtracs_antilles_model.csv`

## Variable cible

La variable cible du projet est `high_risk`.

Une observation est considérée comme à haut risque lorsque la vitesse du vent atteint au moins 64 nœuds.

- `0` : risque faible ou modéré ;
- `1` : risque élevé.

Le seuil de 64 nœuds est utilisé comme point de référence pour identifier les observations cycloniques les plus dangereuses.

## Variables utilisées par les modèles

Pour éviter que le modèle apprenne directement la règle de décision, la variable `WMO_WIND` est utilisée pour créer la cible, mais elle n’est pas utilisée comme variable d’entrée du modèle.

Les variables utilisées pour l’entraînement sont :

- `SEASON` : année de l’observation ;
- `LAT` : latitude ;
- `LON` : longitude ;
- `WMO_PRES` : pression atmosphérique ;
- `NATURE` : nature de l’événement.

Cette approche rend le modèle plus crédible, car il doit prédire le risque à partir d’informations indirectes liées à la situation cyclonique.

## Modèles entraînés

Deux modèles ont été entraînés et comparés.

### Logistic Regression

La régression logistique sert de modèle de référence. Elle est simple, rapide à entraîner et facilement interprétable.

### Random Forest

Le Random Forest est un modèle d’ensemble basé sur plusieurs arbres de décision. Il permet de capturer des relations plus complexes entre les variables.

Les modèles entraînés sont sauvegardés dans le dossier `models/` :

- `models/log_reg.joblib`
- `models/random_forest.joblib`

## Métriques d’évaluation

Les modèles sont évalués avec les métriques suivantes :

- accuracy ;
- precision ;
- recall ;
- F1-score.

Dans ce projet, le recall est particulièrement important, car il mesure la capacité du modèle à identifier les observations réellement dangereuses.

Le F1-score est également utile, car il équilibre la precision et le recall.

## Résultats

Les résultats sont sauvegardés automatiquement dans :

`results/model_metrics.csv`

Ils sont également affichés dans l’application Streamlit.

Dans la version actuelle du POC, les deux modèles obtiennent de bonnes performances. Le Random Forest présente un F1-score légèrement supérieur, ce qui en fait le modèle le plus performant sur ce jeu de test.

## Application Streamlit

L’application Streamlit présente :

- l’objectif business ;
- le dataset préparé ;
- les indicateurs clés ;
- la répartition du risque ;
- l’évolution des observations par année ;
- la comparaison des modèles ;
- l’interprétation des résultats ;
- les limites du projet.

## Structure du projet

```text
ml-poc-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
├── models/
├── notebooks/
├── plots/
├── results/
├── scripts/
│   └── main.py
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── data.py
│   └── metrics.py
│
├── README.md
└── requirements.txt