# NYC Taxi Trip Duration Prediction with PySpark

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-NYC_Taxi_Trip_Duration-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-PySpark-orange?logo=apachespark)](https://spark.apache.org/docs/latest/api/python/)
[![Spark MLlib](https://img.shields.io/badge/Spark%20MLlib-Regression%20Models-blueviolet)](https://spark.apache.org/mllib/)

## Introduction

Ce projet s’inscrit dans la compétition Kaggle **NYC Taxi Trip Duration**, dont l’objectif est de prédire la durée totale d’un trajet de taxi à New York à partir d’informations spatio-temporelles et contextuelles. Le problème est formulé comme une tâche de **régression supervisée** à grande échelle, avec plus de **1,4 million de trajets** dans le jeu d’entraînement.

Compte tenu de la volumétrie des données et de la nature des transformations nécessaires, l’ensemble du pipeline repose exclusivement sur **Apache Spark via PySpark**, en combinant Spark SQL pour la préparation des données et Spark MLlib pour la modélisation. Le projet vise autant la performance prédictive que la construction d’un pipeline robuste, reproductible et scalable, proche de standards industriels.

## Problématique et objectifs

La durée d’un trajet urbain dépend de multiples facteurs non linéaires : distance réelle parcourue, heure de départ, jour de la semaine, congestion liée aux périodes de pointe ou encore caractéristiques propres aux zones géographiques. Or, les données mises à disposition ne fournissent ni l’itinéraire exact ni l’état du trafic en temps réel.

L’objectif du projet est donc double. Il s’agit d’abord de concevoir un **feature engineering pertinent** permettant de capturer indirectement ces phénomènes à partir d’informations limitées. Il s’agit ensuite de comparer plusieurs familles de modèles de régression afin d’évaluer leur capacité à modéliser ces relations complexes et à généraliser correctement sur des données non vues.

## Données utilisées

Les données proviennent exclusivement des fichiers `train.csv` et `test.csv` fournis par Kaggle. Le jeu d’entraînement contient environ **1 458 644 observations** avec la variable cible `trip_duration` (en secondes), tandis que le jeu de test contient environ **625 134 trajets** sans la cible.

Les fichiers sont chargés sous forme de **DataFrames Spark**, permettant une inspection rapide du schéma, des types de données et des valeurs aberrantes. Une attention particulière est portée à l’homogénéité entre les jeux d’entraînement et de test afin de garantir la cohérence du pipeline de modélisation.

À l’issue de la phase de feature engineering, les données nettoyées et enrichies sont exportées dans le répertoire `data/cleaned/`. Cette séparation explicite entre préparation des données et modélisation permet de faciliter les itérations et de réduire le temps de recalcul lors des expérimentations sur les modèles.

## Organisation du repository

Le projet est structuré autour de deux notebooks principaux correspondant aux grandes étapes du workflow :

- `projet/features.ipynb` : exploration des données, nettoyage et feature engineering avec PySpark.
- `projet/ml.ipynb` : assemblage des variables, entraînement des modèles Spark MLlib et évaluation des performances.

Cette organisation reflète une logique proche d’un pipeline de production, où la préparation des données et l’entraînement des modèles sont clairement découplés, tout en restant interconnectés via des jeux de données intermédiaires standardisés.

## Feature engineering

La phase de feature engineering constitue un élément central du projet. Elle débute par un **nettoyage rigoureux de la variable cible**, avec la suppression des trajets présentant des durées négatives ou excessivement longues (supérieures à trois heures). Ce filtrage vise à limiter l’impact des outliers extrêmes, susceptibles de biaiser l’apprentissage des modèles.

Les variables temporelles sont ensuite exploitées de manière approfondie. À partir de la colonne `pickup_datetime`, plusieurs composantes sont extraites, telles que le mois, l’heure de la journée, le jour de la semaine ainsi qu’un indicateur binaire distinguant les week-ends des jours ouvrés. Ces variables permettent de capturer les effets cycliques liés aux habitudes de déplacement et aux variations de trafic au cours du temps.

Une transformation clé du pipeline consiste en l’application d’un **logarithme sur la variable cible** via `log1p(trip_duration)`. Cette transformation permet de réduire l’asymétrie marquée de la distribution des durées, de stabiliser la variance et d’améliorer le comportement numérique des modèles linéaires. Elle rend également l’optimisation plus cohérente avec la métrique RMSLE utilisée par Kaggle.

Sur le plan spatial, la distance entre le point de prise en charge et le point de dépose est approximée à l’aide de la **formule de Haversine**, calculée à partir des coordonnées GPS converties en radians. Bien qu’il s’agisse d’une approximation de la distance réelle parcourue, cette variable constitue un proxy robuste et peu coûteux pour la longueur du trajet.

Enfin, les variables catégorielles de faible cardinalité, telles que `store_and_fwd_flag`, sont encodées numériquement à l’aide de `StringIndexer`. L’ensemble des variables numériques finales est sélectionné de manière cohérente et sauvegardé pour être utilisé dans la phase de modélisation.

## Pipeline de modélisation

Dans le notebook `ml.ipynb`, les données nettoyées sont chargées depuis le répertoire `data/cleaned/`. Les différentes variables explicatives sont ensuite combinées au sein d’un vecteur unique grâce à `VectorAssembler`, étape indispensable pour l’utilisation des algorithmes de Spark MLlib.

Les données sont séparées en un jeu d’entraînement et un jeu de validation selon un split aléatoire 75/25. Ce choix permet d’obtenir une estimation fiable des performances tout en conservant un volume important de données pour l’apprentissage, compte tenu de la taille du dataset.

## Modèles de régression évalués

Plusieurs modèles de régression sont entraînés afin de comparer différentes hypothèses de modélisation. Une **régression linéaire** est utilisée comme baseline, aussi bien sur la cible brute que sur la cible log-transformée, afin de mesurer le gain apporté par des modèles plus complexes.

Une **régression Ridge** est ensuite mise en œuvre afin d’introduire une régularisation L2, permettant de limiter le sur-apprentissage et de stabiliser les coefficients dans un contexte de variables potentiellement corrélées.

Les modèles arborés permettent ensuite de capturer des relations non linéaires. Un **arbre de décision régressif** est d’abord testé, puis étendu à une **Random Forest**, combinant plusieurs arbres entraînés sur des sous-échantillons de données afin de réduire la variance.

Un modèle de **Gradient Boosted Trees** est également évalué. Ce modèle repose sur le principe du boosting, où chaque arbre successif cherche à corriger les erreurs des précédents. Les paramètres de profondeur, de nombre d’itérations et de taux d’apprentissage sont ajustés pour obtenir un compromis satisfaisant entre biais et variance.

Enfin, une **Generalized Linear Regression** avec distribution Gamma et lien logarithmique est testée afin de modéliser explicitement le caractère strictement positif et asymétrique de la durée des trajets.

## Évaluation des performances

Les performances des modèles sont évaluées à l’aide de plusieurs métriques complémentaires. Les indicateurs classiques de régression — RMSE, MAE et R² — sont calculés via `RegressionEvaluator` afin de comparer les modèles sur des bases standardisées.

La métrique principale de la compétition, le **RMSLE**, est calculée séparément à partir des prédictions. Cette métrique pénalise davantage les erreurs relatives importantes et est particulièrement adaptée à la distribution des durées de trajets, caractérisée par une forte asymétrie.

## Stratégie de tuning et limites

Le tuning des hyperparamètres est réalisé de manière itérative et manuelle, en s’appuyant sur l’analyse des performances sur le jeu de validation et sur la compréhension théorique des modèles. Les paramètres contrôlant la complexité des modèles arborés (profondeur, nombre d’arbres, taux d’apprentissage) sont ajustés progressivement afin de limiter le sur-apprentissage.

Cette approche volontairement simple met en évidence les compromis classiques rencontrés en pratique, mais ouvre également la voie à des améliorations futures via l’utilisation de `CrossValidator` ou `TrainValidationSplit` pour une recherche plus systématique des hyperparamètres.

## Conclusion et perspectives

Ce projet illustre la mise en œuvre complète d’un pipeline de machine learning distribué avec PySpark, depuis la préparation de données massives jusqu’à l’évaluation comparative de plusieurs modèles de régression. Il met en évidence l’importance du feature engineering, en particulier dans un contexte spatio-temporel, ainsi que l’apport progressif de modèles de complexité croissante.

Les principales pistes d’amélioration incluent l’enrichissement des features géographiques (clustering des zones de pickup et dropoff, intégration de distances routières), l’optimisation systématique des hyperparamètres et l’alignement encore plus direct de l’apprentissage sur la métrique RMSLE. Dans une optique portfolio ou industrielle, ce projet constitue une base solide démontrant la maîtrise de PySpark, de Spark MLlib et des problématiques de machine learning à grande échelle.
