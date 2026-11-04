# Prédiction des Matchs de Ligue 1

Ce dépôt contient l'implémentation en Python du modèle de Poisson bivarié dynamique pour l'analyse et la prévision des résultats de football, basé sur l'article de Koopman et Lit (2012).  
## À Propos du Projet
(Description courte à compléter : contexte, source des données de la Ligue 1, et application de la méthodologie reposant sur l'espace d'états et l'échantillonnage préférentiel).   
## Fonctionnalités Principales
Extraction et formatage des données historiques du championnat français.
Modélisation stochastique des forces d'attaque et de défense des équipes évoluant dans le temps.   
Estimation des paramètres du modèle par la méthode du maximum de vraisemblance de Monte Carlo.   
Prédiction des probabilités de victoire, nul ou défaite pour élaborer une stratégie de pari.   
Prérequis et Installation
Python 3.8+
Bibliothèques principales : numpy, pandas, scipy (à ajuster selon votre code final).
(Insérer ici la commande d'installation : pip install -r requirements.txt).
## Structure du Dépôt
- **data/** : Historique des matchs de Ligue 1 (fichiers bruts et traités).
- **src/** : Scripts de calcul (distribution de Poisson bivariée, filtre de Kalman, optimisation).
- **notebooks/** : Visualisation de l'évolution des forces des équipes et tests de validité.
## Utilisation
(Insérer ici un bloc de code exemple montrant comment instancier le modèle, l'entraîner sur un fichier CSV de la Ligue 1 et prédire la prochaine journée).
## Résultats et Évaluation
(Espace pour résumer vos performances hors échantillon, la précision des prévisions et les gains simulés face aux cotes réelles des bookmakers).   
## Références
Koopman, S. J., & Lit, R. (2012). A dynamic bivariate Poisson model for analysing and forecasting match results in the English Premier League.   