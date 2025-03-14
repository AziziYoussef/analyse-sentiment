#!/usr/bin/env python3

# Ce programme prédit la polarité d'un texte en se basant sur un lexique de mots
# positifs et négatifs liés à l'écologie.

import sys
import pandas as pd

# Lexique de mots positifs et négatifs liés à l'écologie
MOTS_POSITIFS = [
    "écologique", "durable", "renouvelable", "vert", "énergie", "propre", 
    "recyclage", "préserver", "protéger", "avenir", "solution", "bio", "nature",
    "développement", "progrès", "économie", "alternative", "vertueux"
]

MOTS_NEGATIFS = [
    "pollution", "réchauffement", "climatique", "crise", "menace", "risque", 
    "problème", "danger", "catastrophe", "émission", "CO2", "carbone", "gaspillage",
    "détruire", "extinction", "dommage", "dégradation", "toxique"
]

def predict_sentiment(text):
    """Prédit le sentiment d'un texte basé sur un lexique simple."""
    # Convertir en minuscules pour la comparaison
    text = text.lower()
    
    # Calculer le score basé sur le nombre de mots positifs et négatifs
    score = 0
    for mot in MOTS_POSITIFS:
        if mot in text:
            score += 1
    
    for mot in MOTS_NEGATIFS:
        if mot in text:
            score -= 1
    
    # Déterminer la polarité basée sur le score
    if score > 0:
        return "+"      # Positif
    elif score < 0:
        return "-"      # Négatif
    else:
        return "="      # Neutre

# Vérifier les arguments en ligne de commande
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <input_file>")
    sys.exit(1)

input_file = sys.argv[1]

# Lire le fichier d'entrée
try:
    data = pd.read_csv(input_file, sep='\t', header=None, names=['text', 'label'])
except Exception as e:
    print(f"Erreur lors de la lecture du fichier: {e}")
    sys.exit(1)

# Prédire le sentiment pour chaque texte
for i, row in data.iterrows():
    text = row['text']
    sentiment = predict_sentiment(text)
    # Afficher le résultat au format attendu
    print(f"{text}\t{sentiment}")