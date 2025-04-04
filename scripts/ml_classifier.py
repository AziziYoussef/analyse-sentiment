#!/usr/bin/env python3

"""
Script de classification de textes utilisant l'apprentissage automatique.
Ce script prend un fichier texte en entrée et produit des prédictions pour chaque ligne.

Usage:
    ./ml_classifier.py --train --model [svm|nb|char]
    ./ml_classifier.py <fichier_entrée> > <fichier_sortie> --model [svm|nb|char]

Exemples:
    ./ml_classifier.py --train --model svm
    ./ml_classifier.py dev.txt > dev-predict.txt --model svm
"""

import sys
import argparse
import numpy as np
import pandas as pd
import re
import string
import warnings
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Supprimer les avertissements pour une sortie plus propre
warnings.filterwarnings("ignore")

def preprocess_text(text):
    """Prétraite un texte pour la classification."""
    # Convertir en minuscules
    text = text.lower()
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path):
    """Charge les données à partir d'un fichier texte."""
    data = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                text = parts[0]
                label = parts[1]
                if label == "??":  # Pour les données de test
                    label = None
                data.append(text)
                labels.append(label)
    
    return pd.DataFrame({'text': data, 'label': labels})

def train_model(model_type='svm'):
    """Entraîne un modèle de classification et l'évalue sur dev."""
    # Charger les données d'entraînement et de développement
    train_df = load_data('./data/train.txt')
    dev_df = load_data('./data/dev.txt')
    
    # Prétraiter les textes
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    dev_df['processed_text'] = dev_df['text'].apply(preprocess_text)
    
    # Encoder les étiquettes
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_dev = label_encoder.transform(dev_df['label'])
    
    # Créer et entraîner le pipeline selon le modèle choisi
    if model_type == 'nb':
        # Naïve Bayes avec TF-IDF
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(min_df=2, max_df=0.95, max_features=10000)),
            ('classifier', MultinomialNB())
        ])
    elif model_type == 'char':
        # SVM avec n-grammes de caractères
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(analyzer='char', ngram_range=(2, 4), 
                                           min_df=2, max_df=0.95, max_features=10000)),
            ('classifier', LinearSVC(random_state=42))
        ])
    else:  # par défaut, SVM avec TF-IDF sur les mots
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(min_df=2, max_df=0.95, max_features=10000)),
            ('classifier', LinearSVC(random_state=42))
        ])
    
    # Entraîner le modèle
    pipeline.fit(train_df['processed_text'], y_train)
    
    # Évaluer sur les données de développement
    y_pred = pipeline.predict(dev_df['processed_text'])
    accuracy = np.mean(y_pred == y_dev)
    print(f"Exactitude sur dev: {accuracy:.4f}", file=sys.stderr)
    
    # Sauvegarder le modèle et le label_encoder
    model_file = f"model_{model_type}.pkl"
    encoder_file = f"label_encoder_{model_type}.pkl"
    
    with open(model_file, 'wb') as f:
        pickle.dump(pipeline, f)
    
    with open(encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Modèle sauvegardé dans {model_file}", file=sys.stderr)
    print(f"Label encoder sauvegardé dans {encoder_file}", file=sys.stderr)
    
    return pipeline, label_encoder

def predict(input_file, model_type='svm'):
    """Prédit les étiquettes pour un fichier d'entrée."""
    model_file = f"model_{model_type}.pkl"
    encoder_file = f"label_encoder_{model_type}.pkl"
    
    # Vérifier si le modèle existe, sinon l'entraîner
    if not (os.path.exists(model_file) and os.path.exists(encoder_file)):
        print(f"Modèle non trouvé. Entraînement en cours...", file=sys.stderr)
        pipeline, label_encoder = train_model(model_type)
    else:
        # Charger le modèle et le label_encoder
        with open(model_file, 'rb') as f:
            pipeline = pickle.load(f)
        
        with open(encoder_file, 'rb') as f:
            label_encoder = pickle.load(f)
    
    # Lire et prétraiter les données d'entrée
    lines = []
    for line in input_file:
        line = line.strip()
        if line:
            parts = line.split("\t")
            text = parts[0]
            lines.append(text)
    
    # Prétraiter les textes
    processed_texts = [preprocess_text(text) for text in lines]
    
    # Prédire les étiquettes
    y_pred = pipeline.predict(processed_texts)
    predicted_labels = label_encoder.inverse_transform(y_pred)
    
    # Afficher les résultats
    for i, line in enumerate(lines):
        print(f"{line}\t{predicted_labels[i]}")

if __name__ == "__main__":
    # Traiter les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Classifier des textes en utilisant ML.')
    parser.add_argument('textfile', type=argparse.FileType('r', encoding='UTF-8'), nargs='?',
                       help='Fichier texte, avec une phrase par ligne, UTF-8')
    parser.add_argument('--model', type=str, default='svm',
                       help='Modèle à utiliser: svm, nb (Naive Bayes), ou char (caractères n-grammes)')
    parser.add_argument('--train', action='store_true',
                       help='Mode entraînement (nécessite train.txt et dev.txt dans le répertoire courant)')
    args = parser.parse_args()

    # Vérifier que textfile est fourni si --train n'est pas utilisé
    if not args.train and args.textfile is None:
        parser.error("l'argument textfile est requis lorsque --train n'est pas utilisé")
    
    # Exécuter l'action appropriée
    if args.train:
        train_model(args.model)
    else:
        predict(args.textfile, args.model)