#!/usr/bin/env python3

import sys
import re
from collections import Counter

# Fonctions de prétraitement et d'extraction
def preprocess_for_char_ngrams(text):
    # Conversion en minuscules
    text = text.lower()
    # Suppression des URLs
    text = re.sub(r'http\S+', '', text)
    # Suppression des mentions
    text = re.sub(r'@\w+', '', text)
    # Conservation de certains caractères spéciaux qui peuvent être informatifs
    # mais suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_char_ngrams(text, n=3):
    # Ajouter des espaces au début et à la fin pour capturer les limites de mots
    text = ' ' + text + ' '
    
    # Extraction des n-grammes
    char_ngrams = []
    for i in range(len(text) - n + 1):
        char_ngrams.append(text[i:i+n])
    
    return char_ngrams

# Fonctions de construction de vocabulaire - CORRIGÉE
def build_char_ngram_vocabulary(texts, labels, n=3):
    positive_ngrams = Counter()
    negative_ngrams = Counter()
    neutral_ngrams = Counter()
    
    for text, label in zip(texts, labels):
        # Prétraitement du texte
        processed_text = preprocess_for_char_ngrams(text)
        
        # Extraction des n-grammes
        char_ngrams = extract_char_ngrams(processed_text, n)
        
        # Mise à jour des compteurs selon la catégorie
        if label == '+':
            positive_ngrams.update(char_ngrams)
        elif label == '-':
            negative_ngrams.update(char_ngrams)
        else:  # label == '='
            neutral_ngrams.update(char_ngrams)
    
    return positive_ngrams, negative_ngrams, neutral_ngrams

def find_discriminative_ngrams(pos_counter, neg_counter, neu_counter, top_n=100):
    # Calcul du total de n-grammes par catégorie
    total_pos = sum(pos_counter.values())
    total_neg = sum(neg_counter.values())
    total_neu = sum(neu_counter.values())
    
    # Calcul des scores de discrimination pour les n-grammes positifs
    pos_discriminative = {}
    for ngram, count in pos_counter.items():
        # Calcul du ratio de présence dans la catégorie positive par rapport aux autres
        pos_ratio = count / total_pos if total_pos > 0 else 0
        neg_ratio = neg_counter[ngram] / total_neg if total_neg > 0 else 0
        neu_ratio = neu_counter[ngram] / total_neu if total_neu > 0 else 0
        
        # Score de discrimination (présence dans la catégorie / présence dans les autres)
        discriminative_score = pos_ratio / (neg_ratio + neu_ratio + 0.0001)
        pos_discriminative[ngram] = discriminative_score
    
    # Pour les n-grammes négatifs
    neg_discriminative = {}
    for ngram, count in neg_counter.items():
        neg_ratio = count / total_neg if total_neg > 0 else 0
        pos_ratio = pos_counter[ngram] / total_pos if total_pos > 0 else 0
        neu_ratio = neu_counter[ngram] / total_neu if total_neu > 0 else 0
        
        discriminative_score = neg_ratio / (pos_ratio + neu_ratio + 0.0001)
        neg_discriminative[ngram] = discriminative_score
    
    # Pour les n-grammes neutres
    neu_discriminative = {}
    for ngram, count in neu_counter.items():
        neu_ratio = count / total_neu if total_neu > 0 else 0
        pos_ratio = pos_counter[ngram] / total_pos if total_pos > 0 else 0
        neg_ratio = neg_counter[ngram] / total_neg if total_neg > 0 else 0
        
        discriminative_score = neu_ratio / (pos_ratio + neg_ratio + 0.0001)
        neu_discriminative[ngram] = discriminative_score
    
    # Trier par score de discrimination et prendre les top_n
    top_pos = {ngram for ngram, _ in sorted(pos_discriminative.items(), 
                                           key=lambda x: x[1], reverse=True)[:top_n]}
    top_neg = {ngram for ngram, _ in sorted(neg_discriminative.items(), 
                                           key=lambda x: x[1], reverse=True)[:top_n]}
    top_neu = {ngram for ngram, _ in sorted(neu_discriminative.items(), 
                                           key=lambda x: x[1], reverse=True)[:top_n]}
    
    return top_pos, top_neg, top_neu

# Fonction de classification
def classify_by_char_ngrams(text, disc_pos, disc_neg, disc_neu, n=3):
    # Prétraitement
    processed = preprocess_for_char_ngrams(text)
    
    # Extraction des n-grammes
    char_ngrams = extract_char_ngrams(processed, n)
    
    # Calcul des scores pour chaque catégorie
    pos_score = sum(1 for ngram in char_ngrams if ngram in disc_pos)
    neg_score = sum(1 for ngram in char_ngrams if ngram in disc_neg)
    neu_score = sum(1 for ngram in char_ngrams if ngram in disc_neu)
    
    # Normalisation par le nombre de n-grammes dans le texte
    if len(char_ngrams) > 0:
        pos_score /= len(char_ngrams)
        neg_score /= len(char_ngrams)
        neu_score /= len(char_ngrams)
    
    # Détermination de la catégorie avec le plus grand score
    scores = {'+': pos_score, '-': neg_score, '=': neu_score}
    
    if all(score == 0 for score in scores.values()):
        return '='  # Par défaut: neutre
    
    return max(scores, key=scores.get)


def classify_by_multi_ngrams(text, all_disc_pos, all_disc_neg, all_disc_neu, n_values=[2, 3, 4]):
    """Classification utilisant plusieurs tailles de n-grammes ensemble"""
    processed = preprocess_for_char_ngrams(text)
    
    # Scores cumulés sur toutes les tailles de n-grammes
    total_pos_score = 0
    total_neg_score = 0
    total_neu_score = 0
    
    for n_idx, n in enumerate(n_values):
        # Extraction des n-grammes pour cette taille n
        char_ngrams = extract_char_ngrams(processed, n)
        
        # Accès aux dictionnaires correspondants
        disc_pos = all_disc_pos[n_idx]
        disc_neg = all_disc_neg[n_idx]
        disc_neu = all_disc_neu[n_idx]
        
        # Calcul et normalisation des scores
        if len(char_ngrams) > 0:
            pos_score = sum(1 for ngram in char_ngrams if ngram in disc_pos) / len(char_ngrams)
            neg_score = sum(1 for ngram in char_ngrams if ngram in disc_neg) / len(char_ngrams)
            neu_score = sum(1 for ngram in char_ngrams if ngram in disc_neu) / len(char_ngrams)
            
            # Accumulation des scores
            total_pos_score += pos_score
            total_neg_score += neg_score
            total_neu_score += neu_score
    
    # Détermination finale basée sur le score cumulé
    scores = {'+': total_pos_score, '-': total_neg_score, '=': total_neu_score}
    
    if all(score == 0 for score in scores.values()):
        return '='  # Par défaut: neutre
    
    return max(scores, key=scores.get)
def calculate_ngram_weights(pos_counter, neg_counter, neu_counter):
    """Calcule le poids de chaque n-gramme basé sur son pouvoir discriminant"""
    # Total des occurrences par catégorie
    total_pos = sum(pos_counter.values())
    total_neg = sum(neg_counter.values())
    total_neu = sum(neu_counter.values())
    
    # Dictionnaire pour stocker les poids de chaque n-gramme pour chaque catégorie
    ngram_weights = {}
    
    # Collecter tous les n-grammes uniques
    all_ngrams = set(pos_counter.keys()) | set(neg_counter.keys()) | set(neu_counter.keys())
    
    for ngram in all_ngrams:
        pos_freq = pos_counter.get(ngram, 0) / total_pos if total_pos > 0 else 0
        neg_freq = neg_counter.get(ngram, 0) / total_neg if total_neg > 0 else 0
        neu_freq = neu_counter.get(ngram, 0) / total_neu if total_neu > 0 else 0
        
        # Calculer le pouvoir discriminant pour chaque catégorie
        total_freq = pos_freq + neg_freq + neu_freq
        if total_freq > 0:
            pos_weight = pos_freq / total_freq
            neg_weight = neg_freq / total_freq
            neu_weight = neu_freq / total_freq
            
            # Plus le poids est élevé, plus le n-gramme est discriminant pour cette catégorie
            ngram_weights[ngram] = {
                '+': pos_weight,
                '-': neg_weight,
                '=': neu_weight
            }
    
    return ngram_weights

def classify_with_weighted_ngrams(text, ngram_weights, n=3):
    """Classification utilisant des poids pour chaque n-gramme"""
    processed = preprocess_for_char_ngrams(text)
    char_ngrams = extract_char_ngrams(processed, n)
    
    # Scores pondérés
    pos_score = 0
    neg_score = 0
    neu_score = 0
    
    for ngram in char_ngrams:
        if ngram in ngram_weights:
            weights = ngram_weights[ngram]
            pos_score += weights.get('+', 0)
            neg_score += weights.get('-', 0)
            neu_score += weights.get('=', 0)
    
    # Normalisation
    if len(char_ngrams) > 0:
        pos_score /= len(char_ngrams)
        neg_score /= len(char_ngrams)
        neu_score /= len(char_ngrams)
    
    scores = {'+': pos_score, '-': neg_score, '=': neu_score}
    
    if all(score == 0 for score in scores.values()):
        return '='
    
    return max(scores, key=scores.get)

# Fonction principale
def main():
    if len(sys.argv) != 2:
        print("Usage: ./char_ngrams_classifier.py input_file", file=sys.stderr)
        sys.exit(1)
        
    input_file = sys.argv[1]
    train_file = './data/train.txt'  # Chemin par défaut
    
    try:
        # Chargement des données d'entraînement
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    text, label = line.strip().split('\t', 1)
                    train_data.append((text, label))
        
        # Séparation des textes et des étiquettes
        train_texts = [text for text, _ in train_data]
        train_labels = [label for _, label in train_data]
        
        # Test automatique pour trouver la meilleure taille de n-grammes
        best_accuracy = 0
        best_n = 3  # Valeur par défaut
        print("Recherche de la meilleure taille de n-grammes...", file=sys.stderr)

        # Tester différentes tailles de n-grammes
        for n_value in [2, 3, 4, 5]:
            print(f"Test avec n={n_value}...", file=sys.stderr)
            pos_counter_test, neg_counter_test, neu_counter_test = build_char_ngram_vocabulary(train_texts, train_labels, n=n_value)
            disc_pos_test, disc_neg_test, disc_neu_test = find_discriminative_ngrams(pos_counter_test, neg_counter_test, neu_counter_test)
            
            # Test sur un sous-ensemble pour déterminer la meilleure taille
            correct = 0
            test_size = min(100, len(train_texts))
            for i in range(test_size):
                pred = classify_by_char_ngrams(train_texts[i], disc_pos_test, disc_neg_test, disc_neu_test, n=n_value)
                if pred == train_labels[i]:
                    correct += 1
            
            accuracy = correct / test_size
            print(f"Exactitude avec n={n_value}: {accuracy:.4f}", file=sys.stderr)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_n = n_value

        print(f"Meilleure taille de n-grammes: {best_n} (exactitude: {best_accuracy:.4f})", file=sys.stderr)

        pos_counter, neg_counter, neu_counter = build_char_ngram_vocabulary(train_texts, train_labels, n=best_n)
        disc_pos, disc_neg, disc_neu = find_discriminative_ngrams(pos_counter, neg_counter, neu_counter)
        
        # Chargement du fichier d'entrée
        input_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    text, _ = line.strip().split('\t', 1)
                    input_data.append(text)
        
                # Calcul des poids pour chaque n-gramme
        use_weighted_ngrams = True  # Mettre à True pour utiliser la pondération
        if use_weighted_ngrams:
            print("Calcul des poids pour chaque n-gramme...", file=sys.stderr)
            ngram_weights = calculate_ngram_weights(pos_counter, neg_counter, neu_counter)

        # Génération et affichage des prédictions
        for text in input_data:
            if use_weighted_ngrams:
                prediction = classify_with_weighted_ngrams(text, ngram_weights, n=best_n)
            else:
                prediction = classify_by_char_ngrams(text, disc_pos, disc_neg, disc_neu, n=best_n)
            print(f"{text}\t{prediction}")
            
    except FileNotFoundError as e:
        print(f"Erreur: Fichier non trouvé - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur: {e}", file=sys.stderr)
        sys.exit(1)

# Point d'entrée du script
if __name__ == "__main__":
    main()