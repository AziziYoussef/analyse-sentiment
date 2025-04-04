#!/usr/bin/env python3

import pandas as pd
from collections import Counter
import re
import sys

# Fonction pour charger les données
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                text, label = line.strip().split('\t', 1)  # Utilisation de split avec limite 1
                data.append((text, label))
    return data

# Fonction de prétraitement basique
def preprocess_text(text):
    # Conversion en minuscules
    text = text.lower()
    # Suppression de la ponctuation et caractères spéciaux
    text = re.sub(r'[^\w\s]', ' ', text)
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fonction pour filtrer les mots communs
def filter_counter(counter, common_words):
    return Counter({word: count for word, count in counter.items() 
                  if word not in common_words and len(word) > 1})

# Fonction de classification par vocabulaire
def classify_by_vocabulary(text, positive_vocab, negative_vocab, neutral_vocab):
    # Prétraitement du texte
    processed = preprocess_text(text)
    words = processed.split()
    
    # Compter les mots dans chaque catégorie
    pos_score = sum(1 for word in words if word in positive_vocab)
    neg_score = sum(1 for word in words if word in negative_vocab)
    neu_score = sum(1 for word in words if word in neutral_vocab)
    
    # Déterminer la catégorie avec le plus grand score
    scores = {'+': pos_score, '-': neg_score, '=': neu_score}
    max_score = max(scores.values())
    
    # Si aucun mot du vocabulaire n'est trouvé
    if max_score == 0:
        return '='  # Par défaut: neutre
    
    # En cas d'égalité, ordonner par priorité
    if scores['='] == max_score:
        return '='
    elif scores['+'] == max_score:
        return '+'
    else:
        return '-'

# Version améliorée avec des poids
def classify_weighted(text, pos_counter, neg_counter, neu_counter):
    # Prétraitement
    processed = preprocess_text(text)
    words = processed.split()
    
    # Calcul des scores pondérés par la fréquence
    pos_score = sum(pos_counter.get(word, 0) for word in words)
    neg_score = sum(neg_counter.get(word, 0) for word in words)
    neu_score = sum(neu_counter.get(word, 0) for word in words)
    
    # Normalisation par le total de mots dans chaque catégorie
    total_pos = sum(pos_counter.values())
    total_neg = sum(neg_counter.values())
    total_neu = sum(neu_counter.values())
    
    if total_pos > 0: pos_score /= total_pos
    if total_neg > 0: neg_score /= total_neg
    if total_neu > 0: neu_score /= total_neu
    
    # Détermination de la catégorie
    scores = {'+': pos_score, '-': neg_score, '=': neu_score}
    
    if all(score == 0 for score in scores.values()):
        return '='  # Par défaut
    
    return max(scores, key=scores.get)

# Fonction principale qui sera exécutée lors de l'appel au script
def main():
    # Vérifier les arguments
    if len(sys.argv) != 2:
        print("Usage: ./script.py fichier_entrée", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    train_file = './data/train.txt'  # Chemin par défaut
    
    try:
        # Chargement et préparation des données d'entraînement
        train_data = load_data(train_file)
        train_df = pd.DataFrame(train_data, columns=['text', 'label'])
        train_df['processed_text'] = train_df['text'].apply(preprocess_text)
        
        # Construction des vocabulaires
        positive_words = Counter()
        negative_words = Counter()
        neutral_words = Counter()
        
        for _, row in train_df.iterrows():
            words = row['processed_text'].split()
            
            if row['label'] == '+':
                positive_words.update(words)
            elif row['label'] == '-':
                negative_words.update(words)
            else:  # label == '='
                neutral_words.update(words)
        
        # Filtrage des mots communs
        common_words = {'le', 'la', 'les', 'des', 'et', 'en', 'du', 'de', 'un', 'une', 
                        'à', 'au', 'aux', 'pour', 'dans', 'sur', 'par', 'avec', 'ce', 
                        'cette', 'ces', 'est', 'sont', 'ont', 'qui', 'que', 'quoi', 
                        'comment', 'pourquoi', 'où', 'quand', 'pas', 'plus', 'moins'}
        
        filtered_positive = filter_counter(positive_words, common_words)
        filtered_negative = filter_counter(negative_words, common_words)
        filtered_neutral = filter_counter(neutral_words, common_words)
        
        # Création des ensembles de vocabulaire pour la classification simple
        pos_vocab = set(word for word, _ in filtered_positive.most_common(100))
        neg_vocab = set(word for word, _ in filtered_negative.most_common(100))
        neu_vocab = set(word for word, _ in filtered_neutral.most_common(100))
        
        # Test pour déterminer quelle méthode est la meilleure (sur un sous-ensemble)
        test_subset = train_data[:100]  # Utilisez les 100 premiers exemples comme test rapide
        
        # Test méthode simple
        simple_correct = 0
        for text, true_label in test_subset:
            pred = classify_by_vocabulary(text, pos_vocab, neg_vocab, neu_vocab)
            if pred == true_label:
                simple_correct += 1
        
        # Test méthode pondérée
        weighted_correct = 0
        for text, true_label in test_subset:
            pred = classify_weighted(text, filtered_positive, filtered_negative, filtered_neutral)
            if pred == true_label:
                weighted_correct += 1
        
        # Choisir la meilleure méthode
        use_weighted = weighted_correct > simple_correct
        
        # Traitement du fichier d'entrée
        input_data = load_data(input_file)
        
        # Classification et écriture des résultats sur stdout
        for text, _ in input_data:
            if use_weighted:
                prediction = classify_weighted(text, filtered_positive, filtered_negative, filtered_neutral)
            else:
                prediction = classify_by_vocabulary(text, pos_vocab, neg_vocab, neu_vocab)
            
            # Écrire sur stdout (pour redirection avec >)
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