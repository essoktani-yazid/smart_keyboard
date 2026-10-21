# ⌨️ Smart Keyboard V1 — Clavier Intelligent & Autocomplétion N-gramme

Interface moderne d'autocomplétion et de prédiction de texte basée sur un modèle de langage n-gramme avec lissage add-$k$ et back-off, entraîné sur les critiques de films Allociné.

Inspiré du design SaaS épuré de **Hostinger** et **Paddle**.

---

## ✨ Fonctionnalités Clés

1. **Suggestions en temps réel (Événement de frappe)** :
   - Plus besoin d'appuyer sur `Entrée` ! Les suggestions s'actualisent instantanément à chaque frappe avec une latence inférieure à **1 milliseconde**.
2. **Double Niveau de Suggestion (Tout-en-Un & Mode Dédié)** :
   - **Complétion du mot en cours** : complète le préfixe actuellement tapé (ex: `mag` ➔ `magnifique`, `magique`) en priorisant le contexte n-gramme puis la fréquence unigramme.
   - **Prédiction du mot suivant** : suggère les mots suivants les plus probables selon l'historique de la phrase grâce au modèle trigramme avec back-off.
   - **Mode Tout-en-Un (Recommandé)** : un éditeur intelligent unifié avec rubans de suggestions synchronisés.
   - **Mode Deux Entrées** : deux zones distinctes (l'une pour la complétion de préfixe, l'autre pour la prédiction de contexte).
3. **Raccourcis Clavier Intuitifs** :
   - <kbd>Tab ⇥</kbd> : Complète automatiquement le mot en cours avec le candidat ayant le meilleur score.
   - <kbd>→ Flèche</kbd> : Insère immédiatement la prédiction du mot suivant la plus probable.
   - <kbd>Clic</kbd> : Cliquer sur n'importe quel badge insère instantanément le mot.
4. **Affichage Créatif des Scores** :
   - **Jauges radiales circulaires SVG** intégrées affichant le pourcentage de confiance normalisé.
   - Préfixe tapé mis en valeur en surbrillance avec contraste dynamique.
   - Badges indiquant la source n-gramme utilisée (`3-gram`, `2-gram`, `1-gram`).
5. **Interface SaaS Moderne (Style Hostinger / Paddle)** :
   - Thème sombre ardoise (`#0b0f19`), cartes en verre dépoli avec bordures subtiles.
   - Typographie soignée (*Plus Jakarta Sans* / *JetBrains Mono*).
   - Barre d'exemples rapides (presets) et bouton de copie instantanée vers le presse-papier.

---

## 🚀 Installation & Lancement

### 1. Activer l'environnement virtuel
```bash
source /home/yazid/Bureau/Master_SDIA/env_ai310/bin/activate
```

### 2. Installer les dépendances (si nécessaire)
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application
```bash
streamlit run app.py
```

---

## 📊 Évaluation du Modèle
Pour évaluer la perplexité et la précision top-3 sur le corpus :
```bash
python -m src.evaluate
```
