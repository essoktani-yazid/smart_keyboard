# Référence API — Smart Keyboard

> Documentation technique complète des endpoints exposés par le micro-serveur d'inférence en temps réel de **Smart Keyboard**.

- **URL de base (Développement local) :** `http://127.0.0.1:8555` *(ou port dynamique assigné dans la plage 8555–8605)*
- **Format d'échange :** `application/json; charset=utf-8`
- **Politique CORS :** Activée (`Access-Control-Allow-Origin: *`) pour autoriser les requêtes cross-origin asynchrones du composant frontend.

---

## Authentification & Sécurité

Afin de garantir une **latence sub-milliseconde (< 1 ms)** à chaque frappe de touche, le micro-serveur d'inférence est conçu comme un service local ultra-rapide sans surcoût de session ou de déchiffrement de jetons JWT.

- **Contrôle d'accès :** Le serveur s'exécute en local et n'écoute par défaut que sur l'interface de boucle locale (`127.0.0.1`).
- **En-têtes CORS pré-vol :** La méthode `OPTIONS` est interceptée et retourne un statut `200 OK` avec les en-têtes d'autorisation appropriés.

---

## Endpoints Disponibles

### 1. `GET /api/health` — Vérification de l'état du service

Permet de vérifier que le micro-serveur est opérationnel et que les modèles de langage n-grammes (Unigramme, Bigramme, Trigramme) sont bien instanciés en mémoire et prêts pour l'inférence.

- **URL :** `/api/health`
- **Méthode :** `GET`
- **Authentification requise :** Non
- **Paramètres :** Aucun

#### Exemple de requête

```bash
curl -X GET http://127.0.0.1:8555/api/health
```

#### Réponse de succès (`200 OK`)

```json
{
  "status": "ok",
  "ready": true
}
```

---

### 2. `GET /api/suggest` — Prédictions et complétions en temps réel

Analyse le texte fourni pour renvoyer simultanément la complétion du mot en cours de frappe (si applicable) et les prédictions contextuelles des mots suivants à l'aide des modèles n-grammes lissés (*add-k*) et de la stratégie de repli (*hard back-off*).

- **URL :** `/api/suggest`
- **Méthode :** `GET`
- **Authentification requise :** Non

#### Paramètres de requête (Query Params)

| Paramètre | Type | Requis | Défaut | Description |
| :--- | :--- | :--- | :--- | :--- |
| `text` | `string` | Non | `""` | Texte actuellement saisi dans le champ de frappe. |
| `k_comp` | `integer` | Non | `4` | Nombre maximum de suggestions de complétion pour le mot en cours. |
| `k_next` | `integer` | Non | `4` | Nombre maximum de suggestions pour le mot suivant. |

#### Comportement selon le texte saisi

1. **Mot en cours de frappe (`ends_with_space = false`) :**
   - Le dernier token est considéré comme un préfixe incomplet (`current_prefix`).
   - Le moteur recherche dans le vocabulaire tous les mots commençant par ce préfixe et les ordonne par *tier* de n-gramme (Tier 3 = Trigramme, Tier 2 = Bigramme, Tier 1 = Unigramme), puis par probabilité décroissante.
   - Les tokens précédents constituent le contexte pour anticiper le mot suivant.
2. **Mot achevé ou espace final (`ends_with_space = true`) :**
   - Aucun préfixe n'est en cours (`current_prefix = ""`), la liste `completions` est vide.
   - Tous les mots saisis forment le contexte d'historique pour prédire les mots suivants les plus probables (`next_words`) via repli Trigramme $\rightarrow$ Bigramme $\rightarrow$ Unigramme.

#### Exemples de requêtes

```bash
# Exemple A : Complétion en cours de frappe du préfixe "cin"
curl -G "http://127.0.0.1:8555/api/suggest" \
  --data-urlencode "text=j'aime ce cin" \
  -d "k_comp=4" \
  -d "k_next=4"

# Exemple B : Prédiction du mot suivant après un mot achevé avec espace
curl -G "http://127.0.0.1:8555/api/suggest" \
  --data-urlencode "text=le film est " \
  -d "k_comp=4" \
  -d "k_next=4"
```

#### Réponse de succès (`200 OK`)

```json
{
  "text": "j'aime ce cin",
  "ends_with_space": false,
  "current_prefix": "cin",
  "context_words": [
    "j'aime",
    "ce"
  ],
  "completions": [
    {
      "word": "cinéma",
      "prob": 0.428,
      "norm_score": 75.3,
      "tier": 2,
      "source": "2-gram",
      "prefix": "cin",
      "suffix": "éma"
    },
    {
      "word": "cinématographique",
      "prob": 0.140,
      "norm_score": 24.7,
      "tier": 1,
      "source": "1-gram",
      "prefix": "cin",
      "suffix": "ématographique"
    }
  ],
  "next_words": [
    {
      "word": "film",
      "display": "film",
      "prob": 0.354,
      "norm_score": 46.2,
      "is_eos": false,
      "source": "2-gram"
    },
    {
      "word": "moment",
      "display": "moment",
      "prob": 0.215,
      "norm_score": 28.1,
      "is_eos": false,
      "source": "2-gram"
    },
    {
      "word": "genre",
      "display": "genre",
      "prob": 0.197,
      "norm_score": 25.7,
      "is_eos": false,
      "source": "2-gram"
    }
  ],
  "order": "2-gram",
  "vocab_size": 1542,
  "latency_ms": 0.38
}
```

#### Détail des champs de la réponse

- **`text`** *(string)* : Chaîne exacte transmise dans la requête.
- **`ends_with_space`** *(boolean)* : Indique si le texte se termine par un caractère d'espacement.
- **`current_prefix`** *(string)* : Préfixe en cours de frappe identifié pour l'autocomplétion.
- **`context_words`** *(array[string])* : Historique des mots stabilisés utilisés pour le conditionnement n-gramme.
- **`completions`** *(array[object])* : Liste des candidats complétant le préfixe courant :
  - `word` : Mot complet proposé.
  - `prob` : Probabilité brute calculée par le modèle avec lissage add-k.
  - `norm_score` : Score relatif normalisé sur 100 parmi les $k$ candidats sélectionnés.
  - `tier` : Niveau hiérarchique du modèle ayant validé le mot (3 pour Trigramme, 2 pour Bigramme, 1 pour Unigramme).
  - `source` : Libellé textuel de la source (`3-gram`, `2-gram`, `1-gram`).
  - `prefix` : Fragment déjà saisi par l'utilisateur.
  - `suffix` : Fragment manquant à insérer pour compléter le mot.
- **`next_words`** *(array[object])* : Liste des mots suivants les plus probables :
  - `word` : Mot suivant prédit (ou `</s>` pour marquer la fin de phrase).
  - `display` : Texte d'affichage convivial (`Fin de phrase` si `word == "</s>"`).
  - `prob` : Probabilité brute issue du modèle.
  - `norm_score` : Score de confiance normalisé sur 100.
  - `is_eos` : Booléen signalant s'il s'agit du token de fin de phrase (*End Of Sentence*).
  - `source` : Ordre n-gramme ayant fourni la prédiction après repli (*back-off*).
- **`order`** *(string)* : Ordre n-gramme retenu pour la prédiction (`3-gram`, `2-gram` ou `1-gram`).
- **`vocab_size`** *(integer)* : Taille totale du vocabulaire ($|V|$) en mémoire.
- **`latency_ms`** *(float)* : Temps d'exécution du calcul en millisecondes.

#### Réponse si les modèles ne sont pas encore prêts

```json
{
  "error": "Models not loaded yet",
  "completions": [],
  "next_words": []
}
```

---

## Codes de Statut et d'Erreurs

| Code HTTP | Signification | Description |
| :--- | :--- | :--- |
| `200 OK` | Succès | Requête traitée avec succès, payload JSON retourné. |
| `404 Not Found` | Route introuvable | Le chemin demandé ne correspond ni à `/api/health` ni à `/api/suggest`. |
| `500 Internal Server Error` | Erreur interne | Exception inattendue lors du traitement Python. |

---
[⬅ Retour au README principal](../README.md)
