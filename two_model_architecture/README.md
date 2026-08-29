# Procédé 2: Architecture à deux modèles via Langage Abstrait (LA)

**Status:** Concept + local experimental baseline active
**Source:** e-Soleau (Section 4.2)

## 1. Principe Général

Ce procédé découple la **traduction** (langage naturel ↔ langage interne) du **raisonnement** (langage interne ↔ langage interne). Il introduit un **Langage Abstrait (LA)** comme pivot standardisé entre plusieurs systèmes.

## 2. Architecture Cible

Le développement futur devra implémenter trois modules distincts :

### Module A : Encodeur LA (Traducteur Entrée)
*   **Entrée :** Langage naturel (Texte).
*   **Fonction :** Encode l'entrée et la quantifie en une séquence de codes discrets (le Langage Abstrait).
*   **Sortie :** Séquence LA.

### Module B : Moteur de Raisonnement (Interne ou Externe)
*   **Entrée :** Séquence LA (État initial).
*   **Fonction :** Transforme logiquement la séquence LA en une nouvelle séquence LA (État final / Solution).
*   **Sortie :** Séquence LA transformée.
*   *Note :* Ce module peut être un simple Transformeur "Decoder-only" entraîné uniquement sur des séquences de codes, ou un système symbolique externe.

### Module C : Décodeur LA (Traducteur Sortie)
*   **Entrée :** Séquence LA.
*   **Fonction :** Traduit le LA en langage naturel ou structuré (JSON, Code).
*   **Sortie :** Texte lisible.

## 3. Stratégie d'Apprentissage (Guidelines)

1.  **Phase 1 : Apprentissage Traductionnel (Auto-encodeur disjoint)**
    *   Entraîner Module A + Module C ensemble pour reconstruire du texte via le goulot LA.
    *   Objectif : S'assurer que le LA capture toute l'information sémantique nécessaire.

2.  **Phase 2 : Apprentissage du Raisonnement**
    *   Entraîner Module B sur des paires `(LA_problème, LA_solution)`.
    *   Module A et C peuvent être gelés (frozen) durant cette phase.

## 4. Intérêt Technique & Industriel
*   **Modularité :** Permet de changer le moteur de raisonnement sans ré-entraîner la compréhension du langage.
*   **Sécurité :** Le "cœur" du raisonnement (Module B) peut être propriétaire et caché derrière une API qui n'expose que les traducteurs.
*   **Multi-lingue :** Un seul moteur de raisonnement (B) peut servir plusieurs langues si des couples (A, C) sont entraînés pour chaque langue vers le même LA pivot.

## 5. Local Experimental Track (Current)

The first rigorous local run for this architecture is:

- Script: `two_model_architecture/tm_exp01_reasoner_local.py`
- Run notes: `two_model_architecture/TM_EXP01_NOTES.md`
- Experiment log: `two_model_architecture/TM_LOCAL_EXPERIMENT_LOG.md`
- Comparison run notes: `two_model_architecture/TM_EXP04_NOTES.md`
- OOD sweep notes: `two_model_architecture/TM_EXP05_NOTES.md`
- Strong baseline notes: `two_model_architecture/TM_EXP07_NOTES.md`
- Factorized LA notes: `two_model_architecture/TM_EXP08_NOTES.md`
- Subtraction curriculum notes: `two_model_architecture/TM_EXP09_NOTES.md`
- Scheduled sampling notes: `two_model_architecture/TM_EXP10_NOTES.md`
- Outperform signal status: `two_model_architecture/TM_OUTPERFORM_SIGNAL.md`
- Outputs: `two_model_architecture/results/tm_exp*`

Design used in TM-EXP-01:
- `A`: deterministic text -> LA parser (setup-controlled).
- `B`: learned LA -> LA reasoner.
- `C`: deterministic LA -> text answer renderer.

Scientific protocol:
- preregistered hypothesis and acceptance criteria,
- mandatory setup validation (split/parser/target checks),
- causal controls (`shuffle`, `drop`, `random_b`),
- multi-seed reproducibility check.
