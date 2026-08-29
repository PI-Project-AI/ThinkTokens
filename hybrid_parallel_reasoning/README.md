# Procédé 3: Raisonnement parallèle IR + CoT en langage naturel

**Status:** Future Development / Conceptual
**Source:** e-Soleau (Section 4.3)

## 1. Principe Général

Ce procédé combine un **flux de raisonnement interne** (IR / Machine-Native) et un **flux de raisonnement externe** (CoT / Human-Readable). L'objectif est de laisser le modèle apprendre *où et quand* utiliser le langage naturel (pour l'explication) versus le langage interne (pour l'efficacité).

## 2. Architecture Cible

Le modèle doit posséder deux "pistes" (tracks) de traitement simultanées ou entrelacées :

### Piste A : Chain-of-Thought (CoT) Textuelle
*   **Nature :** Tokens de vocabulaire standard (Anglais/Français).
*   **Rôle :** Explicabilité, interaction utilisateur, structuration "humaine" du problème.
*   **Supervision :** Entraîné sur des traces de raisonnement humaines (datasets CoT classiques).

### Piste B : Internal Reasoning (IR) Discret
*   **Nature :** Jetons de Raisonnement Interne (JRI) issus d'un codebook (VQ).
*   **Rôle :** Calcul dense, gestion d'états complexes, "intuition" machine.
*   **Supervision :** Émergente (via reconstruction/prédiction) ou alignée sur la piste A.

## 3. Mécanismes de Couplage (Guidelines)

L'innovation réside dans l'interaction entre ces deux pistes :

1.  **Attention Croisée (Cross-Attention) :**
    *   La piste CoT peut "lire" l'état de la piste IR pour générer une explication.
    *   La piste IR peut "lire" la piste CoT pour s'ancrer dans les contraintes explicites.

2.  **Pertes de Consistance (Consistency Losses) :**
    *   Entraîner le modèle pour que `Résultat(IR)` soit cohérent avec `Résultat(CoT)`.
    *   *Test de Causalité :* Si on masque la piste IR, la qualité de la piste CoT (ou de la réponse finale) doit se dégrader, prouvant que l'IR n'est pas décoratif.

## 4. Stratégie d'Utilisation

*   **Mode "Explicable" :** Le modèle génère les deux pistes. L'utilisateur voit le CoT.
*   **Mode "Turbo" / "Privé" :** Le modèle n'active que la piste IR pour calculer la réponse finale. Le CoT est supprimé pour économiser des tokens et de la latence.

## 5. Intérêt Technique & Industriel
*   Combine le meilleur des deux mondes : la **performance** des représentations machines et la **confiance** des explications humaines.
*   Permet une **flexibilité de déploiement** (choix du coût/latence au moment de l'inférence).
