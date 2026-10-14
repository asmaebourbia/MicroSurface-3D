# Protocole d'évaluation pour un petit rapport scientifique

## Questions de recherche
- RQ1 : le cGAN peut-il générer des cartes de hauteurs dont la pente RMS et l'anisotropie suivent une condition fournie ?
- RQ2 : la contrainte géométrique différentiable améliore-t-elle la fidélité aux conditions sans effondrement de la diversité ?
- RQ3 : dans un contexte idéal de stéréophotométrie calibrée, avec quel degré de précision peut-on récupérer les normales et une hauteur périodique ?

## Baselines et ablations
- Baseline procédurale : surfaces gaussiennes anisotropes (source des données).
- A0 : cGAN avec `--lambda-physics 0`.
- A1 : cGAN avec `--lambda-physics 5`.
- A2 : différentes forces de perte (`2`, `10`) uniquement si le temps le permet.
- Photometric : bruit 0, 0.002 et 0.01 pour comparer la robustesse.
- GAN : trois seeds indépendants souhaitables pour des conclusions solides.

## Maillages
- Comparer le raffinement adaptatif Delaunay et le maillage régulier à budget de sommets proche.
- Rapporter RMSE d'interpolation et nombre de triangles ; ne pas généraliser à une surface 3D non représentable par une heightmap.

## Métriques
- Pente RMS cible/générée : erreur absolue moyenne.
- Ratio anisotropie cible/générée : erreur absolue moyenne et erreur logarithmique.
- Histograms d'orientations : divergence Jensen-Shannon (proxy, pas NDF calibrée).
- Diversité : distance quadratique moyenne entre cartes produites sous condition fixe.
- Photometric : MAE angulaire des normales en degrés ; RMSE des hauteurs après alignement des moyennes.
- Pruning : densité des poids, variation des métriques et, pour une étude sérieuse, **mesures de latence sur GPU**.

## Précautions
- Jeu de données entièrement synthétique et simple : risque de sur-estimer la transférabilité vers des surfaces réelles.
- N'utiliser ni PSNR ni SSIM entre des surfaces de hauteurs non appariées comme critère principal.
- Ne pas appeler les résultats d'un surrogate GGX « mesures BRDF ».
- Ne pas assimiler maillage triangulé régulier et maillage adaptatif.
- Déclarer les limites et éviter une conclusion de type « résolution du problème inverse optique ».

## Plan du mini-rapport
1. Contexte scientifique et objectif.
2. Génération des données et définition des unités.
3. Architecture du cGAN et pertes.
4. Photometric Stereo.
5. Critères d'évaluation et ablations.
6. Résultats (tableau rempli **après** entraînement).
7. Limites / perspectives vers BRDF conditionnée et matériaux réels.
