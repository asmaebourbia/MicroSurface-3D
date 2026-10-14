# Provenance des résultats / Results provenance

Les huit images de `docs/figures/`, les trois JSON de `results/metrics/`, le CSV et le NPZ de `results/samples/` proviennent des sorties communiquées par l'utilisatrice pour MicroSurface-3D. Le graphique `anisotropy_fidelity.png` est **nouveau**, calculé uniquement à partir du CSV `sample_metrics.csv` corrigé.

## Important : identité du run à confirmer avant publication

Le log d'évaluation du job `23276` communiqué précédemment rapportait notamment :

- `rms_slope_mae = 0.007341733202338219`
- `anisotropy_ratio_mae = 0.056647252291440964`
- `normal_orientation_js_proxy = 0.0012115943105499543`

Les fichiers JSON téléchargés ultérieurement affichent respectivement **0.006805849**, **0.057130471**, **0.001125638**. Il s'agit donc d'un export d'évaluation **distinct ou recalculé** ; la provenance exacte du checkpoint, du job et des seeds n'est pas attestée par les fichiers seuls.

**Avant de publier un résultat comme provenant d'un job nommé, associer** : ID SLURM, commit Git, seed d'entraînement, chemin du checkpoint et métriques exportées dans un `run_manifest.json`. Ne pas amalgamer les deux sorties numériques.

Les statistiques figurant dans les READMEs utilisent **uniquement les trois JSON effectivement joints à ce paquet**, pas celles du log `23276`.

## Correction CSV

L'ancien CSV utilisait les intitulés :

```text
target_slope,generated_slope,target_anis,generated_anis
```

Les valeurs étaient écrites sous la forme `[*c,*m]`, avec `c=(target_slope,target_anis)` et `m=(generated_slope,generated_anis)`. L'ordre exact est donc :

```text
target_slope,target_anis,generated_slope,generated_anis
```

La version publiée corrige le CSV **et** `microsurface/evaluate.py`. Les valeurs elles-mêmes ne changent pas.

## Limites

Les images de surfaces et de normales sont des sorties de données synthétiques, les figures GGX sont des *surrogates* analytiques et les reconstructions par stéréophotométrie ont été obtenues dans un scénario lambertien idéal. Aucune propriété de matériaux optiques réels n'a été mesurée.
