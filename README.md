# Prompt engineering contre les hallucinations des LLM en génomique

![tests](https://github.com/AdelTouabi0/llm-variant-hallucinations/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-GPT--4o--mini%20%7C%20Qwen2.5--7B-purple)
![Licence MIT](https://img.shields.io/badge/licence-MIT-green)

> *Benchmarking prompt architectures (naive, few-shot, chain-of-thought,
> CO-STAR) for variant annotation and hallucination resistance on small and
> large language models.*

Un modèle de langage interrogé sur un variant génétique (rsID) répond
presque toujours avec assurance, **même quand le variant n'existe pas**. En
contexte clinique, une annotation inventée est plus dangereuse qu'une absence
de réponse. Cette étude mesure dans quelle mesure la façon de rédiger le
prompt réduit ce risque, selon la taille du modèle.

## Protocole

Pour chaque variant, le modèle doit donner le gène et la pathologie associés,
ou répondre « Information non disponible ». Le jeu de test mélange :

- des variants **réels bien caractérisés** (rs334 / HBB, rs6025 / F5...) ;
- des variants **réels complexes**, aux associations faibles ou débattues ;
- des **rsID synthétiques** qui n'existent pas : pièges à hallucinations.

Quatre architectures de prompt (`llm_variants/prompts.py`) :

| Prompt | Principe |
|---|---|
| Naïf | question brute, sans contrainte |
| Few-Shot | deux exemples annotés, dont un refus |
| CoT / 4D | rôle, tâche découpée, balise `<reflexion>` obligatoire avec auto-évaluation de la certitude, statut « Confirmé » ou « Probable » |
| CO-STAR | Context, Objective, Style, Tone, Audience, Response ; format strict, sans réflexion |

Chaque requête est indépendante (pas d'historique), à température basse
(0,1), avec une pause entre deux appels pour respecter les limites de débit.

## Résultats de l'étude originale

Étude réalisée en avril 2026 (30 variants : 10 bien caractérisés,
10 complexes, 10 synthétiques ; notation manuelle).

**Petit modèle (Qwen-7B)** : score moyen de fiabilité factuelle sur 3 de
1,35 (Naïf), 2,25 (Few-Shot) et 2,55 (CoT).
- Le prompt naïf invente des associations gène-maladie plausibles et
  rédigées sur un ton académique, et « annote » 4 variants synthétiques
  sur 10.
- Le Few-Shot rejette tous les variants synthétiques, mais présente un
  biais de **fuite des exemples** (*few-shot leakage*) : le modèle recopie
  la pathologie d'un exemple du prompt au lieu de mobiliser ses
  connaissances.
- Le CoT supprime les hallucinations au prix du rappel : refus pour 18 des
  19 variants réels. Ses balises de réflexion montrent que le modèle ne
  connaît pas les bons gènes et l'admet.

**Grand modèle (GPT-4o-mini)** : CoT/4D et CO-STAR rejettent tous deux les
10 variants synthétiques. La différence porte sur la **calibration** :
CO-STAR attribue le statut « Confirmé » à 8 variants réels sur 10 quelle que
soit la solidité des preuves, et produit une association erronée grave,
tandis que le CoT/4D nuance correctement sa certitude (score de calibration
de 9/10).

**Conclusion** : le prompt engineering module le comportement du modèle mais
ne remplace pas la connaissance du domaine. Un cadre qui oblige le modèle à
exprimer son incertitude est plus sûr qu'un format rigide. Pour un usage
clinique, la génération augmentée par récupération (RAG, interrogation de
ClinVar) est le complément nécessaire.

## Relancer le benchmark

Ce dépôt automatise l'étude : appels aux API, notation automatique et
figures. Le fichier `data/variants.csv` est un jeu de démonstration
(18 variants) ; les résultats d'une relance ne sont donc pas directement
comparables à ceux de l'étude ci-dessus.

```bash
pip install -r requirements.txt

# Essai à sec, sans clé d'API ni appel réseau
python -m llm_variants.benchmark --provider dry-run

# GPT-4o-mini (Phase 2)
export OPENAI_API_KEY=...
python -m llm_variants.benchmark --provider openai --prompts cot_4d costar

# Qwen2.5-7B via Hugging Face Inference (Phase 1)
export HF_TOKEN=...
python -m llm_variants.benchmark --provider hf --prompts naif few_shot cot_4d
```

Chaque exécution écrit dans `results/<modèle>_<date>/` les réponses brutes
(`responses.csv`), les métriques (`summary.json`) et une figure.

**Notation automatique** (`llm_variants/scoring.py`) : la balise
`<reflexion>` est retirée, puis la réponse est classée en *correct* (bon
symbole de gène), *refus*, *erreur* pour un variant réel, et en
*refus correct* ou *hallucination* pour un variant synthétique. Elle ne
juge que le gène : une relecture humaine reste nécessaire pour la
pathologie et les cas ambigus.

## Limites

- Petit nombre de variants et une seule exécution par condition : les
  écarts observés sont indicatifs, pas statistiquement établis.
- Les connaissances des modèles évoluent d'une version à l'autre ; la
  version exacte du modèle est enregistrée dans chaque `summary.json`.

## Tests

```bash
pytest -q
```

Les tests utilisent un faux client déterministe : aucune clé d'API ni
connexion n'est nécessaire, et l'intégration continue n'appelle jamais de
service payant.

## Contexte

Projet de recherche universitaire (2026), dont le code a été réécrit pour ce
portfolio. Développé avec l'assistance d'outils d'IA (Claude).

**Références** : Brown T. *et al.* (2020), Language models are few-shot
learners, *NeurIPS* ; Wei J. *et al.* (2022), Chain-of-thought prompting
elicits reasoning in large language models, *NeurIPS* ; Landrum M. J.
*et al.* (2018), ClinVar, *Nucleic Acids Research*, 46, D1062-D1067.

## Auteur

Adel Touabi, Master 1 Bio-informatique, Université Paris Cité.
