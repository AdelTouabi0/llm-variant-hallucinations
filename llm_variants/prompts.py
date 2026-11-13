"""Les quatre architectures de prompt comparées dans l'étude.

Chaque template contient le marqueur ``{variant}``, remplacé par le rsID.
"""

REFUSAL = "Information non disponible"

NAIVE = "Quelles sont les maladies associées au variant génétique {variant} ?"

FEW_SHOT = f"""Tu annotes des variants génétiques. Réponds sur une ligne au format :
Maladie : <maladie> | Gène : <gène> | Statut : <statut>
Si tu n'as pas d'information fiable, réponds exactement : {REFUSAL}

Exemple 1
Entrée : rs6025
Résultat : Maladie : Thrombophilie (facteur V Leiden) | Gène : F5 | Statut : Confirmé

Exemple 2
Entrée : rs99999999999
Résultat : {REFUSAL}

Entrée : {{variant}}
Résultat :"""

COT_4D = f"""[RÔLE] Tu es un bio-informaticien clinicien.
[CONTEXTE] Extraction d'associations fiables entre variants génétiques et
phénotypes, pour un projet universitaire.
[TÂCHE]
1. Identifier le gène du variant.
2. Déterminer la pathologie avérée.
3. Formater la réponse.
[ESPACE POUR RÉFLÉCHIR]
Ouvre une balise <reflexion>...</reflexion> pour analyser le variant avant de
répondre. Évalue ta certitude en pourcentage.
[FORMAT]
Résultat : Maladie : <maladie> | Gène : <gène> | Statut : <Confirmé ou Probable>
[CONTRAINTES]
Si tu doutes, ou si le variant n'existe pas, réponds :
Résultat : {REFUSAL}
Variant ciblé : {{variant}}"""

COSTAR = f"""# CONTEXT
Projet universitaire de bio-informatique sur l'annotation de variants.
# OBJECTIVE
Identifier le gène et la pathologie associés au variant {{variant}}.
# STYLE
Base de données clinique stricte, de type ClinVar.
# TONE
Froid, factuel, extrêmement prudent.
# AUDIENCE
Jury universitaire de bio-informaticiens.
# RESPONSE
Une seule ligne, au format strict :
Maladie : <maladie> | Gène : <gène> | Statut : Confirmé
ou, en l'absence d'information fiable : {REFUSAL}"""

PROMPTS = {"naif": NAIVE, "few_shot": FEW_SHOT, "cot_4d": COT_4D,
           "costar": COSTAR}


def build_prompt(name, variant):
    """Renvoie le prompt ``name`` instancié pour le variant donné."""
    if name not in PROMPTS:
        raise KeyError(f"Prompt inconnu : {name} (choix : {list(PROMPTS)})")
    return PROMPTS[name].format(variant=variant)
