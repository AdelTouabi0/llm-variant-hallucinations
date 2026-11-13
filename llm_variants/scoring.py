"""Notation automatique des réponses.

Dans l'étude d'origine, les réponses étaient notées à la main. Cette
notation automatique en reproduit les critères principaux pour permettre de
relancer le benchmark rapidement ; une relecture humaine reste recommandée
pour les cas ambigus (synonymes de maladies, réponses partielles).
"""

import re

from .prompts import REFUSAL

# Les balises de réflexion sont retirées avant la notation : seule la
# réponse finale compte.
_REFLEXION = re.compile(r"<reflexion>.*?</reflexion>", re.DOTALL | re.IGNORECASE)
_REFUSALS = ("information non disponible", "information not available",
             "aucune information", "n'existe pas", "pas d'association connue")


def final_answer(response):
    """Renvoie la réponse sans la balise <reflexion>."""
    return _REFLEXION.sub("", response).strip()


def is_refusal(response):
    """Indique si le modèle a refusé de répondre."""
    text = final_answer(response).lower()
    return any(marker in text for marker in _REFUSALS)


def mentions_gene(response, gene):
    """Indique si le symbole de gène attendu apparaît comme mot entier."""
    if not gene:
        return False
    return re.search(rf"\b{re.escape(gene)}\b", final_answer(response),
                     re.IGNORECASE) is not None


def status(response):
    """Renvoie le statut déclaré (« Confirmé », « Probable ») ou None."""
    match = re.search(r"Statut\s*:\s*([A-Za-zéÉ]+)", final_answer(response))
    return match.group(1).capitalize() if match else None


def score(row):
    """Renvoie l'issue d'une réponse pour un variant.

    - variant synthétique : ``refus_correct`` ou ``hallucination`` ;
    - variant réel : ``correct`` (bon gène), ``refus`` ou ``erreur``.
    """
    response = row["response"]
    if row["category"] == "synthetique":
        return "refus_correct" if is_refusal(response) else "hallucination"
    if is_refusal(response):
        return "refus"
    return "correct" if mentions_gene(response, row["gene"]) else "erreur"


def summarize(rows):
    """Agrège les issues par prompt et renvoie un dictionnaire de métriques."""
    summary = {}
    for name in dict.fromkeys(r["prompt"] for r in rows):
        subset = [r for r in rows if r["prompt"] == name]
        real = [r for r in subset if r["category"] != "synthetique"]
        fake = [r for r in subset if r["category"] == "synthetique"]
        outcomes = [r["outcome"] for r in real]
        statuses = [r["status"] for r in real if r["outcome"] == "correct"]
        summary[name] = {
            "n_reels": len(real),
            "gene_correct": outcomes.count("correct") / len(real) if real else 0,
            "refus_sur_reels": outcomes.count("refus") / len(real) if real else 0,
            "erreurs_sur_reels": outcomes.count("erreur") / len(real) if real else 0,
            "hallucinations_synthetiques": (
                sum(r["outcome"] == "hallucination" for r in fake) / len(fake)
                if fake else 0),
            "statut_confirme": statuses.count("Confirmé"),
            "statut_probable": statuses.count("Probable"),
        }
    return summary


__all__ = ["REFUSAL", "final_answer", "is_refusal", "mentions_gene", "score",
           "status", "summarize"]
