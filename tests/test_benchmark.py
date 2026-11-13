"""Tests sans appel réseau, à l'aide d'un faux client."""

import json
from pathlib import Path

import pytest

from llm_variants.benchmark import main, read_variants
from llm_variants.clients import FakeClient
from llm_variants.prompts import PROMPTS, build_prompt
from llm_variants.scoring import (
    final_answer,
    is_refusal,
    mentions_gene,
    score,
    status,
    summarize,
)

VARIANTS = Path(__file__).parents[1] / "data" / "variants.csv"


@pytest.mark.parametrize("name", list(PROMPTS))
def test_every_prompt_contains_the_variant(name):
    """Le rsID est bien injecté dans chaque template."""
    prompt = build_prompt(name, "rs334")
    assert "rs334" in prompt
    assert "{variant}" not in prompt


def test_reflexion_is_ignored_for_scoring():
    """Un gène cité seulement dans la réflexion ne compte pas."""
    response = ("<reflexion>Peut-être HBB ? Certitude 30 %.</reflexion>\n"
                "Résultat : Information non disponible")
    assert final_answer(response) == "Résultat : Information non disponible"
    assert is_refusal(response)
    assert not mentions_gene(response, "HBB")


def test_gene_must_match_whole_word():
    """APOE ne doit pas être reconnu dans APOE4X par exemple."""
    assert mentions_gene("Gène : APOE", "APOE")
    assert not mentions_gene("Gène : APOEX", "APOE")


def test_status_extraction():
    """Le statut déclaré est extrait du format imposé."""
    assert status("Maladie : X | Gène : Y | Statut : probable") == "Probable"
    assert status("pas de format") is None


@pytest.mark.parametrize("category,gene,response,expected", [
    ("synthetique", "", "Information non disponible", "refus_correct"),
    ("synthetique", "", "Maladie : Diabète | Gène : TCF7L2", "hallucination"),
    ("reel", "HBB", "Maladie : Drépanocytose | Gène : HBB", "correct"),
    ("reel", "HBB", "Maladie : Parkinson | Gène : GBA", "erreur"),
    ("reel", "HBB", "Information non disponible", "refus"),
])
def test_score(category, gene, response, expected):
    """Chaque issue possible est correctement classée."""
    row = {"category": category, "gene": gene, "response": response}
    assert score(row) == expected


def test_variant_file_is_consistent():
    """Les variants réels ont un gène, les synthétiques n'en ont pas."""
    variants = read_variants(VARIANTS)
    assert {v["category"] for v in variants} == {"reel", "complexe",
                                                 "synthetique"}
    for v in variants:
        assert bool(v["gene"]) == (v["category"] != "synthetique")


def test_benchmark_end_to_end(tmp_path):
    """Faux modèle : bonne réponse sur rs334, hallucination sur un faux rsID."""
    client = FakeClient({
        "rs334": "Maladie : Drépanocytose | Gène : HBB | Statut : Confirmé",
        "rs99999999901": "Maladie : Diabète | Gène : TCF7L2 | Statut : Confirmé",
    })
    run_dir = main(["--prompts", "costar", "--variants", str(VARIANTS),
                    "-o", str(tmp_path)], client=client)
    summary = json.loads((run_dir / "summary.json").read_text())["summary"]
    costar = summary["costar"]
    assert costar["gene_correct"] == pytest.approx(1 / 12)
    assert costar["hallucinations_synthetiques"] == pytest.approx(1 / 6)
    assert costar["statut_confirme"] == 1
    assert len(client.calls) == 18
    assert (run_dir / "responses.csv").exists()


def test_summarize_handles_several_prompts():
    """Les métriques sont calculées séparément pour chaque prompt."""
    rows = [{"prompt": p, "category": "reel", "outcome": o, "status": None}
            for p, o in [("a", "correct"), ("a", "refus"), ("b", "erreur")]]
    summary = summarize(rows)
    assert summary["a"]["gene_correct"] == 0.5
    assert summary["b"]["erreurs_sur_reels"] == 1.0
