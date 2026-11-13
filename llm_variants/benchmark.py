r"""Lance le benchmark : chaque prompt x chaque variant, un appel par paire.

Exemples :
    python -m llm_variants.benchmark --provider dry-run
    OPENAI_API_KEY=... python -m llm_variants.benchmark --provider openai \\
        --prompts cot_4d costar --temperature 0.1
    HF_TOKEN=... python -m llm_variants.benchmark --provider hf \\
        --model Qwen/Qwen2.5-7B-Instruct --prompts naif few_shot cot_4d
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .clients import complete_with_delay, make_client
from .prompts import PROMPTS, build_prompt
from .scoring import score, status, summarize

matplotlib.use("Agg")


def read_variants(path):
    """Lit le CSV des variants (variant, category, gene, disease)."""
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(client, variants, prompt_names, delay=0.0):
    """Interroge le modèle pour chaque paire (prompt, variant).

    Chaque requête est indépendante : aucun historique n'est conservé entre
    deux variants, pour qu'une réponse n'influence pas la suivante.
    """
    rows = []
    for name in prompt_names:
        for variant in variants:
            prompt = build_prompt(name, variant["variant"])
            response = complete_with_delay(client, prompt, delay)
            row = {"prompt": name, **variant, "response": response}
            row["outcome"] = score(row)
            row["status"] = status(response)
            rows.append(row)
            print(f"  [{name}] {variant['variant']:<14} -> {row['outcome']}")
    return rows


def plot_summary(summary, output_path, title):
    """Diagramme en barres groupées des principales métriques."""
    metrics = {"gene_correct": "Gène correct (réels)",
               "refus_sur_reels": "Refus (réels)",
               "hallucinations_synthetiques": "Hallucination (synthétiques)"}
    names = list(summary)
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(1.8 * len(names) + 3, 4.5))
    for i, (key, label) in enumerate(metrics.items()):
        values = [summary[n][key] for n in names]
        ax.bar(x + (i - 1) * width, values, width, label=label)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    """Lit les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=["openai", "hf", "dry-run"],
                        default="dry-run")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS),
                        default=list(PROMPTS))
    parser.add_argument("--variants", type=Path,
                        default=Path("data/variants.csv"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--delay", type=float, default=1.5,
                        help="pause entre deux requêtes, en secondes")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def main(argv=None, client=None):
    """Exécute le benchmark et écrit réponses, métriques et figure."""
    args = parse_args(argv)
    client = client or make_client(args.provider, args.model, args.temperature)
    variants = read_variants(args.variants)
    model_name = getattr(client, "model", args.provider)
    print(f"Modèle {model_name}, {len(variants)} variants, "
          f"prompts {args.prompts}")
    delay = 0.0 if args.provider == "dry-run" else args.delay
    rows = run(client, variants, args.prompts, delay)

    run_dir = args.output / (model_name.replace("/", "_") + "_"
                             + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    # utf-8-sig : le CSV s'ouvre avec les accents corrects dans Excel.
    with open(run_dir / "responses.csv", "w", newline="",
              encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    meta = {"model": model_name, "provider": args.provider,
            "temperature": args.temperature, "summary": summary}
    (run_dir / "summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_summary(summary, run_dir / "summary.png", f"{model_name}")
    for name, metrics in summary.items():
        print(f"{name:<10} gène correct {metrics['gene_correct']:.0%} | "
              f"refus {metrics['refus_sur_reels']:.0%} | hallucinations "
              f"synthétiques {metrics['hallucinations_synthetiques']:.0%}")
    print(f"Résultats écrits dans {run_dir}/")
    return run_dir


if __name__ == "__main__":
    main()
