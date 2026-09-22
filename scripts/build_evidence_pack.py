"""Build a compact assessor-facing evidence pack from executed RAG results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
EVIDENCE_DIR = Path("evidence")
EVIDENCE_TABLES = EVIDENCE_DIR / "tables"


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def fmt(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    EVIDENCE_TABLES.mkdir(parents=True, exist_ok=True)

    selected = [
        "phase1_bm25_demo.csv",
        "phase2_retrieval_comparison.csv",
        "phase2_scifact_comparison.csv",
        "phase3_reranking_demo.csv",
        "phase3_real_rag.csv",
    ]
    for name in selected:
        copy_if_exists(TABLES_DIR / name, EVIDENCE_TABLES / name)

    lines = [
        "# Real Experiment Results",
        "",
        "This evidence pack was generated from executed retrieval/RAG workflows.",
        "",
    ]

    phase1 = TABLES_DIR / "phase1_bm25_demo.csv"
    if phase1.exists():
        frame = pd.read_csv(phase1)
        lines.extend(
            [
                "## BM25 lexical retrieval",
                "",
                f"- Mean Recall@1: **{fmt(frame['recall_at_1'].mean())}**",
                f"- Mean Recall@3: **{fmt(frame['recall_at_3'].mean())}**",
                f"- Mean Precision@1: **{fmt(frame['precision_at_1'].mean())}**",
                f"- Mean reciprocal rank: **{fmt(frame['reciprocal_rank'].mean())}**",
                "",
            ]
        )

    phase2 = TABLES_DIR / "phase2_retrieval_comparison.csv"
    if phase2.exists():
        frame = pd.read_csv(phase2)
        lines.extend(["## BM25 vs dense semantic retrieval", ""])
        for row in frame.itertuples(index=False):
            lines.append(
                f"- {row.retriever}: Recall@1={fmt(row.recall_at_1)}, "
                f"Recall@3={fmt(row.recall_at_3)}, "
                f"MRR={fmt(row.reciprocal_rank)}"
            )
        lines.append("")

    scifact = TABLES_DIR / "phase2_scifact_comparison.csv"
    if scifact.exists():
        frame = pd.read_csv(scifact)
        lines.extend(["## Public SciFact benchmark", ""])
        for row in frame.itertuples(index=False):
            lines.append(
                f"- {row.retriever}: Recall@10={fmt(row.recall_at_10)}, "
                f"Precision@3={fmt(row.precision_at_3)}, "
                f"MRR={fmt(row.reciprocal_rank)}"
            )
        lines.append("")

    real_rag = TABLES_DIR / "phase3_real_rag.csv"
    if real_rag.exists():
        frame = pd.read_csv(real_rag)
        lines.extend(
            [
                "## Real cross-encoder + generator run",
                "",
                f"- Mean Recall@1: **{fmt(frame['recall_at_1'].mean())}**",
                f"- Mean MRR: **{fmt(frame['reciprocal_rank'].mean())}**",
                f"- Mean citation precision: **{fmt(frame['citation_precision'].mean())}**",
                f"- Unsupported-citation rate: **{fmt(frame['unsupported_citation'].mean())}**",
                f"- Abstention rate: **{fmt(frame['abstained'].mean())}**",
                "",
                "### Generated answers",
                "",
            ]
        )
        for row in frame.itertuples(index=False):
            lines.append(f"- {row.query_id}: {row.answer}")
        lines.append("")

    summary = {
        "project": "am2-03-modern-rag-benchmark",
        "bm25_executed": phase1.exists(),
        "dense_executed": phase2.exists(),
        "scifact_executed": scifact.exists(),
        "real_rag_executed": real_rag.exists(),
    }
    (EVIDENCE_DIR / "final_project_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines.extend(
        [
            "## Evidence files",
            "",
            "- tables/phase1_bm25_demo.csv",
            "- tables/phase2_retrieval_comparison.csv",
            "- tables/phase2_scifact_comparison.csv when enabled",
            "- tables/phase3_reranking_demo.csv",
            "- tables/phase3_real_rag.csv when full models are enabled",
            "- final_project_summary.json",
            "",
            "Generated automatically by scripts/build_evidence_pack.py.",
        ]
    )

    (EVIDENCE_DIR / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Evidence pack written to {EVIDENCE_DIR.resolve()}")


if __name__ == "__main__":
    main()
