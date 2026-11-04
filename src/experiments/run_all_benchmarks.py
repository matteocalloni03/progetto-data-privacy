"""
Master Benchmark Runner and Figure Generator.
Executes all benchmarks and generates Figure 2(a), Figure 2(b), Figure 3, and Figure 4.
Saves CSV result tables and high-resolution figures in data/ and data/figures/.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.config import FIGURES_DIR, DATA_DIR
from src.experiments.benchmark_dp import run_dp_mean_benchmark
from src.experiments.benchmark_kanon import run_kanon_benchmark
from src.experiments.benchmark_range import run_range_queries_benchmark


def plot_figure_2(df_dp: pd.DataFrame, df_kanon: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate Figure 2:
    (a) Error of the DP mean versus epsilon, for all patients (N=1000) and small oncology group (N=64).
    (b) k-Anonymity: Information loss (NCP) and maximum linkage risk versus k.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot (a): DP Error vs Epsilon
    ax1.plot(
        df_dp["epsilon"], df_dp["emp_mae_all"],
        marker="o", color="#1f77b4", linewidth=2, label="All patients (N=1000) - MAE"
    )
    ax1.plot(
        df_dp["epsilon"], df_dp["emp_mae_onco"],
        marker="s", color="#d62728", linewidth=2, linestyle="--", label="Oncology group (N=64) - MAE"
    )
    ax1.set_title("(a) Error of DP Mean vs Privacy Budget (ε)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Privacy Budget (ε)", fontsize=11)
    ax1.set_ylabel("Mean Absolute Error (mmHg)", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)

    # Subplot (b): k-Anonymity Trade-off vs k
    ax2_twin = ax2.twinx()

    p1 = ax2.plot(
        df_kanon["k"], df_kanon["info_loss_ncp"],
        marker="^", color="#2ca02c", linewidth=2.2, label="Information Loss (NCP)"
    )
    p2 = ax2_twin.plot(
        df_kanon["k"], df_kanon["max_linkage_risk"],
        marker="D", color="#ff7f0e", linewidth=2.2, linestyle="-.", label="Max Linkage Risk (1/k)"
    )

    ax2.set_title("(b) k-Anonymity: Utility Loss vs Linkage Risk", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Anonymity Threshold (k)", fontsize=11)
    ax2.set_ylabel("Information Loss (Normalized Certainty Penalty)", color="#2ca02c", fontsize=11)
    ax2_twin.set_ylabel("Maximum Linkage Risk (1/|EC|)", color="#ff7f0e", fontsize=11)
    ax2.grid(True, ls=":", alpha=0.6)

    # Combined legend for twin axes
    plots = p1 + p2
    labels = [l.get_label() for l in plots]
    ax2.legend(plots, labels, loc="center right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    fig_path = output_dir / "figure_2_combined.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[+] Saved Figure 2 to {fig_path}")


def plot_figure_3(df_range: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate Figure 3:
    Error of range-count queries: DP error grows with query sequence Q,
    whereas k-anonymity error is constant.
    """
    plt.figure(figsize=(8.5, 5.5), dpi=300)

    plt.plot(
        df_range["num_queries_Q"], df_range["dp_empirical_mae"],
        marker="o", color="#d62728", linewidth=2.2, label="Differential Privacy (Split Budget ε_total=1.0)"
    )
    plt.plot(
        df_range["num_queries_Q"], df_range["kanon_constant_mae"],
        color="#1f77b4", linewidth=2.2, linestyle="--", label="k-Anonymity (k=5, Released Table - Constant)"
    )

    plt.title("Figure 3: Range-Count Query Error vs Number of Queries (Q)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Number of Sequential Queries (Q)", fontsize=11)
    plt.ylabel("Mean Absolute Error in Count", fontsize=11)
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")

    plt.tight_layout()
    fig_path = output_dir / "figure_3_range_queries.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[+] Saved Figure 3 to {fig_path}")


def plot_figure_4(output_dir: Path) -> None:
    """
    Generate Figure 4:
    NIS2 Art. 23 reporting timeline compared with GDPR Art. 33 deadline.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)

    # Timeline intervals (in hours)
    milestones = [
        (0, "T0: Incident Awareness\n(Detection / Breach)", "#333333"),
        (24, "24h: NIS2 Art. 23(4)(a)\nEarly Warning to CSIRT (CNCS)", "#e74c3c"),
        (72, "72h: NIS2 Art. 23(4)(b) Notification\n& GDPR Art. 33 DPA (CNPD)", "#f39c12"),
        (720, "1 Month: NIS2 Art. 23(4)(e)\nFinal Incident Report", "#27ae60")
    ]

    # Draw timeline axis
    ax.axhline(0, color="#7f8c8d", linewidth=3, zorder=1)

    positions = [0, 1.5, 3.0, 5.0]
    for pos, (hours, label, color) in zip(positions, milestones):
        ax.scatter(pos, 0, s=180, color=color, zorder=3, edgecolors="black", linewidth=1.5)
        ax.annotate(
            label,
            xy=(pos, 0),
            xytext=(pos, 0.25 if pos % 2 == 0 else -0.35),
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
        )

    ax.set_xlim(-0.8, 5.8)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    ax.set_title("Figure 4: NIS2 Art. 23 Incident Reporting Steps & GDPR Art. 33 Alignment", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    fig_path = output_dir / "figure_4_nis2_timeline.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[+] Saved Figure 4 to {fig_path}")


def run_all() -> None:
    print("[*] Running Benchmark 1: Differential Privacy Mean Error (Table 12 & Fig 2a)...")
    df_dp = run_dp_mean_benchmark()
    dp_csv = DATA_DIR / "benchmark_dp_table12.csv"
    df_dp.to_csv(dp_csv, index=False)
    print(f"[+] Saved Table 12 to {dp_csv}")

    print("[*] Running Benchmark 2: k-Anonymity Trade-off vs k (Table 13 & Fig 2b)...")
    df_kanon = run_kanon_benchmark()
    kanon_csv = DATA_DIR / "benchmark_kanon_table13.csv"
    df_kanon.to_csv(kanon_csv, index=False)
    print(f"[+] Saved Table 13 to {kanon_csv}")

    print("[*] Running Benchmark 3: Range-Count Query Sequence (Fig 3)...")
    df_range = run_range_queries_benchmark()
    range_csv = DATA_DIR / "benchmark_range_queries.csv"
    df_range.to_csv(range_csv, index=False)
    print(f"[+] Saved Range Query Results to {range_csv}")

    print("[*] Rendering High-Resolution Publication Plots...")
    plot_figure_2(df_dp, df_kanon, FIGURES_DIR)
    plot_figure_3(df_range, FIGURES_DIR)
    plot_figure_4(FIGURES_DIR)

    print("\n[SUCCESS] All benchmarks completed and figures generated successfully.")


if __name__ == "__main__":
    run_all()
