"""
Automated validation of benchmark routines (EXP-01, EXP-02, EXP-03).
"""

import pytest
from src.experiments.benchmark_dp import run_dp_mean_benchmark
from src.experiments.benchmark_kanon import run_kanon_benchmark
from src.experiments.benchmark_range import run_range_queries_benchmark


def test_dp_mean_benchmark_fast():
    df_res = run_dp_mean_benchmark(epsilons=[0.1, 1.0], n_repetitions=50)
    assert len(df_res) == 2
    assert "scale_b_all" in df_res.columns
    assert "scale_b_onco" in df_res.columns

    # Small oncology group noise scale must be significantly higher than all patients
    for _, row in df_res.iterrows():
        assert row["scale_b_onco"] > row["scale_b_all"]
        assert row["emp_mae_onco"] > row["emp_mae_all"]


def test_kanon_benchmark_fast():
    df_res = run_kanon_benchmark(k_values=[3, 5])
    assert len(df_res) == 2
    assert "info_loss_ncp" in df_res.columns
    assert "max_linkage_risk" in df_res.columns

    # As k increases from 3 to 5, max linkage risk must decrease from 1/3 to 1/5
    row_k3 = df_res[df_res["k"] == 3].iloc[0]
    row_k5 = df_res[df_res["k"] == 5].iloc[0]
    assert row_k5["max_linkage_risk"] < row_k3["max_linkage_risk"]
    assert row_k5["min_class_size"] >= 5


def test_range_query_benchmark_fast():
    df_res = run_range_queries_benchmark(query_counts=[1, 5], n_repetitions=30)
    assert len(df_res) == 2

    # For DP, noise scale b must grow with Q
    row_q1 = df_res[df_res["num_queries_Q"] == 1].iloc[0]
    row_q5 = df_res[df_res["num_queries_Q"] == 5].iloc[0]
    assert row_q5["dp_noise_scale_b"] > row_q1["dp_noise_scale_b"]

    # For k-anonymity, error must be constant
    assert row_q1["kanon_constant_mae"] == row_q5["kanon_constant_mae"]
