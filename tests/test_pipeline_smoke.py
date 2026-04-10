"""
test_pipeline_smoke.py
======================
Smoke test: verifies the full pipeline can run end-to-end
with simulated data without raising exceptions.

Run with:
    pytest tests/test_pipeline_smoke.py -v
"""

import os
import sys
import pytest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WORKSPACE, "src"))


def test_simulate_data(tmp_path):
    """Test that simulated h5ad data can be generated."""
    from data.simulate_data import generate_baseline_data, generate_intervention_data

    out_dir = str(tmp_path / "simulated")
    adatas, combined = generate_baseline_data(out_dir)
    assert combined.n_obs > 0
    assert combined.n_vars == 500

    pre, post = generate_intervention_data(out_dir)
    assert pre.n_obs > 0
    assert post.n_obs > 0


def test_rank_token_dataset(tmp_path):
    """Test Rank-Token DataLoader construction."""
    import numpy as np
    import anndata as ad
    import scipy.sparse as sp
    from data.rank_token_dataset import build_gene_vocab, build_dataloaders
    from data.simulate_data import generate_baseline_data

    out_dir = str(tmp_path / "simulated")
    _, adata = generate_baseline_data(out_dir)

    gene_vocab = build_gene_vocab(adata.var_names.tolist())
    train_loader, val_loader, split_info = build_dataloaders(
        adata, gene_vocab, top_n=64, batch_size=8
    )
    batch = next(iter(train_loader))
    assert batch["token_ids"].shape[1] == 64
    assert batch["age"].shape[0] == min(8, split_info["train_cells"])


def test_model_forward():
    """Test model forward pass produces correct output shapes."""
    import torch
    from model.rank_transformer import PBMCGPTConfig, PBMCGPTModel

    config = PBMCGPTConfig.dummy(vocab_size=502)
    model = PBMCGPTModel(config)
    model.eval()

    B, L = 2, 256
    token_ids = torch.randint(2, 502, (B, L))
    expr_ranks = torch.rand(B, L)

    with torch.no_grad():
        out = model(token_ids, expr_ranks, output_attentions=True)

    assert out["age_pred"].shape == (B,)
    assert out["cls_vector"].shape == (B, config.hidden_size)
    assert out["last_hidden"].shape == (B, L + 1, config.hidden_size)
    assert len(out["attentions"]) == config.num_hidden_layers
