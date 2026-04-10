"""Phase 1 launcher - Windows UTF-8 log wrapper"""
import subprocess, sys, os

env = os.environ.copy()
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"

log_path = os.path.join(os.path.dirname(__file__), "outputs", "phase1_run.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# 使用小配置快速验证：128d x 2层, batch=32, epochs=2
cmd = [
    sys.executable, "-c",
    """
import sys, os, torch, torch.nn as nn
sys.path.insert(0, os.getcwd())

from src.model.rank_transformer import PBMCGPTConfig, PBMCGPTModel
from src.data.rank_token_dataset import PBMCRankTokenDataset, build_gene_vocab
from src.model.MLM_head import mask_tokens
import anndata as ad
import numpy as np

print('=' * 60)
print('Phase 1: DAPT (Small Config Test)')
print('=' * 60)

# 加载数据
adata = ad.read_h5ad('data/phase1_corpus.h5ad')
adata.obs['_dummy_age'] = 50.0
adata.obs['_dummy_cell_type'] = 'Unknown'
gene_vocab = build_gene_vocab(adata.var_names.tolist())
vocab_size = len(gene_vocab)

# 小配置模型
config = PBMCGPTConfig(
    vocab_size=vocab_size,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=512,
)
model = PBMCGPTModel(config)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model params: {n_params:,}')

# 添加 MLM Head (不使用 shift)
mlm_head = nn.Linear(config.hidden_size, vocab_size).to('cpu')

dataset = PBMCRankTokenDataset(
    adata, gene_vocab, top_n=256,
    age_col='_dummy_age', cell_type_col='_dummy_cell_type'
)
print(f'Dataset: {len(dataset)} cells, vocab={vocab_size}')

# 简单训练循环
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(mlm_head.parameters()), lr=1e-4
)
loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
model.train()
mlm_head.train()
batch_size = 32
n_epochs = 2

for epoch in range(1, n_epochs + 1):
    total_loss = 0
    n_batches = 0
    for i in range(0, len(dataset), batch_size):
        batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        token_ids = torch.stack([b['token_ids'] for b in batch])
        expr_ranks = torch.stack([b['expr_ranks'] for b in batch])
        
        # MLM 掩码
        masked_ids, labels = mask_tokens(token_ids, vocab_size=vocab_size)
        # labels: (B, L)，掩码位置=原始token，未掩码=-100
        
        attn_mask = model._make_attention_mask(token_ids, 0)
        
        # Forward
        hidden = model.embeddings(masked_ids, expr_ranks)  # (B, L+1, H)
        
        # 移除 CLS 位置，只保留 L 个位置的输出
        hidden = hidden[:, 1:, :]  # (B, L, H)
        
        # MLM Head (不使用 shift，直接预测每个位置)
        logits = mlm_head(hidden)  # (B, L, vocab_size)
        
        # MLM Loss: logits[i] 预测 labels[i]
        B, L, V = logits.shape
        logits_flat = logits.contiguous().view(-1, V)  # (B*L, V)
        labels_flat = labels.contiguous().view(-1)  # (B*L,)
        
        loss = loss_fn(logits_flat, labels_flat).mean()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    print(f'Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.4f}')

# 保存 checkpoint
os.makedirs('checkpoints', exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'mlm_head_state': mlm_head.state_dict(),
    'config': config.__dict__,
    'vocab': gene_vocab,
}, 'checkpoints/phase1_pbmc_base.pt')
print(f'[OK] Saved: checkpoints/phase1_pbmc_base.pt')
""",
]

with open(log_path, "w", encoding="utf-8") as log_f:
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(__file__),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
    )
    proc.wait()

print(f"Phase 1 finished. exit={proc.returncode}. Log: {log_path}")
