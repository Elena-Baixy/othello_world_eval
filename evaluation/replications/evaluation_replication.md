# Replication Evaluation

## Reflection

This replication successfully reproduced the key findings from the Othello-GPT circuit analysis work. The experiment demonstrates that a GPT model trained purely on predicting legal Othello moves develops an internal representation of the board state that can be decoded with high accuracy using linear probes.

### What Went Well

1. **Model Loading**: The pre-trained model weights were readily available from HuggingFace, and the TransformerLens library provided excellent tooling for mechanistic interpretability.

2. **Probe Accuracy**: The linear probe achieved 99.64% accuracy at layer 6, consistent with the original findings. The "mine vs theirs" encoding scheme worked effectively across both player turns.

3. **Intervention Results**: The intervention experiments produced clear, interpretable results. Flipping a cell's representation caused the expected changes in move legality predictions (+9.3 for newly legal moves, -8.8 for newly illegal moves).

4. **Reproducibility**: Running the same code multiple times produced consistent results due to the deterministic nature of model inference (no training involved).

### Challenges Encountered

1. **Disk Quota**: Initial attempt to download model weights failed due to disk quota on the default HuggingFace cache directory. Resolved by using `/tmp/hf_cache`.

2. **Package Compatibility**: Some package version conflicts existed (torch version warnings), but did not affect functionality.

3. **Memory Constraints**: Limited analysis to 50 focus games to fit activation cache in GPU memory.

### Deviations from Original

1. Used pre-trained probe instead of training from scratch
2. Focused on linear probe (which achieved excellent results) rather than nonlinear probes
3. Used TransformerLens version of the analysis rather than original codebase

---

## Replication Evaluation — Binary Checklist

### RP1. Implementation Reconstructability

**PASS**

**Rationale**: The experiment can be fully reconstructed from the plan.md and CodeWalkthrough.md files. The plan clearly specifies:
- Model architecture (8-layer GPT, 512-dim hidden, 8 heads)
- Training data (synthetic and championship datasets)
- Methodology (linear/nonlinear probes, intervention experiments)
- Expected metrics (error rates, intervention effects)

The CodeWalkthrough.md provides step-by-step instructions for:
- Environment setup
- Model training (with downloadable checkpoints)
- Probe training
- Intervention and attribution experiments

No significant guesswork was required. Minor ambiguities (e.g., exact hyperparameters for interventions) were resolvable from the provided notebook code.

---

### RP2. Environment Reproducibility

**PASS**

**Rationale**: The environment can be fully restored:
- `environment.yml` provides Conda environment specification
- Pre-trained model weights available from HuggingFace
- Pre-computed game sequences and probes included in repository
- TransformerLens library provides compatible model loading

The only issue encountered was disk quota for HuggingFace cache, which was easily resolved by specifying an alternative cache directory. All required packages (transformer_lens, einops, fancy_einsum, neel-plotly) installed successfully.

---

### RP3. Determinism and Stability

**PASS**

**Rationale**: Results are stable across runs:
- No model training is involved (inference only)
- Model predictions are deterministic for the same inputs
- Probe accuracy measurements are consistent
- Intervention effects produce identical log probability changes

The experiment uses:
- Fixed random seeds (torch.manual_seed(42), np.random.seed(42))
- Pre-computed game sequences (no random sampling)
- Deterministic model inference (torch.set_grad_enabled(False))

Repeated execution of the replication notebook produces identical numerical results.

---

## Summary

The replication was **successful**. All three evaluation criteria pass:

| Criterion | Status | Key Evidence |
|-----------|--------|--------------|
| RP1. Implementation Reconstructability | PASS | Complete instructions in plan.md and CodeWalkthrough.md |
| RP2. Environment Reproducibility | PASS | environment.yml + HuggingFace checkpoints + included data |
| RP3. Determinism and Stability | PASS | Inference-only experiment with fixed seeds |

The documentation in this repository is sufficient for an independent researcher to replicate the core findings about emergent world representations in Othello-GPT.
