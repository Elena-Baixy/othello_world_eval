# Documentation Evaluation Summary

## Replicator-Documentation Evaluation for Othello-World

**Date:** 2025-12-25

---

## Results Comparison

The replicated documentation reports linear probe accuracy across layers 0-7, with the best accuracy at Layer 6 achieving 99.64% (0.36% error). However, the original documentation (plan.md) explicitly states that linear probes "never dip below 20% error across all layers" with the best linear probe achieving only 20.4% error (layer 3) for the synthetic model. The original work emphasizes that nonlinear probes are necessary, achieving 1.7% error at layer 7.

This represents a significant discrepancy: the replication claims linear probes achieve dramatically better results (0.36% error) than the original claims (>20% error always). The intervention results showing +9.3/-8.8 log probability changes are qualitatively consistent with the original's finding that interventions successfully modify predictions, though the metrics differ.

---

## Conclusions Comparison

The replicated documentation correctly concludes that:
1. The model develops an emergent internal representation of the board state
2. The representation has a causal role in model predictions

However, the replication's conclusion that "the linear probe achieves excellent accuracy (comparable to reported nonlinear probe results), suggesting the 'mine vs theirs' encoding captures the essential representation" **directly contradicts** the original's core finding that "nonlinear probes are necessary to decode board state from internal activations, while linear probes fail."

This is a fundamental inconsistency: the original work's key scientific claim is that linear probes fail and nonlinear probes are required, while the replication suggests linear probes work well.

---

## External or Hallucinated Information

The replication documentation does not introduce external or hallucinated information. All referenced tools (TransformerLens, HuggingFace models) are legitimately connected to the Othello-GPT work as mentioned in the original CodeWalkthrough.md. The replication clearly distinguishes between its own experimental results and the original findings, and properly cites methodological sources. The specific neuron analysis (L5N1393) and "mine vs theirs" encoding are standard approaches in this line of research.

---

## Evaluation Summary Table

| Criterion | Status | Notes |
|-----------|--------|-------|
| **DE1: Result Fidelity** | **FAIL** | Linear probe error rates differ significantly (0.36% vs >20%). Intervention results use different metrics. |
| **DE2: Conclusion Consistency** | **FAIL** | Replication concludes linear probes work well; original concludes linear probes fail and nonlinear are necessary. |
| **DE3: No External/Hallucinated Information** | **PASS** | No external or fabricated information introduced. Sources properly cited. |

---

## Final Verdict

**REVISION REQUIRED**

The replicated documentation fails on two critical criteria:
1. **Result Fidelity (DE1):** The reported linear probe accuracy (0.36% error) contradicts the original's claim that linear probes never achieve below 20% error.
2. **Conclusion Consistency (DE2):** The replication's conclusion that linear probes work well directly contradicts the original's key finding that nonlinear probes are necessary.

These discrepancies may stem from:
- Different probe implementations or encodings ("mine vs theirs" vs original encoding)
- Different evaluation methodologies
- Using a different model variant (TransformerLens version vs original)

Revision should clarify these methodological differences and reconcile the conflicting findings about linear vs nonlinear probe performance.
