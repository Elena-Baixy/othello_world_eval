# Plan
## Objective
Investigate whether language models trained on sequence prediction tasks develop internal representations of the underlying process generating sequences, using a GPT variant trained on predicting legal moves in Othello as a testbed.

## Hypothesis
1. A GPT model trained on Othello game transcripts develops an emergent nonlinear internal representation of the board state despite having no a priori knowledge of game rules.
2. The emergent board state representation has a causal role in the model's predictions and can be used to control network output.
3. Nonlinear probes are necessary to decode board state from internal activations, while linear probes fail.

## Methodology
1. Train an 8-layer GPT model (Othello-GPT) with 8-head attention and 512-dimensional hidden space on game transcripts using autoregressive cross-entropy loss, with no a priori knowledge of board structure or rules.
2. Use two datasets: championship (140,526 expert games) and synthetic (20 million randomly generated legal games) to train separate models.
3. Train nonlinear probes (2-layer MLPs) on internal activations to predict board state (black/white/empty for each tile), and compare with linear probe baselines.
4. Perform interventional experiments using gradient descent to modify activations at multiple layers sequentially (starting from layer Ls) to change predicted board states and measure effects on move predictions.
5. Create latent saliency maps by intervening on each tile's representation and measuring prediction probability changes to visualize which board tiles causally contribute to predictions.

## Experiments
### Legal move prediction accuracy
- What varied: Training dataset (synthetic vs championship vs untrained baseline)
- Metric: Error rate (percentage of illegal top-1 predictions)
- Main result: Synthetic-trained model achieves 0.01% error, championship-trained 5.17%, untrained baseline 93.29%. A skewed dataset (25% game tree removed) yields 0.02% error, ruling out pure memorization.

### Linear probe accuracy across layers
- What varied: Model layer (1-8) and training condition (randomized vs championship vs synthetic)
- Metric: Error rate (%) in predicting tile state (black/white/empty)
- Main result: Linear probes never dip below 20% error across all layers. Synthetic best: 20.4% (layer 3), Championship: 23.6% (layers 4-5), only marginally better than randomized baseline (26.7-28.9%).

### Nonlinear probe accuracy across layers
- What varied: Model layer (1-8) and training condition (randomized vs championship vs synthetic)
- Metric: Error rate (%) in predicting tile state
- Main result: Nonlinear probes dramatically outperform linear. Synthetic achieves 1.7% error (layer 7), championship 9.4% (layer 4). Randomized baseline shows minimal improvement (25.4-26.4%), confirming nontrivial representation.

### Interventional validation on natural benchmark
- What varied: Number of intervened layers (starting layer Ls from 1 to 8)
- Metric: Average error (false positives + false negatives in top-N predictions)
- Main result: Intervening 5 layers (Ls=4) achieves 0.12 error vs null intervention baseline of 2.68, demonstrating causal role of representation.

### Interventional validation on unnatural benchmark
- What varied: Number of intervened layers on board states unreachable by legal play
- Metric: Average error in top-N predictions
- Main result: Achieves 0.06 error (Ls=4) vs 2.59 baseline, showing intervention works even on out-of-distribution positions far from training data.

### Latent saliency map comparison
- What varied: Training dataset (synthetic vs championship)
- Metric: Qualitative saliency patterns for top-1 predictions
- Main result: Synthetic model shows high saliency precisely on tiles required for move legality. Championship model shows complex global patterns reflecting strategic considerations beyond simple legality.