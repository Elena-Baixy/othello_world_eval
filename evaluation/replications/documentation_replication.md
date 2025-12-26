# Othello-GPT Circuit Analysis Replication

## Goal

This replication aims to verify the key findings from the "Emergent World Representations" paper (Li et al., ICLR 2023), which investigates whether a GPT model trained on Othello move sequences develops internal representations of the board state.

The specific objectives are:
1. Verify that linear probes can decode board state from internal activations
2. Demonstrate that probe directions have a causal role in model predictions through interventions
3. Analyze circuit structure by examining layer contributions
4. Validate neuron-level interpretability

## Data

### Model
- **Architecture**: 8-layer GPT (Othello-GPT)
  - 8 attention heads per layer
  - 512-dimensional hidden space
  - 2048 neurons per MLP layer
  - Vocabulary size: 61 (60 playable moves + pass token)
  - Context length: 59 moves
- **Training**: Synthetic dataset of 20 million randomly generated legal Othello games
- **Source**: Pre-trained weights from HuggingFace (`NeelNanda/Othello-GPT-Transformer-Lens`)

### Game Data
- 100,000 complete Othello games (60 moves each)
- Two formats:
  - `board_seqs_int`: Model vocabulary format (1-60, skipping center squares)
  - `board_seqs_string`: Board position format (0-63)

### Probe
- Linear probe pre-trained on model activations
- Shape: [3, 512, 8, 8, 3] - modes × d_model × rows × cols × options
- Modes: black-to-play, white-to-play, all
- Options: empty, white, black

## Method

### 1. Model Verification
- Loaded Othello-GPT using TransformerLens library
- Verified model outputs match expected predictions on sample inputs

### 2. Probe Accuracy Analysis
- Applied linear probe to residual stream at each layer (0-7)
- Computed accuracy in predicting board state (empty/mine/theirs)
- Used "mine vs theirs" encoding to handle alternating player turns

### 3. Intervention Experiments
- Created interpretable probe directions:
  - `blank_probe`: empty vs filled
  - `my_probe`: my color vs their color
- Intervened on model activations by adding/subtracting probe directions
- Measured effect on move predictions (log probabilities)

### 4. Activation Patching
- Set up clean/corrupted input pairs differing only in final move
- Patched activations from clean run into corrupted run
- Identified which layers are causally important for predictions

### 5. Neuron Analysis
- Analyzed neuron L5N1393 (identified in original work)
- Projected input/output weights onto probe directions
- Verified hypothesis about neuron's feature detection

## Results

### Linear Probe Accuracy

| Layer | Accuracy | Error Rate |
|-------|----------|------------|
| 0     | 82.17%   | 17.83%     |
| 1     | 89.48%   | 10.52%     |
| 2     | 94.05%   | 5.95%      |
| 3     | 96.80%   | 3.20%      |
| 4     | 98.29%   | 1.71%      |
| 5     | 98.75%   | 1.25%      |
| 6     | 99.64%   | 0.36%      |
| 7     | 88.95%   | 11.05%     |

**Best accuracy at Layer 6: 99.64%**

This matches the original finding that the model develops an accurate internal representation of the board state, with the representation most accurate at layer 6.

### Intervention Results

Flipping cell F4's representation at Layer 4:
- **D2 (newly legal)**: log prob changed from -11.5 to -2.2 (+9.3)
- **G4 (newly illegal)**: log prob changed from -2.2 to -11.0 (-8.8)

This confirms that the probe directions have a causal role in the model's predictions.

### Activation Patching Results

Most important layers for F0 prediction:
| Layer | Attention | MLP |
|-------|-----------|-----|
| 0     | -0.20%    | 85.84% |
| 5     | 1.03%     | 77.65% |
| 6     | 3.36%     | 64.88% |
| 7     | 26.54%    | -0.23% |

MLP layers 0, 5, and 6 are most important for move legality predictions.

### Neuron Analysis (L5N1393)

Hypothesis: Detects C0=blank & D1=theirs & E2=mine

| Configuration | Mean Activation | N samples |
|---------------|-----------------|-----------|
| With config   | 0.9230          | 114       |
| Without       | -0.0292         | 2836      |

The neuron strongly activates when the hypothesized configuration is present, confirming the interpretability hypothesis.

## Analysis

### Key Findings Replicated

1. **Board State Representation**: The model develops an accurate internal representation of the Othello board state, with best probe accuracy at layer 6 (99.64% accuracy).

2. **Causal Role of Representation**: Interventions on the probe directions successfully change model predictions in the expected ways, demonstrating that these representations are causally used by the model.

3. **Layer Contributions**: Different layers contribute differently to the board state computation:
   - Early layers (0-3): Build up the representation progressively
   - Layer 6: Peak accuracy
   - Layer 7: Accuracy drops, suggesting the representation is being transformed for output

4. **MLP Importance**: MLP layers are more important than attention layers for computing move legality, consistent with the hypothesis that MLPs store and apply knowledge about game rules.

5. **Interpretable Neurons**: Individual neurons can be understood in terms of their input/output relationships with probe directions, enabling fine-grained circuit analysis.

### Comparison with Original Results

| Metric | Original (Plan) | Replication |
|--------|-----------------|-------------|
| Best probe accuracy | ~99% (nonlinear) | 99.64% (linear, layer 6) |
| Intervention effect | Significant | +9.3/-8.8 log prob change |
| Layer importance | MLP layers dominate | MLP layers dominate |

The replication results are consistent with the original findings, with the notable observation that the linear probe achieves excellent accuracy (comparable to reported nonlinear probe results), suggesting the "mine vs theirs" encoding captures the essential representation.

### Limitations

1. **Sample Size**: Analysis limited to 50 focus games due to memory constraints
2. **Probe Training**: Used pre-trained probe rather than training from scratch
3. **Single Model**: Only tested synthetic model, not championship model
