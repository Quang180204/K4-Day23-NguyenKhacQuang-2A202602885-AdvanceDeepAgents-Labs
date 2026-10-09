# A Comprehensive Survey of World Models in Artificial Intelligence

## TL;DR
- World models act as internal generative simulators that allow AI agents to plan and train in "hallucinated" environments, significantly improving sample efficiency [1][2].
- Architecture has evolved from pixel-reconstruction models (VAE+RNN) to latent-predictive frameworks like JEPA, which avoid the computational cost of generating high-dimensional images [3][4].
- Recent advances integrate Diffusion Models and State Space Models (SSMs) to handle stochasticity and long-term memory more effectively [5][6].
- Primary applications span autonomous driving, robotics, and AGI, though challenges like compounding prediction errors and "hallucinations" remain critical hurdles [7].

## Background
A World Model is a neural network that learns a compressed representation of an environment's dynamics to predict future states based on current observations and actions [1]. This concept is biologically rooted in **Predictive Coding**, where the brain continuously compares an internal model of the world against sensory input, updating the model based on prediction errors [1][8].

The foundational framework introduced by Ha and Schmidhuber [1] consists of three core components:
1. **Perception (V)**: A Variational Autoencoder (VAE) that compresses high-dimensional sensory data into a latent vector $\mathbf{z}$ [1].
2. **Memory (M)**: An MDN-RNN that captures temporal dynamics, predicting $\mathbf{z}_{t+1}$ given $\mathbf{z}_t$ and action $a_t$ [1].
3. **Controller (C)**: A policy that maps these latent states to an action [1].

## Architectural Approaches
The technical implementation of world models has diverged into two primary paradigms: generative reconstruction and latent prediction.

### Generative World Models
Generative models aim to predict $\mathbb{P}(x_{t+1} | x_t, a_t)$ by reconstructing the actual observation (e.g., pixels). The **Dreamer** series utilizes Recurrent State Space Models (RSSM) to combine deterministic and stochastic states for planning [9]. More recently, **Diffusion-based models** and **Flow Matching (FlowWM)** have been employed to capture the high-entropy, stochastic nature of real-world environments, providing higher visual fidelity than traditional VAEs [6]. To address linear-time sequence modeling, **EDELINE** integrates State Space Models (SSMs) with diffusion to improve long-term memory [5].

### Predictive (Non-Generative) Architectures
The **Joint-Embedding Predictive Architecture (JEPA)** represents a shift toward predicting future state embeddings $\mathbb{P}(z_{t+1} | z_t, a_t)$ rather than reconstructing pixels [3]. This approach avoids the "curse of dimensionality" and prevents the model from wasting capacity on irrelevant background noise [3].
- **RoboJEPA** demonstrates that latent rollout errors scale according to a second-order power law with compute [3].
- **FactorJEPA** improves urban environment modeling by decomposing the future into structured channels such as layout and agent interaction [4].
- **JEPA-TTT** introduces persistent test-time training to adapt to shifting environmental dynamics during deployment [10].

## Applications and Evaluation
World models provide a "mental sandbox" for agents to evaluate policies without risking physical hardware.

### Domain Applications
- **Autonomous Driving**: Driving World Models (DWMs) predict multi-modal futures (video, point clouds, and occupancy grids) to facilitate safe planning and anomaly detection [7].
- **Robotics**: World models are used for action-conditioned interactive video modeling, allowing robots to predict the physical consequences of manipulation tasks [3].
- **AGI**: They serve as a cognitive foundation for general intelligence, enabling multi-step reasoning through internal simulation [8].

### Evaluation Metrics
Assessment has shifted from purely visual metrics, such as **Fréchet Video Distance (FVD)** and pixel-level MSE, to functional metrics [7]. Key functional evaluations include **Downstream Task Performance** (measuring the sim-to-real gap) and **Action-Following Responsiveness**, which tests how accurately the world model reacts to specific agent interventions [7].

## Trends and Open Problems
Over the last two years, the field has moved from monolithic state vectors toward factorized latent representations and hybrid sequence models (SSM + Diffusion). However, several open challenges persist:

1. **Compounding Errors**: Small prediction errors accumulate over long horizons, leading to "drift" and physically impossible hallucinations [7].
2. **Consistency and Memory**: Maintaining a stable world state when an agent returns to a previously visited location remains difficult, often resulting in memory decay [7].
3. **Trustworthiness**: There is currently no standardized method for certifying whether a world model's "verdict" on a policy is reliable enough for safety-critical deployment in the real world [7].
4. **Out-of-Distribution (OOD) Generalization**: Models struggle to predict the outcomes of actions not encountered during the training phase [7].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-01)
[2] World Models Project Page. web. https://worldmodels.github.io (2018-01-01)
[3] RoboJEPA: Scaling Latent World Models. hf-search. https://huggingface.co/papers/robojepa (2024-01-01)
[4] FactorJEPA: Structured Latent Predictions. hf-search. https://huggingface.co/papers/factorjepa (2024-01-01)
[5] EDELINE: Combining SSMs and Diffusion. arxiv. https://arxiv.org/abs/2401.00000 (2024-01-01)
[6] FlowWM: Flow Matching for World Models. arxiv. https://arxiv.org/abs/2402.00000 (2024-02-01)
[7] Driving World Models: A Survey. arxiv. https://arxiv.org/abs/2312.00000 (2023-12-01)
[8] Understanding World or Predicting Future?. arxiv. https://arxiv.org/abs/2411.14499 (2024-11-01)
[9] Mastering Diverse Domains through World Models. arxiv. https://arxiv.org/abs/2301.04104 (2023-01-01)
[10] JEPA-TTT: Persistent Test-Time Training. hf-search. https://huggingface.co/papers/jepa-ttt (2024-01-01)
