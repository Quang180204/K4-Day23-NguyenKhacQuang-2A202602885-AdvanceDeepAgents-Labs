# Reinforcement Learning for LLM Reasoning: A Comprehensive Survey

## TL;DR
- Reinforcement Learning (RL) has evolved from general alignment (PPO, DPO) to specialized reasoning optimization, enabling emergent "thinking" behaviors like backtracking and self-verification [1].
- Process-supervised Reward Models (PRMs) provide step-by-step guidance, significantly improving mathematical reasoning, though their effectiveness can vary by task complexity [2].
- Recent algorithmic shifts include Sequence-Level PPO (SPPO) for long-horizon stability [3] and Iterative DPO for computational efficiency in reasoning tasks [4].
- Open-source frameworks like TRL, DeepSpeed-Chat, and OpenRLHF have democratized the implementation of these complex RL pipelines [5][6][7].

## Background
Reinforcement Learning (RL) in the context of Large Language Models (LLMs) typically aims to align model outputs with specific objectives, such as human preferences or correctness in logical tasks. Historically, Reinforcement Learning from Human Feedback (RLHF) utilized Proximal Policy Optimization (PPO) to optimize a policy against a reward model [8]. For reasoning, the goal is to optimize the "Chain-of-Thought" (CoT), where the model generates intermediate steps to arrive at a final answer. The shift from Outcome-based Reward Models (ORMs), which only reward the final answer, to Process-based Reward Models (PRMs), which reward each individual step, has been central to improving reasoning accuracy [2].

## Foundational RL Algorithms for Reasoning
The landscape of RL for LLMs is dominated by two primary paradigms: PPO and DPO.

### Proximal Policy Optimization (PPO)
PPO remains a powerful baseline for complex logic and code generation [9]. However, it is prone to instability, particularly in long-horizon reasoning. To address this, Sequence-Level PPO (SPPO) treats reasoning trajectories as contextual bandit problems, decoupling value functions to stabilize training over long CoT sequences [3]. Other variants like PPO-max focus on improving the stability of the policy model relative to the supervised fine-tuning (SFT) base [8].

### Direct Preference Optimization (DPO)
DPO offers a computationally efficient, RL-free alternative by optimizing the policy directly from preference pairs [4]. While efficient, it has theoretical gaps in alignment provability, which Constrained Preference Optimization (CPO) aims to bridge [10]. Various specialized versions of DPO have emerged to enhance reasoning, such as Curry-DPO, which employs curriculum learning [11], and Uni-DPO, which adaptively weights preference pairs based on data quality [12].

## Advanced Reasoning Strategies
Beyond basic alignment, recent work has focused on evoking emergent reasoning capabilities.

### Emergent "Thinking" via Self-Play
Reinforcement Learning via Self-Play (RLSP) has demonstrated that decoupling exploration from correctness signals can lead to the emergence of "thinking" behaviors. These include the ability to backtrack when a reasoning path is incorrect and to perform self-verification of intermediate steps [1]. This marks a shift from merely imitating human CoT to discovering optimal reasoning paths through exploration.

### Process vs. Outcome Supervision
The debate between Process-supervised Reward Models (PRMs) and Outcome-based Reward Models (ORMs) is critical. PRMs reward the correctness of each step in a reasoning chain, which theoretically reduces the "reward hacking" common in ORMs [2]. However, empirical evidence suggests that while PRMs are highly effective for simple math, their performance gains may diminish or even reverse on extremely complex tasks [2].

### Reward Model Optimization
Recent surveys emphasize the critical role of reward model selection and the prevention of reward hacking [13]. Techniques such as Question Augmentation (QuestA) increase sample efficiency by introducing partial solutions during RL training, specifically aiding in multi-step mathematical reasoning [14].

## Implementation Ecosystem
The practical application of RL for reasoning is supported by several key open-source frameworks.

Hugging Face's **TRL (Transformer Reinforcement Learning)** provides a highly integrated pipeline for both PPO and DPO, making it accessible for researchers [5]. For those requiring massive scalability, **DeepSpeed-Chat** utilizes ZeRO optimization to train large models on limited hardware [6], while **OpenRLHF** leverages Ray and DeepSpeed for high-throughput distributed training of reasoning models [7]. These frameworks allow for the rapid prototyping of new RL objectives and reward functions tailored to reasoning tasks.

## Trends and Open Problems
The trend in the last two years has shifted toward "reasoning-native" models that integrate RL throughout the training process rather than as a post-hoc alignment step [15]. Open challenges remain in the stability of long-horizon reasoning and the cost of generating high-quality process-level labels for PRMs. There is also an ongoing dispute regarding the minimalist versus complex approach to RL strategies; some research suggests that a minimalist combination of techniques may outperform highly complex pipelines [16].

## References
[1] On the Emergence of Thinking in LLMs I: Searching for the Right Intuition. hf-search. https://huggingface.co/papers/2502.06773 (2025-02-10)
[2] Let's Reinforce Step by Step. hf-search. https://huggingface.co/papers/2311.05821 (2023-11-10)
[3] Sequence-Level PPO (SPPO) for Long-Horizon Reasoning. arxiv. https://arxiv.org/abs/2604.08865 (2026-04-01)
[4] Iterative DPO for LLM Reasoning. arxiv. https://arxiv.org/abs/2503.12854 (2025-03-01)
[5] TRL (Transformer Reinforcement Learning). web. https://github.com/huggingface/trl (2024-01-01)
[6] DeepSpeed-Chat. web. https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat (2024-01-01)
[7] OpenRLHF. web. https://github.com/OpenRLHF/OpenRLHF (2024-01-01)
[8] PPO-max: Improving Stability of Policy Model Training. arxiv. https://arxiv.org/abs/2307.04964 (2023-07-01)
[9] Policy Filtration PPO (PF-PPO) for Code Generation. arxiv. https://arxiv.org/abs/2409.06957 (2024-09-01)
[10] Constrained Preference Optimization (CPO). arxiv. https://arxiv.org/abs/2605.20834 (2026-05-01)
[11] Curry-DPO: Curriculum Learning for Preference Optimization. arxiv. https://arxiv.org/abs/2403.07230 (2024-03-01)
[12] Uni-DPO: Adaptive Weighting for Preference Pairs. arxiv. https://arxiv.org/abs/2506.10054 (2025-06-01)
[13] Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey. hf-search. https://huggingface.co/papers/2510.01925 (2025-10-02)
[14] QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation. hf-search. https://huggingface.co/papers/2507.13266 (2025-07-17)
[15] Reasoning Language Models: A Blueprint. hf-search. https://huggingface.co/papers/2501.11223 (2025-01-20)
[16] Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning. hf-search. https://huggingface.co/papers/2508.08221 (2025-08-11)
