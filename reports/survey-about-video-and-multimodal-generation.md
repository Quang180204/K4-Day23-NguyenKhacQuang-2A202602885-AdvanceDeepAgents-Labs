# A Survey of Video and Multimodal Generation: Architectures, SOTA Models, and Implementation

## TL;DR
- The field has transitioned from early hybrid VAE-GAN frameworks [1] to stable, high-fidelity Latent Video Diffusion Models (LVDMs) [2] and recently toward hybrid Autoregressive-Diffusion (AR-Diffusion) architectures like ART•V [3].
- Current State-of-the-Art (SOTA) models have shifted from U-Net backbones to Diffusion Transformers (DiT), with models like Lumina-Video [4] and Latte [5] scaling efficiency and quality.
- Temporal consistency is now addressed via single-pass generation (e.g., Lumiere's Space-Time U-Net [6]) or training-free attention control mechanisms (e.g., DiTCtrl [7]).
- The open-source ecosystem has matured with large-scale foundation models such as Wan-Video [8], HunyuanVideo [9], and Open-Sora [10], often deployed via Hugging Face Diffusers and ComfyUI.

## Background
Video and multimodal generation involves the synthesis of coherent temporal sequences of images (videos) conditioned on various modalities (text, image, or audio). Early foundational work relied on Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) to model the complex distribution of video frames [1]. However, these early approaches often suffered from temporal instability and "mode collapse," leading to the adoption of Diffusion Models. Diffusion models, particularly Latent Video Diffusion (LVDM), reduce the computational burden by operating in a low-dimensional 3D latent space, ensuring higher stability and fidelity for longer sequences [2][11].

## Theoretical Evolution and Foundations
The architectural journey of video generation is marked by a transition from spatial-only models to spatio-temporal frameworks [11]. 

### From GANs to Diffusion
Early hybrid VAE-GANs [1] provided a structured latent space and sharpened outputs, but lacked the stability to generate long-term coherent videos. The introduction of Denoising Diffusion Probabilistic Models (DDPMs) shifted the paradigm toward iterative denoising, which provides superior distribution coverage. Latent Video Diffusion Models [2] further optimized this by performing the diffusion process in a compressed latent space, which became the foundation for most modern SOTA systems.

### The Rise of Hybrid AR-Diffusion
To combat the "drifting" effect in pure autoregressive (AR) models, recent frameworks like ART•V [3] combine the sequential nature of AR models with the per-frame quality of diffusion. This hybrid approach ensures that each frame is conditioned on previous ones while maintaining the high fidelity typical of diffusion. Furthermore, recent work like NOVA [12] demonstrates that efficient AR video generation is possible without requiring Vector Quantization (VQ), challenging the traditional dependence on VQ-VAEs.

## State-of-the-Art Architectures and Techniques
Modern SOTA models have largely abandoned the traditional U-Net in favor of Transformer-based backbones to improve scalability and resolution.

### Diffusion Transformers (DiT)
Diffusion Transformers (DiT) are now central to high-fidelity synthesis. Models such as Lumina-Video [4] use multi-scale "Next-DiT" architectures combined with motion scoring to enhance efficiency. Similarly, Latte [5] models video distributions directly using Transformer blocks in the latent space, allowing for better scaling. xGen-VideoSyn-1 [13] further improves fidelity by utilizing a spatial-temporal divide-and-merge strategy paired with a Video VAE.

### Temporal Consistency and Motion Control
Temporal coherence remains a primary challenge. Lumiere [6] addresses this by using a Space-Time U-Net that generates the entire video in a single pass, eliminating the flickering common in frame-by-frame methods. For those using MM-DiT architectures, DiTCtrl [7] provides a training-free method to manipulate attention mechanisms, ensuring smooth transitions in multi-prompt videos. Additionally, AnimateLCM [14] utilizes decoupled consistency learning to accelerate the generation process while maintaining personalized adapters.

## Practical Implementations and Open Source Ecosystem
The transition from research papers to production tools has been accelerated by the open-source community.

### Foundation Model Repositories
Several large-scale models have been released with open weights. Wan-Video (Wan2.2) [8] implements Mixture-of-Experts (MoE) and supports high-compression VAEs for consumer GPU deployment. HunyuanVideo [9] utilizes Diffusion Transformers and Unified Sequence Parallelism (USP) to scale to over 13 billion parameters. Open-Sora [10] provides a democratized pipeline for the full T2V lifecycle, from data preprocessing to inference. LTX-Video [15] supports native 4K resolution and synchronized audio, offering an optimized 8-bit version for performance.

### Frameworks and Tooling
The deployment of these models is largely standardized through the Hugging Face Diffusers library, which provides unified pipelines for models like CogVideo [16] and HunyuanVideo [9]. For professional iterative workflows, ComfyUI has become the leading node-based GUI, integrating plugins for AnimateDiff [17] and LTX-Video [15]. AnimateDiff [17] specifically stands out as a plug-and-play motion module that transforms text-to-image models into animation generators without requiring full retraining. VACE [18] further unifies various editing tasks (R2V, V2V, masked V2V) on top of foundations like LTX-Video.

## Trends and Open Problems
Recent developments in the last two years (2023-2025) show a clear trend toward scaling laws for video (DiT scaling) and the integration of multimodal inputs (Speech-to-Video in Wan-Video [8]). However, several open challenges remain:
1. **Computational Efficiency**: While LVDMs [2] and quantized models (e.g., LTX-VideoQ8 [15]) exist, real-time interactive generation remains a computationally expensive task.
2. **Temporal Long-term Memory**: Despite the use of Space-Time U-Nets [6] and AR-Diffusion [3], generating videos with consistent characters and objects over several minutes remains an open problem.
3. **Consistency vs. Fidelity**: The trade-off between high per-frame fidelity and temporal smoothness (flickering) is still being optimized through techniques like attention control [7] and decoupled consistency learning [14].

## References
[1] Video Generation From Text. arxiv. https://arxiv.org/abs/1710.00421 (2017-10-01)
[2] Latent Video Diffusion Models for High-Fidelity Long Video Generation. arxiv. https://arxiv.org/abs/2211.13221 (2022-11-23)
[3] ART•V: Auto-Regressive Text-to-Video Generation with Diffusion Models. arxiv. https://arxiv.org/abs/2311.18834 (2023-11-30)
[4] Lumina-Video: Multi-scale Next-DiT for Efficient Video Generation. hf-search. https://huggingface.co/papers/lumina-video (2025-01-01)
[5] Latte: Latent Diffusion Transformer for Video Generation. hf-search. https://huggingface.co/papers/latte (2024-01-01)
[6] Lumiere: A Space-Time Diffusion Model for Video Generation. hf-daily. https://huggingface.co/papers/lumiere (2024-01-01)
[7] DiTCtrl: Training-Free Attention Control for Video Generation. hf-search. https://huggingface.co/papers/ditctrl (2024-01-01)
[8] Wan-Video (Wan2.2) GitHub Repository. web. https://github.com/wan-video/wan2.2 (2025-01-01)
[9] HunyuanVideo GitHub Repository. web. https://github.com/Tencent-Hunyuan/HunyuanVideo (2024-01-01)
[10] Open-Sora GitHub Repository. web. https://github.com/hpcaitech/Open-Sora (2024-01-01)
[11] Video Diffusion Models: A Survey. arxiv. https://arxiv.org/abs/2405.03150 (2024-05-06)
[12] Autoregressive Video Generation without Vector Quantization (NOVA). arxiv. https://arxiv.org/abs/2412.14169 (2024-12-18)
[13] xGen-VideoSyn-1: High-Fidelity Synthesis via Spatial-Temporal Divide-and-Merge. hf-search. https://huggingface.co/papers/xgen-videosyn (2024-01-01)
[14] AnimateLCM: Decoupled Consistency Learning for Fast Video Generation. hf-daily. https://huggingface.co/papers/animatelcm (2024-01-01)
[15] LTX-Video GitHub Repository. web. https://github.com/Lightricks/LTX-Video (2024-01-01)
[16] CogVideo GitHub Repository. web. https://github.com/zai-org/CogVideo (2023-01-01)
[17] AnimateDiff GitHub Repository. web. https://github.com/guoyww/AnimateDiff (2023-01-01)
[18] VACE GitHub Repository. web. https://github.com/ali-vilab/VACE (2024-01-01)
