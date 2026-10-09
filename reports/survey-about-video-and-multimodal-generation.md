# Video and Multimodal Generation: A Survey of Foundations, Model Families, Evaluation and Open Problems

## TL;DR

- Video generation has largely converged on latent diffusion transformers: a video VAE compresses clips, and a DiT-style backbone denoises the latents with flow matching, as in Wan [1] building on DiT [2] and latent diffusion [3][4].
- Two architectural families compete for long and interactive video: diffusion transformers, which add camera and trajectory control [5][6][7], and autoregressive or hybrid models that use memory and KV-cache tricks for long rollouts [8][9][10][11].
- Unified multimodal models split into discrete next-token designs (Emu3 [12], Emu3.5 [13]) and diffusion-transformer designs, with joint audio-video generation emerging as a distinct cluster [14][15][16][17].
- Evaluation is moving from superficial metrics toward intrinsic faithfulness (physics, commonsense) and human-aligned or MLLM-based reward models, but reward hacking and annotation noise remain open issues [18][19][20].

## Background

Diffusion models generate data by iteratively denoising noise. Latent diffusion moves this process into the latent space of a pretrained autoencoder, with cross-attention for conditioning, which reduces cost relative to pixel-space diffusion while keeping high quality [3]. The DiT paper replaced the U-Net backbone with a transformer operating on latent patches and showed that forward-pass compute (Gflops) predicts sample quality: the largest model, DiT-XL/2, reached FID 2.27 on class-conditional ImageNet 256x256 [2]. Align Your Latents adapts latent diffusion to high-resolution video synthesis by introducing a temporal dimension, enabling text-to-video with fine-tuning [4].

Video is harder than images because of the added temporal dimension, and models use full spatio-temporal attention to keep content coherent across frames [1]. This is why the field focuses on video VAEs that compress space and time jointly [1][21][22][23], and on attention schemes over spatio-temporal tokens [1]. Multimodal generation adds further requirements: text, image, audio and video must be produced and understood in one model [24][25].

Recent surveys give the scope of the area. A 2025 survey traces text-to-video from early GANs and VAEs to hybrid DiT architectures [26]. A comparative review covers more than 170 papers from January 2023 to April 2025 and frames two dominant families, autoregressive sequence models and diffusion models, with a third hybrid branch [27]. A survey of unified multimodal understanding and generation organises models into diffusion-based, autoregressive and hybrid approaches [24].

## Foundations: latent diffusion transformers and video VAEs

The modern recipe combines three pieces. First, a video VAE: Wan uses a 3D causal VAE obtained by inflating a 2D image VAE, trained with L1, KL and LPIPS losses and then fine-tuned with a GAN loss [1]. Newer work targets the VAE itself. LeanVAE uses wavelet transforms and compressed sensing to reduce cost [21], OD-VAE compresses along both spatial and temporal axes [22], and a 2025 analysis argues that spectral properties of video VAE latents determine how easily diffusion models learn them ("diffusability") [23]. LTX-Video is presented as a real-time latent diffusion model that integrates a Video-VAE with a denoising transformer [28].

Second, a transformer denoiser trained with flow matching. Wan is built on the diffusion transformer paradigm, and its report pairs the DiT with flow matching, a rectified-flow style training objective [1]. Wan models come in 1.3B and 14B sizes, the 1.3B model needing 8.19 GB of VRAM, and it uses full spatio-temporal attention with cross-attention for text [1]. The same DiT-plus-flow-matching recipe has been carried to other generative domains, for example billion-parameter text-to-motion generation over more than 3,000 hours of motion data [29].

Third, large captioned video data. OpenVid-1M provides a large-scale high-quality text-to-video dataset together with MVDiT, a multimodal video diffusion transformer [30]. CogVideoX pairs a 3D VAE with an expert transformer for coherent text-to-video generation [31].

## Text-to-video and image-to-video model families

Diffusion transformers dominate open text-to-video systems. Beyond CogVideoX [31] and Wan [1], work on controllability builds on this backbone. VD3D adds 3D camera control to large video diffusion transformers using spatiotemporal embeddings [5], and AC3D analyses camera conditioning in video DiTs, optimising pose-conditioning schedules and selectively injecting camera conditioning [6]. Training-free alternatives exist: DiTraj enhances trajectory control in DiT-based text-to-video by separating foreground and background prompts and modifying position embeddings [7]. 

Autoregressive and hybrid models target long videos. Context Forcing uses a long-context teacher and a Slow-Fast Memory to address student-teacher mismatch in long rollouts, reporting context beyond 20 seconds [8]. VideoSSM combines a state-space memory with autoregressive diffusion to coordinate short- and long-term context [9]. Efficiency work targets the same setting: Quant VideoGen compresses the KV cache to 2 bits [10], and TokenTrim prunes unstable conditioning tokens at inference to reduce temporal drift from error accumulation [11]. A comparative review notes that the autoregressive and diffusion branches are increasingly hybridised [27].

Commercial-style systems (Sora, Veo, Kling, Runway, Luma) are named in recent surveys [26], but the technical details of these systems are not publicly documented in the sources gathered for this survey, so the comparison here is restricted to open research models.

## Unified and audio-video multimodal generation

Unified models pursue one architecture that understands and generates several modalities. A common three-part framework uses modality-specific encoders, a fusion backbone and modality-specific decoders [24]. Two design choices dominate. Discrete autoregressive designs tokenise everything and use next-token prediction: Emu3 is trained exclusively with next-token prediction and handles text, image and video with discrete tokens [12]; Emu3.5 is framed as a native multimodal world model enhanced with reinforcement learning and discrete diffusion adaptation [13]; NextFlow is a decoder-only autoregressive transformer over interleaved text-image tokens with next-token and next-scale prediction [32]. Continuous designs instead pair a diffusion or flow head with a semantic or VAE representation; a survey of unified MLLMs contrasts discrete VQGAN tokens with continuous CLIP-style features and argues for semantic-level representations over pixel-level VAE tokenizers [25].

Audio-video joint generation is a fast-growing cluster. JavisDiT is a joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronisation [14]; AV-DiT reuses a shared pretrained DiT backbone with lightweight modality-specific adaptations [15]; 3MDiT models video, audio and text as jointly evolving streams [17]; Talker-T2AV uses autoregressive diffusion to separate cross-modal reasoning from modality-specific refinement for talking-head synthesis [16]. The survey [25] notes that audio adds stronger temporal structure and synchronisation constraints than text-vision, which makes cross-modal alignment a central difficulty.

Video is also becoming the substrate for world models. An interactive video world-modelling survey defines these models as action-conditioned, recurrently generating future frames from interaction history and new controls, and lists three challenges: action-conditioned controllability, long-horizon memory, and responsiveness for real-time interaction [33]. A 2026 survey of World Action Models argues that such models are "not simply video generators with action heads", and that some recent work repurposes large video generators while a parallel line uses language or vision-language backbones [34].

## Evaluation, rewards and safety

Early evaluation relied on feature-distribution and embedding metrics (IS, FID, FVD, CLIP alignment). Video-Bench argues these misalign with human preference and proposes an MLLM-based evaluation with chain-of-query and few-shot scoring; it validates its scores against human judgment and compares them with EvalCrafter, VBench and ComBench [19]. VBench-2.0 extends VBench, which it describes as measuring "superficial faithfulness", toward "intrinsic faithfulness", with five dimensions (Human Fidelity, Controllability, Creativity, Physics, Commonsense) scored by VLMs/LLMs and specialist anomaly detectors [18]. PhyGenBench provides 160 prompts over 27 physical laws, and its authors report that current models struggle with physical commonsense [35].

Reward models are the second evaluation strand. VideoReward introduces a reward model and alignment algorithms in one RL framework to address unsmooth motion and text misalignment [36]. MJ-VIDEO offers a large fine-grained preference benchmark and a mixture-of-experts reward model [37]. SoliReward targets reward hacking and annotation noise in video reward models using single-item annotations and hierarchical attention [20]. For joint video-audio generation, VA-Judger provides a chain-of-thought reward model and preference dataset, replacing fragmented metrics with dimension-wise RL [38].

Safety is treated in parallel. A 2025 ACM Computing Surveys review covers detection, disruption and authentication (watermarking) against AI-generated visual media, including video, and notes that most detectors generalise poorly across domains and that watermarks can be removed by compression, cropping, reconstruction and adversarial attacks [39].

## Trends and open problems

Several trends stand out in 2025-2026 work. Efficiency is a central theme: rCM is a score-regularised continuous-time consistency model for large-scale diffusion distillation [40]; sparse attention methods aim to cut the latency of long-sequence diffusion transformers, though one 2026 paper states that existing methods degrade quality at high sparsity and proposes training-free MC-Sparse [41]. Real-time interactive systems apply distillation and memory mechanisms, as in StreamChar, which reports a two-stage distillation for streaming audio-video characters [42]. Evaluation has moved toward audio-visual judges and intrinsic physical criteria [38][18].

Open problems remain. First, long-horizon consistency: autoregressive methods trade memory against compute and still drift over long rollouts [8][11], and world models must handle memory for long interactions [33]. Second, physical plausibility: current models struggle with physical commonsense [35], and VBench-2.0 reports the same gap in intrinsic dimensions [18]. Third, reward modelling: reward hacking and label noise undermine RLHF-style alignment of video models [20]. Fourth, the choice between discrete and continuous representations for unified models is disputed, with semantic-level features argued to be preferable to pixel-level VAE tokens [25]. Fifth, audio-video synchronisation requires stronger cross-modal alignment than text-vision [25][14]. Sixth, safety: detectors generalise poorly across domains and watermarks are fragile [39]. Finally, the sources gathered here offer limited evidence on cascaded pipelines (base model plus super-resolution or frame interpolation), image-to-video specifics, and commercial system design, so these remain gaps in this survey.

## References
[1] Wan: Open and Advanced Large-Scale Video Generative Models. web. https://arxiv.org/abs/2503.20314 (2025-04-19)
[2] Scalable Diffusion Models with Transformers (DiT). web. https://arxiv.org/abs/2212.09748 (2022-12-19)
[3] High-Resolution Image Synthesis with Latent Diffusion Models. hf-search. https://huggingface.co/papers/2112.10752 (2021-12-20)
[4] Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. hf-search. https://huggingface.co/papers/2304.08818 (2023-04-18)
[5] VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control. hf-search. https://huggingface.co/papers/2407.12781 (2024-07-17)
[6] AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers. hf-search. https://huggingface.co/papers/2411.18673 (2024-11-27)
[7] DiTraj: training-free trajectory control for video diffusion transformer. hf-search. https://huggingface.co/papers/2509.21839 (2025-09-26)
[8] Context Forcing: Consistent Autoregressive Video Generation with Long Context. hf-search. https://huggingface.co/papers/2602.06028 (2026-02-05)
[9] VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory. hf-search. https://huggingface.co/papers/2512.04519 (2025-12-04)
[10] Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization. hf-search. https://huggingface.co/papers/2602.02958 (2026-02-03)
[11] TokenTrim: Inference-Time Token Pruning for Autoregressive Long Video Generation. hf-search. https://huggingface.co/papers/2602.00268 (2026-01-30)
[12] Emu3: Next-Token Prediction is All You Need. hf-search. https://huggingface.co/papers/2409.18869 (2024-09-27)
[13] Emu3.5: Native Multimodal Models are World Learners. hf-search. https://huggingface.co/papers/2510.26583 (2025-10-30)
[14] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization. hf-search. https://huggingface.co/papers/2503.23377 (2025-03-30)
[15] AV-DiT: Efficient Audio-Visual Diffusion Transformer for Joint Audio and Video Generation. hf-search. https://huggingface.co/papers/2406.07686 (2024-06-11)
[16] Talker-T2AV: Joint Talking Audio-Video Generation with Autoregressive Diffusion Modeling. hf-search. https://huggingface.co/papers/2604.23586 (2026-04-26)
[17] 3MDiT: Unified Tri-Modal Diffusion Transformer for Text-Driven Synchronized Audio-Video Generation. hf-search. https://huggingface.co/papers/2511.21780 (2025-11-26)
[18] VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness. web. https://arxiv.org/abs/2503.21755 (n.d.)
[19] Video-Bench: Human-Aligned Video Generation Benchmark (CVPR 2025). hf-search. https://huggingface.co/papers/2504.04907 (2025-04-07)
[20] SoliReward: Mitigating Reward Hacking and Annotation Noise in Video Generation Reward Models. hf-search. https://huggingface.co/papers/2512.22170 (2025-12-17)
[21] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models. hf-search. https://huggingface.co/papers/2503.14325 (2025-03-18)
[22] OD-VAE: An Omni-dimensional Video Compressor for Improving Latent Video Diffusion Model. hf-search. https://huggingface.co/papers/2409.01199 (2024-09-02)
[23] Delving into Latent Spectral Biasing of Video VAEs for Superior Diffusability. hf-search. https://huggingface.co/papers/2512.05394 (2025-12-05)
[24] Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities. hf-search. https://huggingface.co/papers/2505.02567 (2025-05-05)
[25] Towards Unified Multimodal Large Language Models: More than A Survey. web. https://aclanthology.org/2026.findings-acl.1853.pdf (n.d.)
[26] Bridging Text and Video Generation: A Survey. web. https://arxiv.org/pdf/2510.04999 (n.d.)
[27] A Comparative Review of Autoregressive and Diffusion Models for Video Generation. web. https://liweinlp.com/13198 (n.d.)
[28] LTX-Video: Realtime Video Latent Diffusion. hf-search. https://huggingface.co/papers/2501.00103 (2024-12-30)
[29] HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation. hf-search. https://huggingface.co/papers/2512.23464 (2025-12-29)
[30] OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation. hf-search. https://huggingface.co/papers/2407.02371 (2024-07-02)
[31] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. hf-search. https://huggingface.co/papers/2408.06072 (2024-08-12)
[32] NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation. hf-search. https://huggingface.co/papers/2601.02204 (2026-01-05)
[33] Towards Interactive Video World Modeling: Frontiers, Challenges, Benchmarks, and Future Trends. web. https://arxiv.org/html/2606.01164v1 (n.d.)
[34] World Action Models: A Survey (Dream Less, Act More). web. https://arxiv.org/html/2606.20781 (2026-06-18)
[35] Towards World Simulator: PhyGenBench and PhyGenEval. web. https://proceedings.mlr.press/v267/meng25c.html (2025-10-06)
[36] Improving Video Generation with Human Feedback (VideoReward). hf-search. https://huggingface.co/papers/2501.13918 (2025-01-23)
[37] MJ-VIDEO: Fine-Grained Benchmarking and Rewarding Video Preferences in Video Generation. hf-search. https://huggingface.co/papers/2502.01719 (2025-02-03)
[38] VA-Judger: Reward Modeling from Human Preference Feedback for Joint Video-Audio Generation. hf-search. https://huggingface.co/papers/2608.18607 (2026-08-19)
[39] Survey of detection, disruption and authentication defenses against AI-generated visual media (ACM Computing Surveys). web. https://dl.acm.org/doi/10.1145/3770916 (2025-11-20)
[40] rCM: Score-regularized continuous-time consistency model for large-scale diffusion distillation. hf-search. https://huggingface.co/papers/2510.08431 (2025-10-09)
[41] MC-Sparse: Meta-Cached Sparse Attention for diffusion transformers. hf-daily. https://huggingface.co/papers/2610.06801 (2026-10-05)
[42] StreamChar: real-time streaming audio-video character generation. hf-search. https://huggingface.co/papers/2605.25659 (2026-05-25)
