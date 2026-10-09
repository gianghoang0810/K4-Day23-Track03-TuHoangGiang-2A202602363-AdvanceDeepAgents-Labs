# Efficient Inference and Small Language Models: A Survey (2023–2026)

## TL;DR

- Small language models (SLMs) can match much larger models on some benchmarks when data quality is the focus; the Phi line reports a 3.8B phi-3-mini reaching 69% on MMLU, and an ACL 2025 study reports Phi-3.5-mini (2.7B) as the most accurate SLM it evaluated as of September 2024 [1][2].
- Post-training 4-bit quantization is the most practical compression route: on Qwen2.5-32B, AWQ, GPTQ and GGUF variants stayed within about 6% perplexity of FP16, and the serving kernel changed throughput more than the algorithm did [3]. Pruning and distillation evidence is thinner and mostly qualitative [4][5][6].
- At serving time, memory management (PagedAttention, vAttention) and lossless speculative decoding (EAGLE, Medusa, DFlash) are the main levers, but KV-cache eviction can break system-prompt fidelity and early exit is yielding less on newer dense LLMs [7][8][9][10][11][12].
- Energy efficiency of SLMs is not monotone in size: a study of 70+ SLMs finds smaller models are not automatically more efficient, and the performance–energy Pareto frontier flattens [13][14].
- Small reasoning models (e.g., VibeThinker-3B, Falcon-H1R 7B) show that test-time scaling can be pushed into compact models, but recent work finds self-refinement mostly re-weights already-reachable solutions [15][16][17].

## Background

**Definitions.** Efficient inference covers methods that reduce the memory, latency or energy of running a trained model, including compression, decoding-time acceleration and serving systems. SLMs are compact models (typically under about 7B parameters) designed for resource-limited settings; the surveys in this area define them and benchmark them across architecture, training data and evaluation [18][19][20].

**Why it matters now.** SLM surveys from 2024 to 2025 catalogue many recent compact models (e.g., Phi-4-mini, SmolLM-2, Qwen-2.5 small variants) and argue for standard definitions and evaluation for SLMs [18][20][21]. A 2025 agentic-systems survey argues SLMs suit schema- and API-constrained workloads such as tool use and retrieval-augmented generation, with lower cost and latency, although its evidence is largely qualitative [22]. Efficient reasoning has become a separate research thread, since long chains of thought raise inference cost [23].

**Foundational work.** Three strands anchor the field. (i) Data-centric small models: phi-1 showed a compact code model with high benchmark accuracy trained on textbook-quality data [24], and phi-1.5 (1.3B) was reported as comparable to much larger models on common-sense reasoning [25]. The Phi-3 report extends the recipe to phi-3-mini (3.8B, 3.3T tokens), explicitly positioning the data approach as a deviation from standard scaling laws [1]. (ii) Open small models: TinyLlama trained a 1.1B Llama-2-architecture model on about 1 trillion tokens, deliberately past the compute-optimal token budget, to study small-model capacity [26]. (iii) Post-training compression and pruning: SparseGPT showed that large GPT models can be pruned to 50% sparsity in one shot without retraining [27].

*Coverage caveat.* The arXiv search family returned rate-limit errors on every call during this survey, so classic papers such as GPTQ, AWQ, FlashAttention, scaling-law and speculative-decoding originals, BitNet and MobileLLM are referenced here only through the secondary sources that cite them, and are not separately cited. Readers should consult those primary papers directly.

## Compression: quantization, pruning and distillation

**Quantization is the most mature option.** In a 2026 benchmark on Qwen2.5-32B-Instruct, the FP16 baseline reached word perplexity 6.56 and HumanEval pass@1 56.1%, while 4-bit AWQ, GPTQ, GGUF Q4_K_M and BitsandBytes stayed within about 6% perplexity of baseline. Throughput differed far more by kernel than by algorithm: Marlin-AWQ reached 741 tok/s against 68 tok/s for AWQ without the Marlin kernel [3]. A second comparison on Llama-2-13B finds EXL2 gives the best size–perplexity frontier, while AWQ and GPTQ at 4-bit trade more VRAM for accuracy, and bitsandbytes load_in_4bit is the slowest at 23 tokens/s [28]. Surveys agree that low-cost post-training quantization is the most widely used LLM compression method, yet it "suffers severe accuracy degradation" without improvements [29]. Extreme low-bit settings remain fragile: OneBit proposes 1-bit quantization-aware training [30], and an empirical study finds LLaMA3 degrades significantly at low bit-widths [31].

**Pruning trades accuracy for structure.** SparseGPT established one-shot unstructured pruning at scale [27]. Depth pruning is an alternative: Shortened LLaMA reports that removing layers matches width pruning on zero-shot tasks and improves speed under memory constraints [5]. CFSP proposes structured pruning using coarse-to-fine activation information to cut latency across sparsity budgets [32]. Comparing these families is hard because the sources report different models and metrics; the 2025 review argues for multi-objective evaluation that couples accuracy and perplexity with latency–accuracy and parameter-efficiency trade-offs [4].

**Distillation is the bridge to small models.** Surveys list knowledge distillation alongside quantization and pruning as the three core compression families [4][33]. TAID targets the capacity gap and mode averaging between large teachers and small students with a temporally adaptive interpolation [6]. The phi line can be read as distillation-by-data: phi-2 (2.7B) matched models 25 times larger trained on regular data, and phi-3 used LLM-generated synthetic data alongside filtered web data [1]. No retrieved source compares quantization, pruning and distillation head-to-head on one model, so the relative rankings above are not established by a controlled study.

## Serving and decoding acceleration

**Memory management.** PagedAttention and the vLLM system reduce KV-cache waste and raise serving throughput [7]. vAttention offers an alternative that uses the operating system's demand paging without changing attention kernels or serving frameworks [8]. Both are systems-level and do not change model outputs.

**Speculative decoding.** A small draft model proposes tokens that the target verifies. EAGLE operates on features and is reported as lossless relative to vanilla autoregressive decoding [9]. Medusa adds parallel decoding heads with tree-based attention and reports significant speed-ups with minimal latency [10]. DFlash replaces the drafter with a lightweight block-diffusion model for parallel drafting and claims significant speed-ups while maintaining output quality [34]. The retrieved abstracts give few numbers: the only explicit speed-up in this survey's sources is 1.4x, from speech synthesis rather than text LLMs [35]. Treat cross-method comparisons as qualitative.

**KV-cache compression.** Eviction reduces memory and latency, but it has costs. A 2025 study shows KV compression can degrade multi-instruction tasks, notably system-prompt leakage, and that better eviction policies mitigate this [11]. A 2026 sparse-first engine, SparseEngine, argues that heterogeneous cache representations hinder integration with existing engines and proposes a lifecycle contract per method [36].

**Early exit and skipping.** Layer-level early exit was promising, but a 2026 study finds its effectiveness is decreasing in newer LLMs because of lower layer redundancy, with dense transformers offering more exit potential than MoE and state-space architectures [12]. Token-level and feed-forward skipping methods report minimal quality impact, but the evidence is at abstract level [12].

**Edge and routing.** On edge GPUs, EdgeReasoning characterises the latency–accuracy trade-off of reasoning LLMs through architecture choice, token management and scaling strategies [37]. TokenRouter (2026) argues that fine-grained token-level routing across models can move the cost–quality frontier, and addresses batching problems this creates [38]. For memory-bound decoding, SparseDecoding proposes training-free, Hessian-guided layer pruning that reduces parameters read per decoding step [39].

## Small reasoning models and test-time compute

Compact reasoning models are a 2025–2026 focus. VibeThinker-3B claims state-of-the-art verifiable reasoning in a small model through specialised training, challenging conventional scaling assumptions [15]. Falcon-H1R describes a 7B hybrid model built for efficient test-time scaling [16]. Efficient Reasoning on the Edge combines LoRA adapters, budget forcing via reinforcement learning and parallel test-time scaling under strict resource limits [40]. Token-efficient training methods report that controlling reasoning length can improve small models without a significant accuracy loss [41].

The limits are also documented. A 2026 study of small reasoning models finds self-refinement largely consolidates probability mass onto solutions already reachable, rather than creating new ones, and identifies execution bottlenecks where the correct path is reachable but not taken [17]. An on-policy distillation study finds that the method transfers compositional reasoning skill to unseen structures, but transfers minimal factual knowledge [42]. Together these results suggest that reasoning gains in small models come from skill and procedure more than from new knowledge.

## Trends and open problems

**What changed in the last two years.** The field has moved from proving that small models can be accurate [24][25][1] to measuring their cost. SLM benchmarks now include energy and CO2: SLM-Bench evaluates 15 SLMs on 9 NLP tasks across 4 hardware configurations and reports that accuracy gains often come with higher energy use [14]. A 2026 study of 70+ SLMs finds that newer models improve the performance–energy trade-off, but gains plateau: a one-point accuracy gain from Qwen-3-8B to Qwen-3-14B nearly doubles inference energy on one benchmark [13]. Serving has shifted toward systems and routing [36][38], and reasoning has become a dimension of efficiency [23][40].

**Unresolved open problems.**
- *Evaluation is inconsistent.* Surveys call for standard definitions and metrics [18][20][4], and the on-device study notes that prior on-device benchmarks cover only a few SLMs [2]. The energy studies disable reasoning for fairness [13], which makes them incomparable with reasoning-enabled results.
- *Data and contamination.* Phi-3.5-mini's top accuracy may reflect data engineering and possible dataset overfitting [2]; the phi-family recipe relies on filtered web data plus LLM-generated synthetic data [1], so its benchmark gains depend on that data mix.
- *Reliability under compression.* Quantization and KV eviction degrade specific capabilities (low-bit LLaMA3 [31]; system-prompt leakage [11]) that average benchmarks may hide. Compression benchmarks for agentic capabilities are emerging [23][22] but are not yet standard.
- *Scaling of efficiency techniques.* Early exit is losing effect on newer dense models [12], and draft-model speed-ups are often reported qualitatively [34][10]. Whether these gains persist at frontier scale is open.
- *Small-model in-context learning.* An ACL 2025 study finds SLM in-context learning remains limited even when general accuracy is high, and that efficiency has substantial optimisation headroom [2].
- *Limits of test-time scaling.* Self-refinement and reasoning distillation may not add new reachable solutions or new facts [17][42]; how to get reliable reasoning from very small models is unresolved.

## References
[1] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. hf-search. https://huggingface.co/papers/2404.14219 (2024-04-22)
[2] Demystifying Small Language Models for Edge Deployment. web. https://aclanthology.org/anthology-files/anthology-files/pdf/acl/2025.acl-long.718.pdf (2025-n.d.)
[3] The Complete Guide to LLM Quantization with vLLM. web. https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks (2026-01-07)
[4] A review of state-of-the-art techniques for large language model compression. web. https://link.springer.com/article/10.1007/s40747-025-02019-z (2025-08-01)
[5] Shortened LLaMA: A Simple Depth Pruning for Large Language Models. hf-search. https://huggingface.co/papers/2402.02834 (2024-02-05)
[6] TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models. hf-search. https://huggingface.co/papers/2501.16937 (2025-01-28)
[7] Efficient Memory Management for Large Language Model Serving with PagedAttention. hf-search. https://huggingface.co/papers/2309.06180 (2023-09-12)
[8] vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention. hf-search. https://huggingface.co/papers/2405.04437 (2024-05-07)
[9] EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. hf-search. https://huggingface.co/papers/2401.15077 (2024-01-26)
[10] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. hf-search. https://huggingface.co/papers/2401.10774 (2024-01-19)
[11] The Pitfalls of KV Cache Compression. hf-search. https://huggingface.co/papers/2510.00231 (2025-09-30)
[12] The Diminishing Returns of Early-Exit Decoding in Modern LLMs. hf-search. https://huggingface.co/papers/2603.23701 (2026-03-24)
[13] Mapping the Efficiency Landscape of Small Language Models. web. https://www.ijcai.org/proceedings/2026/0627.pdf (2026-n.d.)
[14] SLM-Bench: A Comprehensive Benchmark of Small Language Models. web. https://aclanthology.org/2025.findings-emnlp.1165.pdf (2025-n.d.)
[15] VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models. hf-search. https://huggingface.co/papers/2606.16140 (2026-06-15)
[16] Falcon-H1R: Pushing the Reasoning Frontiers with a Hybrid Model for Efficient Test-Time Scaling. hf-search. https://huggingface.co/papers/2601.02346 (2026-01-05)
[17] Knowing When Thinking Is Not Enough: Teaching Small Reasoning Models to Reason Beyond Their Parametric Knowledge. hf-search. https://huggingface.co/papers/2609.34327 (2026-09-28)
[18] Small Language Models: Survey, Measurements, and Insights. hf-search. https://huggingface.co/papers/2409.15790 (2024-09-24)
[19] A Survey of Small Language Models. hf-search. https://huggingface.co/papers/2410.20011 (2024-10-25)
[20] A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness. hf-search. https://huggingface.co/papers/2411.03350 (2024-11-04)
[21] A Survey on Small Language Models in the Era of Large Language Models: Architecture, Capabilities, and Trustworthiness. web. https://dl.acm.org/doi/epdf/10.1145/3711896.3736563 (2025-n.d.)
[22] Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade offs. hf-search. https://huggingface.co/papers/2510.03847 (2025-10-04)
[23] Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models. hf-search. https://huggingface.co/papers/2503.16419 (2025-03-20)
[24] Textbooks Are All You Need (phi-1). hf-search. https://huggingface.co/papers/2306.11644 (2023-06-20)
[25] Textbooks Are All You Need II: phi-1.5 technical report. hf-search. https://huggingface.co/papers/2309.05463 (2023-09-11)
[26] TinyLlama: An Open-Source Small Language Model. web. https://arxiv.org/abs/2401.02385 (2024-06-04)
[27] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. hf-search. https://huggingface.co/papers/2301.00774 (2023-01-02)
[28] A detailed comparison between GPTQ, AWQ, EXL2, q4_K_M, q4_K_S, and load_in_4bit (oobabooga blog). web. https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/ (n.d.)
[29] A Comprehensive Survey of Compression Algorithms for Language Models. web. https://arxiv.org/pdf/2401.15347 (n.d.)
[30] OneBit: Towards Extremely Low-bit Large Language Models. hf-search. https://huggingface.co/papers/2402.11295 (2024-02-17)
[31] How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study. hf-search. https://huggingface.co/papers/2404.14047 (2024-04-22)
[32] CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information. hf-search. https://huggingface.co/papers/2409.13199 (2024-09-20)
[33] A Survey on Model Compression for Large Language Models (TACL). web. https://aclanthology.org/2024.tacl-1.85/ (n.d.)
[34] DFlash: Block Diffusion for Flash Speculative Decoding. hf-search. https://huggingface.co/papers/2602.06036 (2026-02-05)
[35] Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding. hf-search. https://huggingface.co/papers/2505.15380 (2025-06-03)
[36] SparseEngine: Sparse-First Inference Engine. hf-daily. https://huggingface.co/papers/2609.39068 (2026-09-30)
[37] EdgeReasoning: Characterizing Reasoning LLM Deployment on Edge GPUs. hf-search. https://huggingface.co/papers/2511.01866 (2025-10-21)
[38] TokenRouter: Efficient Serving System for Token-Level LLM Routing. hf-daily. https://huggingface.co/papers/2610.12242 (2026-10-08)
[39] SparseDecoding: Decoding-Aware Pruning for Accurate and Efficient LLM Inference. hf-daily. https://huggingface.co/papers/2610.12327 (2026-10-08)
[40] Efficient Reasoning on the Edge. hf-search. https://huggingface.co/papers/2603.16867 (2026-03-17)
[41] Making Small Language Models Efficient Reasoners: Intervention, Supervision, Reinforcement. hf-search. https://huggingface.co/papers/2505.07961 (2025-05-12)
[42] On-Policy Distillation Teaches New Skills but Not New Knowledge. hf-daily. https://huggingface.co/papers/2610.09639 (2026-10-07)
