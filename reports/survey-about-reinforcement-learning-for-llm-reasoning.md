# Reinforcement Learning for LLM Reasoning: A Survey

## TL;DR

- Reinforcement learning moved from preference alignment to reasoning. InstructGPT established the SFT, reward model, RL pipeline, and a 1.3B InstructGPT model was preferred over 175B GPT-3 [1]. DPO later removed the explicit RL loop with a classification loss [2].
- Reinforcement learning with verifiable rewards (RLVR) drives the current reasoning recipes. GRPO was introduced as a PPO variant that reduces memory cost for math reasoning [3]. DeepSeek-R1 reports reasoning performance comparable to OpenAI-o1-1217 using RL with multi-stage training and cold-start data [4].
- Recipe-level work (DAPO, Dr. GRPO) addresses reproducibility and token efficiency. DAPO reports 50 points on AIME 2024 with a Qwen2.5-32B base [5], and Dr. GRPO studies how pretraining and the R1-Zero recipe affect RL results [6].
- Credit assignment is the main research axis after outcome rewards. Process reward models (PRMs) gave large gains in some settings [7], and one on-policy tree-search study argues that static, separately trained PRMs can suffer distribution shift and reward hacking [8].
- Open problems include entropy collapse, reward hacking, memorisation shortcuts from spurious rewards, multi-turn agentic instability, and disputes over whether RL extends capability or only sharpens existing behaviour [9][10][11][12][13].

## Background

Reinforcement learning (RL) for language models optimises a policy against a reward signal produced after generation. The canonical alignment pipeline combined supervised fine-tuning on demonstrations, a reward model trained on human rankings, and RL fine-tuning against that reward model. InstructGPT showed that this pipeline improves truthfulness and reduces toxic output with minimal regressions on public NLP datasets [1]. The RL step in that line of work is widely associated with Proximal Policy Optimization (PPO), a policy-gradient method that alternates sampling with a clipped surrogate objective and allows multiple minibatch epochs per batch [14]. The InstructGPT abstract does not itself name PPO, so this link is background knowledge rather than a retrieved claim.

Two developments make the topic urgent for reasoning specifically. First, DPO reparameterises the RLHF objective so that the optimal policy can be extracted with a simple classification loss, without sampling from the LM during fine-tuning [2]. Second, reasoning models trained with RL on verifiable answers (math, code) showed that long chains of thought can emerge from RL alone, as with DeepSeek-R1-Zero [4]. The field now debates which RL ingredients are necessary, what the rewards actually teach, and how far the gains generalise [15][16].

## Preference optimisation: from PPO-RLHF to DPO and its successors

The earliest recipe optimises a learned reward with PPO. InstructGPT reported that the 1.3B model was preferred to GPT-3 with 175B parameters [1]. DPO replaces this loop: in experiments it exceeded PPO-based RLHF at controlling sentiment and matched or improved response quality on summarisation and single-turn dialogue [2]. A later line of analysis reframes GRPO itself as a contrastive objective linked to DPO, and reports that a minimal two-rollout variant performs comparably to larger groups at lower compute [17]. A separate theory paper characterises the GRPO policy gradient as a U-statistic and claims asymptotic equivalence to an oracle policy gradient [18]. The practical lesson is that the distinction between offline preference learning and online RL is narrower for verifiable-reward training than the terminology suggests, although these theoretical claims come from recent preprints whose experimental evidence was not independently checked here.

## Verifiable-reward recipes: GRPO, DAPO, Dr. GRPO and R1-style training

GRPO removes the value network of PPO and computes advantages relative to a group of sampled responses; DeepSeekMath used it to reach 51.7% on the MATH benchmark without external tools, and 60.9% with self-consistency over 64 samples [3]. DeepSeek-R1-Zero applies large-scale RL without a preliminary SFT stage and shows reasoning behaviours emerging, with poor readability and language mixing; DeepSeek-R1 adds cold-start data and multi-stage training [4]. The paper also releases six dense models distilled from R1 [4].

Later recipe papers focus on making these results reproducible and on removing biases. DAPO (decoupled clip and dynamic sampling policy optimisation) argues that the key details of o1 and R1 were withheld, which made RL results hard to reproduce, and reports 50 points on AIME 2024 with Qwen2.5-32B [5]. Dr. GRPO, from a critical study of R1-Zero-like training, identifies effects of pretraining on RL performance and proposes a variant aimed at token efficiency with superior AIME 2024 accuracy [6]. A systematic review of RL tricks argues that a minimalist combination of techniques can outperform more elaborate strategies, although the retrieved summary gives no numbers [15]. Off-policy GRPO is reported to match or exceed on-policy GRPO in RLVR settings [19].

The comparison with distillation is contested. One study reports that a simple distillation method with 920 examples outperforms zero-RL in flexibility and in advanced cognitive behaviours [16]. AceReason-Nemotron claims that large-scale RL improves small and mid-sized reasoning models more than distillation does [20]. The two findings are not directly contradictory, because they vary the model size, starting checkpoint and metric, but they show that the answer depends on the setup.

## Credit assignment, process rewards and tree search

Outcome rewards give one signal per response, which is sparse for long reasoning chains. OmegaPRM uses divide-and-conquer Monte Carlo tree search to locate the first error in a chain of thought by binary search, and collects more than 1.5 million process annotations without human labels; with weighted self-consistency, an instruction-tuned Gemini Pro reaches 69.4% on MATH, up from 51% [7]. Tree-based methods bring this search inside RL. TreeRL integrates on-policy tree search into training, derives per-step signals from global and local advantages, and argues that static PRMs can suffer distribution shift and reward hacking [8]. Tree-GRPO applies the same idea to agentic rollouts, though the retrieved summary gives no numbers [21].

Critic-free and implicit alternatives try to avoid training a separate step-level model. PRIME uses implicit process rewards for online RL [22]; Free Process Rewards parametrises outcome rewards as log-likelihood ratios so that an implicit PRM can be trained without step labels [23]. TEMPO applies prefix-tree temporal-difference corrections for token-level credit and claims to outperform PPO and GRPO, and CAPO uses an LLM as a generative PRM to supply token-level rewards in RLVR [24][25]. The notes did not contain verified numbers for these critic-free and implicit methods, so their comparative claims should be read as reported by the authors.

Test-time scaling links these training signals to inference. RL^V unifies a reasoner with a verifier to improve MATH accuracy and efficient test-time compute scaling [26]. A study of PRM generalisation in test-time scaling finds diminishing returns as PRM scale grows, reports that Monte Carlo Tree Search is the most effective scaling method when compute is abundant and Best-of-N is a practical alternative under limits, and reports that math-trained PRMs transfer comparably to code [27]. For code, CodeScaler proposes an execution-free reward model to reduce latency relative to execution-based methods [28]. The reward-model literature therefore splits between process-level signals used during training and verifiers used at inference, and the two are not yet evaluated on a common benchmark in the sources reviewed.

## Stability, exploration and entropy

A recurring failure mode is policy entropy collapse. A survey of large reasoning models notes that basic RL algorithms can easily cause entropy collapse, and cites work arguing that RLVR gains come from the entropy drop, which would set an entropy-linked ceiling on capability [11]. The entropy mechanism analysis studies how entropy dynamics can be controlled to prevent collapse and improve exploration [10]. A paper on exploration, clipping and spurious rewards argues that these effects can be explained by reduced clipping bias and policy entropy [9]. Clipping-based stability is also under scrutiny: CPGD proposes constraining policy drift and clipping updates to stabilise rule-based RL [29], and ReSPO argues that clipping causes sign-dependent gradient starvation under rollout reuse and replaces it with a smooth two-branch objective [30].

Diversity is a separate concern. LaDi-RL argues that discrete-token RL suffers diversity collapse as entropy decreases, and explores in continuous latent space [31]. Training-inference mismatch is treated as a stability factor: one diagnosis paper states that it can cause training collapse [32], and a later paper proposes an objective for consistent improvement between training and inference policies [33].

## Generalisation, memorisation and spurious rewards

Several results suggest that RLVR gains are not always about the reward content. The spurious-reward study reports that RLVR with various rewards, including spurious ones, improves mathematical reasoning in Qwen2.5-Math-7B, but the effect does not hold consistently across model families [13]. A mechanistic analysis attributes spurious-reward gains to a memorisation shortcut activated in the model, identified with neural circuit analysis and causal steering [12]. Exploration and clipping analysis reaches a related conclusion, that spurious rewards and entropy minimisation help "despite seemingly paradoxical effects" [9].

On generalisation after SFT, the mechanistic survey reports that SFT can cause out-of-distribution drops that RL partly repairs, and that RL after SFT on full traces generalises poorly while RL after SFT on atomic skills generalises out of distribution [11]. Work on weak supervision reports that generalisation depends on reward saturation dynamics and that SFT on explicit traces is crucial for adaptation [34]. The survey of RL under data scarcity identifies scarce fine-grained feedback, preference data, expert annotations and step-by-step traces as bottlenecks [35].

## Trends and open problems

The last two years shifted from single-turn math RL to agentic, multi-turn and code settings [36][37][28]. Multi-turn agentic RL is reported as unstable: RAGEN-2 identifies template collapse as a failure mode that entropy does not detect, and proposes mutual-information proxies and SNR-aware filtering [36]. A practitioner's guide offers design choices for multi-turn RL across environments [37], and T^2PO controls exploration at token and turn levels to stabilise it [38]. Reward hacking is another open front: the CATCH testbed for coding RL states that monitoring and mitigating hacking during training remain challenging, partly because testbeds that reproduce hacking are scarce [39]. Cooper co-optimises policy and reward models to improve robustness [40], and PURE uses min-form credit assignment to reduce reward hacking in PRMs [41].

Length and overthinking are another active area. LASER-D adapts length-based reward shaping to difficulty to reduce redundancy [42]. The mechanistic survey reports an inverse-U relation between accuracy and reasoning length and notes that models often allocate long chains to simple problems and too little to complex ones [11].

Open problems, with the supporting evidence and the disputes:

- **Does RL extend capability or only sharpen it?** The survey reports that numerous studies conclude RL does not truly enhance capability, while another finds that post-RL models produce solutions absent from the base model; how to reconcile these findings is unresolved [11]. Entropy-based explanations are also contested [9][10].
- **Is entropy collapse harmful?** One account credits the entropy drop for RLVR gains, while another argues that reasoning-path compression from entropy collapse degrades out-of-distribution performance [11].
- **Do spurious rewards generalise?** Gains on Qwen2.5-Math-7B do not transfer consistently across model families [13], and the mechanism is attributed to memorisation [12].
- **Which credit signal should be used?** One study argues that static PRMs can suffer distribution shift and reward hacking during RL [8], while other work proposes implicit or critic-free alternatives whose numbers were not verified in this survey [22][23][24][25].
- **How should SFT and RL be scheduled?** The survey states that scheduling SFT and RL is an open research question [11], and data scarcity limits how much human-feedback supervision is available [35].
- **How can multi-turn agentic RL be made stable and reward-robust?** Template collapse, exploration control and reward hacking remain open in agentic settings [36][38][39].

## References
[1] Training language models to follow instructions with human feedback. hf-search. https://huggingface.co/papers/2203.02155 (2022-03-04)
[2] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. hf-search. https://huggingface.co/papers/2305.18290 (2023-05-29)
[3] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. hf-search. https://huggingface.co/papers/2402.03300 (2024-02-05)
[4] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. hf-search. https://huggingface.co/papers/2501.12948 (2025-01-22)
[5] DAPO: An Open-Source LLM Reinforcement Learning System at Scale. hf-search. https://huggingface.co/papers/2503.14476 (2025-03-18)
[6] Understanding R1-Zero-Like Training: A Critical Perspective. hf-search. https://huggingface.co/papers/2503.20783 (2025-03-26)
[7] Improve Mathematical Reasoning in Language Models by Automated Process Supervision. hf-search. https://huggingface.co/papers/2406.06592 (2024-06-05)
[8] TreeRL: LLM Reinforcement Learning with On-Policy Tree Search. hf-search. https://huggingface.co/papers/2506.11902 (2025-06-13)
[9] Exploration v.s. Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward. hf-search. https://huggingface.co/papers/2512.16912 (2025-12-18)
[10] The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models. hf-search. https://huggingface.co/papers/2505.22617 (2025-05-28)
[11] Towards a Mechanistic Understanding of Large Reasoning Models: A Survey of Training, Inference, and Failures. web. https://aclanthology.org/2026.acl-long.889.pdf (2026)
[12] Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs. hf-search. https://huggingface.co/papers/2601.11061 (2026-01-16)
[13] Spurious Rewards: Rethinking Training Signals in RLVR. hf-search. https://huggingface.co/papers/2506.10947 (2025-06-12)
[14] Proximal Policy Optimization Algorithms. web. https://arxiv.org/abs/1707.06347 (2017-07-20)
[15] Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning. hf-search. https://huggingface.co/papers/2508.08221 (2025-08-11)
[16] Why Distillation can Outperform Zero-RL: The Role of Flexible Reasoning. hf-search. https://huggingface.co/papers/2505.21067 (2025-05-27)
[17] It Takes Two: Your GRPO Is Secretly DPO. hf-search. https://huggingface.co/papers/2510.00977 (2025-10-01)
[18] Demystifying Group Relative Policy Optimization: Its Policy Gradient is a U-Statistic. hf-search. https://huggingface.co/papers/2603.01162 (2026-03-01)
[19] Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training. hf-search. https://huggingface.co/papers/2505.22257 (2025-05-28)
[20] AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning. hf-search. https://huggingface.co/papers/2505.16400 (2025-05-22)
[21] Tree-GRPO: Tree Search for LLM Agent Reinforcement Learning. hf-search. https://huggingface.co/papers/2509.21240 (2025-09-25)
[22] PRIME: Process Reinforcement through Implicit Rewards. hf-search. https://huggingface.co/papers/2502.01456 (2025-02-03)
[23] Free Process Rewards without Process Labels. hf-search. https://huggingface.co/papers/2412.01981 (2024-12-02)
[24] TEMPO: Exploiting Tree Structure for Credit Assignment in RL Training of LLMs. hf-search. https://huggingface.co/papers/2509.18314 (2025-09-22)
[25] CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment. hf-search. https://huggingface.co/papers/2508.02298 (2025-08-04)
[26] Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers. hf-search. https://huggingface.co/papers/2505.04842 (2025-05-07)
[27] Generalization of Process Reward Models in Test-Time Scaling. web. https://ojs.aaai.org/index.php/AAAI/article/download/40289/44250 (n.d.)
[28] CodeScaler: Scaling Code LLM Training and Test-Time Inference via Execution-Free Reward Models. hf-search. https://huggingface.co/papers/2602.17684 (2026-02-04)
[29] CPGD: Toward Stable Rule-based Reinforcement Learning for Language Models. hf-search. https://huggingface.co/papers/2505.12504 (2025-05-18)
[30] ReSPO: Reshaped Sequence Policy Optimization for Gradient Starvation in Off-Policy Learning. hf-daily. https://huggingface.co/papers/2609.35433 (2026-09-28)
[31] Beyond Mode Elicitation: Diversity-Preserving RL via Latent Diffusion Reasoner. hf-search. https://huggingface.co/papers/2602.01705 (2026-02-02)
[32] Diagnosing Training Inference Mismatch in LLM Reinforcement Learning. hf-search. https://huggingface.co/papers/2605.14220 (2026-05-14)
[33] The Mirage of Optimizing Training Policies. hf-search. https://huggingface.co/papers/2606.29526 (2026-06-28)
[34] When Can LLMs Learn to Reason with Weak Supervision?. hf-search. https://huggingface.co/papers/2604.18574 (2026-04-20)
[35] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity. web. https://arxiv.org/html/2604.17312v1 (2026-04)
[36] RAGEN-2: Reasoning Collapse in Agentic RL. hf-search. https://huggingface.co/papers/2604.06268 (2026-04-07)
[37] A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning. hf-search. https://huggingface.co/papers/2510.01132 (2025-10-01)
[38] T^2PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic RL. hf-search. https://huggingface.co/papers/2605.02178 (2026-05-04)
[39] CATCH: reward hacking testbed for coding RL. hf-search. https://huggingface.co/papers/2609.39533 (2026-10-01)
[40] Cooper: Co-Optimizing Policy and Reward Models in RL for LLMs. hf-search. https://huggingface.co/papers/2508.05613 (2025-08-07)
[41] Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning. hf-search. https://huggingface.co/papers/2504.15275 (2025-04-21)
[42] Adaptive Length-based Reward Shaping (LASER-D). hf-search. https://huggingface.co/papers/2505.15612 (2025-05-21)
