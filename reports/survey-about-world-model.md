# World Models: A Survey of Learned Environment Simulators for Prediction, Planning and Embodied AI

## TL;DR

- A world model is a learned model of how an environment evolves, used to predict futures, plan, or train an agent inside imagination; the idea goes back to Ha and Schmidhuber's agent trained in its own "dream" [1], and was extended into latent-imagination agents such as Dreamer [2], PlaNet [3] and DayDreamer on physical robots [4].
- Recent work splits into two families: latent-dynamics agents that plan in a compact state space (TD-MPC2 [5], DreamerV3 [6], Dreamer 4 [7]) and generative video world models that synthesise pixels conditioned on actions (Genie 3 [8], Matrix-Game [9], Cosmos [10][11]). Joint-embedding predictive approaches (JEPA) are proposed as a non-generative alternative [12][13].
- Generative video world models are becoming interactive and real-time with longer memory (Matrix-Game 3.0 [14]; Genie 3's real-time, memory-consistent generation [8]), but controllability and physical consistency remain the main bottlenecks [15][16].
- Evaluation is shifting from visual realism toward physical and action-faithfulness measurement: Physics-IQ [15], VideoPhy-2 [17], WorldSimProbe [18] and "World Models' Last Exam in Physics" [19] all report gaps that visual quality alone does not reveal.
- Robotics is a main deployment target, with world models used for policy evaluation [20][21] and policy refinement [22]; robot-focused surveys name physical accuracy and computational demands as open challenges [23][24].

## Background

**Definition.** In the Ha and Schmidhuber formulation, a world model is a generative neural network that learns a compressed spatial and temporal representation of an environment from unsupervised interaction. It has three parts: a VAE encoder (V) that maps frames to latent vectors, a recurrent MDN model (M) that predicts the distribution of the next latent given the action, and a small controller (C) trained by evolution strategies [1]. The controller is very small relative to the model, so most of the capacity sits in the world model, and the agent can be trained inside the model's hallucinated rollouts and transferred back to the real environment [1].

**Foundational line of work.** Dreamer learns behaviours by latent imagination from images and reports gains in data efficiency and final performance over prior methods [2]. PlaNet plans in a learned latent space from pixels [3]. DayDreamer applies the same world-model planning to physical robots and reports locomotion, manipulation and navigation learned directly from real-world interaction, without simulation [4]. LeCun's position paper frames a configurable predictive world model as a core module of autonomous intelligence and argues for non-generative joint-embedding prediction (JEPA), where the prediction error is measured in representation space rather than pixel space [12].

**Why it matters now.** Surveys in 2024–2026 describe world models as a shared substrate for understanding and predicting dynamics, with applications in generative games, autonomous driving, robotics and social simulation [25]. Robot-focused surveys frame video generation models as comprehensive world models for robotics, while naming physical accuracy and computational demands as open challenges [23][24]. Recent "world action model" surveys treat prediction of future states as a basis for decision-making [26]. The term is now used for both latent-planning agents and large generative simulators, which is why the taxonomy below separates them.

## Latent-Dynamics Agents and Planning

Model-based RL methods learn a dynamics model in a latent space and use it either for imagination-based policy learning or for online planning. Three design choices differ across this family.

**Imagination-based actor–critic.** Dreamer-style agents learn a policy by backpropagating value estimates through imagined latent trajectories [2]. DreamerV3 uses fixed hyperparameters across diverse domains and reports collecting diamonds in Minecraft without human data or curricula [6]. Dreamer 4 scales this recipe: it trains an agent with RL inside a fast and accurate world model and reports obtaining Minecraft diamonds from offline data alone [7]. The trade-off is that the quality of the imagined rollouts bounds the learned policy.

**Latent planning with MPC.** TD-MPC combines model-free value learning with model-based planning in a learned latent space, using a terminal value function to make short-horizon model predictive control efficient [27]. TD-MPC2 keeps a single hyperparameter set across tasks and benefits from larger models and datasets, according to its abstract [5]. Newer work continues this line: PL-MPC closes the planning–learning loop by modifying critic supervision [28], and hierarchical latent world models plan at multiple temporal scales for long-horizon embodied control [29]. Gradient-based planning through a learned world model suffers from a train–test gap, which the authors address by enhancing world-model training [30].

**Pixel-level generative agents.** DIAMOND argues that visual details matter for world-model agents on Atari, and achieves top performance on the Atari 100k benchmark with a diffusion world model [31]. MineWorld offers a real-time, open-source autoregressive Transformer world model of Minecraft that generates the next scene from the current scene and an action, and is reported to outperform diffusion-based baselines [32].

**Comparison.** Latent planners are reported as sample-efficient on continuous control [27][5]. Pixel-level generators preserve visual detail that policies can exploit or that humans can inspect [31], but they are more expensive to run, and the reviewed examples are mostly game domains [32][31]. Across the reviewed papers, the advantage of one family over the other is reported with self-selected benchmarks, so direct comparison across families is not yet standardised.

## Generative Video and Interactive World Models

Generative video world models use a video generator as the simulator: given an initial frame or prompt and an action stream, they produce the next frames. The main variants differ in how they receive control.

**Text-prompted, emergent-memory simulators.** Genie 3 generates navigable worlds from a text prompt at 24 frames per second and is described as retaining consistency for a few minutes at 720p. The blog presents consistency as an emergent capability of the generator rather than something imposed through an explicit 3D representation [8]. The same source notes that auto-regressive generation accumulates inaccuracies over time and that the model cannot yet simulate real-world locations with perfect accuracy [8].

**Keyboard/mouse action-conditioned game models.** Matrix-Game is trained in two stages, large-scale unlabelled pretraining followed by action-labelled training on a Minecraft dataset of over 2,700 hours of unlabelled and over 1,000 hours of labelled clips with keyboard and mouse annotations [9]. Its GameWorld Score measures visual quality, temporal quality, action controllability and physical rule understanding, and the authors report gains over Oasis and MineWorld, particularly in controllability and physical consistency [9]. Follow-ups target speed and memory: Matrix-Game 2.0 uses few-step auto-regressive diffusion for real-time streaming [33], and Matrix-Game 3.0 reports real-time 720p synthesis with long-term temporal consistency through memory augmentation [14]. Other game models take a different interface: Hunyuan-GameCraft-2 follows natural-language instructions rather than discrete actions [34], and GameGen-X adds an instruction-tuning stage to a text-to-video pretrained diffusion transformer for interactive open-world game video [35].

**Physical-AI simulators with multimodal and structured control.** NVIDIA's Cosmos line builds customisable world models from video curation, pre-trained models and tokenizers [11], and extends them with Cosmos-Predict2.5 and Cosmos-Transfer2.5 for Sim2Real and Real2Real translation [36]. Cosmos-Transfer1 steers generation with spatial controls such as segmentation, depth and edge maps, a route to control distinct from pure action conditioning [37]. Cosmos 3 is an omnimodal model that jointly generates language, image, video, audio and action sequences in one mixture-of-transformers architecture, with action tokens supporting forward dynamics, inverse dynamics and joint video–action generation [10].

**Robot-specific action-faithful models.** DreamTrue targets action following in robot video prediction. It identifies two obstacles: imprecise calibration can impair action following, and limited coverage of unsuccessful interactions biases predictions toward successful outcomes. It renders action trajectories into image-space conditions with offline geometric calibration, and adds counterfactual post-training [38].

**Comparison.** The interfaces differ in controllability. Text prompts (Genie 3, Hunyuan-GameCraft-2) give broad creative control but weaker, less precise action control [8][34]. Discrete keyboard/mouse conditioning (Matrix-Game) gives precise control within games [9]. Structured spatial maps (Cosmos-Transfer1) and action tokens (Cosmos 3) give control suited to robotics and driving [10][37]. Physical consistency is reported mainly in the model papers themselves, through self-designed scores [9], while independent physics benchmarks show persistent gaps for text-to-video models, including conservation-law violations [39][17].

## Evaluation and Benchmarks

Early evaluation relied on visual realism and frame-level metrics. Recent benchmarks test whether a world model is physically and action-faithful.

**Physical commonsense.** PhyGenBench and PhyGenEval test physical commonsense in text-to-video models and report that the shortcomings cannot be fully fixed by scaling models or prompt engineering [39]. VideoPhy-2 finds significant limitations in adherence to physical commonsense, particularly conservation laws, and introduces an automatic evaluation method [17]. Physics-IQ argues that current video models achieve visual realism without understanding the underlying physical principles [15], and PISA finds that physics post-training on simulated falling-object videos improves freefall modelling but reveals limits in generalisation [40].

**Action faithfulness.** WorldSimProbe formalises an "observable simulator contract": supplied actions must induce corresponding agent motion, and environment responses must be grounded in that motion. It evaluates six open-source action-conditioned models and reports that action realisation degrades as trajectories drift from the training distribution; it also argues that a VLM judge can overstate action following relative to human judgement [18]. WorldMark offers a unified benchmark with identical scenarios and controls across architectures [41]. WorldGym and the policy-evaluation work use video world models as surrogate environments for robot policy evaluation and report correlation with real-world performance [20][21].

**Physical realism by humans.** Physion-Eval uses human reasoning to judge physical realism and reports that, according to expert analysis, over 80% of generated videos contain identifiable physical glitches [16]. "World Models' Last Exam in Physics" argues that existing evaluations often rely on model-based judgements or reference videos, and proposes a measurement-based benchmark of 40 controlled tasks across mechanics, optics, fluids, thermal and phase-change phenomena, electromagnetism and surface phenomena [19]. WorldOlympiad diagnoses physical faithfulness, geometric consistency and interaction fidelity, and reports substantial gaps in physical reasoning for state-of-the-art models [42].

## Trends and Open Problems

**What changed in the last two years.** Three shifts are visible in the sources. First, generative simulators have moved to real-time, streaming and memory-augmented operation: Genie 3 [8], Matrix-Game 2.0 and 3.0 [33][14] and Cosmos 3 [10] all emphasise interactivity or long-horizon operation. Second, the JEPA line has grown and is now combined with vision–language reasoning: ThinkJEPA pairs latent world modelling with a large vision-language model for hand-manipulation trajectory prediction [13], and the JEPA line is being paired with VLM reasoning rather than pixel generation [13]. Third, evaluation has moved from realism to measurement, as the benchmarks above show [15][18][19].

**Compounding errors and long horizons.** A 2026 survey identifies compounding prediction errors, sim-to-real transfer and fragmented evaluation as persistent challenges for world models [43]. Auto-regressive video generation accumulates inaccuracies over time, which Genie 3 explicitly acknowledges [8], and memory mechanisms are one response [14]. Whether memory alone solves long-horizon consistency is not settled.

**Do video models learn physics?** This is disputed. Physics-IQ and PISA suggest that visual realism does not imply physical understanding [15][40], and human-judged benchmarks find frequent glitches [16]. Matrix-Game reports gains in physical rule understanding on its own score [9]. Because the independent results and the self-reported results use different metrics, the question is open; the sources do not show that scaling alone closes the gap [39].

**Action faithfulness and out-of-distribution control.** Action-conditioned models may appear responsive while miscalibrating the action-to-motion mapping [18]. DreamTrue links this to calibration and to the bias toward successful interactions in training data [38]. Systematic evaluation under distribution shift is still limited.

**Sim-to-real and fragmented evaluation.** The 2026 taxonomy survey reports that sim-to-real transfer and fragmented evaluation remain open [43]. Robot-focused surveys point to physical accuracy and computation as the main barriers to using video models as robot world models [23][24]. The use of world models for policy evaluation [20][21] is promising but depends on whether the simulator's ranking of policies matches the real world, which WorldSimProbe suggests can fail out of distribution [18].

**Compute, data and architecture.** Cosmos 3 and the hierarchical latent planners argue for unified and multi-scale designs [10][29], while the JEPA line argues against pixel generation as the training objective [12][13]. Surveys describe a convergence of chain-of-thought reasoning with world-model imagination and call for unified multimodal world models and foundation-scale interactive simulators [44]. The compute and data requirements of these designs are not quantified consistently across the reviewed sources, and no source in this survey settles the JEPA-versus-generative debate.

## References
[1] World Models. web. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] Dream to Control: Learning Behaviors by Latent Imagination. hf-search. https://huggingface.co/papers/1912.01603 (2019-12-03)
[3] Learning Latent Dynamics for Planning from Pixels. hf-search. https://huggingface.co/papers/1811.04551 (2019-06-04)
[4] DayDreamer: World Models for Physical Robot Learning. hf-search. https://huggingface.co/papers/2206.14176 (2022-06-28)
[5] TD-MPC2: Scalable, Robust World Models for Continuous Control. hf-search. https://huggingface.co/papers/2310.16828 (2023-10-25)
[6] Mastering Diverse Domains through World Models. hf-search. https://huggingface.co/papers/2301.04104 (2023-01-10)
[7] Training Agents Inside of Scalable World Models. hf-search. https://huggingface.co/papers/2509.24527 (2025-09-29)
[8] Genie 3: A new frontier for world models. web. https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ (2025-08-05)
[9] Matrix-Game: Interactive World Foundation Model. hf-daily. https://huggingface.co/papers/2506.18701 (2025-06-23)
[10] Cosmos 3: Omnimodal World Models for Physical AI. hf-search. https://huggingface.co/papers/2606.02800 (2026-06-01)
[11] Cosmos World Foundation Model Platform for Physical AI. hf-search. https://huggingface.co/papers/2501.03575 (2025-01-07)
[12] A Path Towards Autonomous Machine Intelligence. web. https://openreview.net/pdf?id=BZ5a1r-kVsf (n.d.)
[13] ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model. hf-search. https://huggingface.co/papers/2603.22281 (2026-03-23)
[14] Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory. hf-search. https://huggingface.co/papers/2604.08995 (2026-04-10)
[15] Do generative video models learn physical principles from watching videos?. hf-search. https://huggingface.co/papers/2501.09038 (2025-01-14)
[16] Physion-Eval: Evaluating Physical Realism in Generated Video via Human Reasoning. hf-search. https://huggingface.co/papers/2603.19607 (2026-03-20)
[17] VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation. hf-search. https://huggingface.co/papers/2503.06800 (2025-03-09)
[18] WorldSimProbe: Diagnosing Simulator Faithfulness in Action-Conditioned World Models for Embodied Manipulation. web. https://arxiv.org/html/2608.09298 (2026-08-10)
[19] World Models' Last Exam in Physics. hf-search. https://huggingface.co/papers/2610.08791 (2026-10-06)
[20] Scalable Policy Evaluation with Video World Models. hf-search. https://huggingface.co/papers/2511.11520 (2025-11-14)
[21] WorldGym: World Model as An Environment for Policy Evaluation. hf-search. https://huggingface.co/papers/2506.00613 (2025-09-30)
[22] World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation. hf-search. https://huggingface.co/papers/2509.19080 (2025-09-23)
[23] Video Generation Models in Robotics -- Applications, Research Challenges, Future Directions. hf-search. https://huggingface.co/papers/2601.07823 (2026-01-12)
[24] World Model for Robot Learning: A Comprehensive Survey. hf-search. https://huggingface.co/papers/2605.00080 (2026-04-30)
[25] Understanding World or Predicting Future? A Comprehensive Survey of World Models. hf-search. https://huggingface.co/papers/2411.14499 (2024-11-21)
[26] World Action Models: A Survey. hf-search. https://huggingface.co/papers/2606.20781 (2026-06-18)
[27] Temporal Difference Learning for Model Predictive Control. hf-search. https://huggingface.co/papers/2203.04955 (2022-03-09)
[28] Beyond Policy Alignment: Closing the Planning-Learning Loop for Robot Control with Learned World Models. hf-search. https://huggingface.co/papers/2609.39751 (2026-09-30)
[29] Hierarchical Planning with Latent World Models. hf-search. https://huggingface.co/papers/2604.03208 (2026-04-03)
[30] Closing the Train-Test Gap in World Models for Gradient-Based Planning. hf-search. https://huggingface.co/papers/2512.09929 (2025-12-10)
[31] Diffusion for World Modeling: Visual Details Matter in Atari. hf-search. https://huggingface.co/papers/2405.12399 (2024-05-20)
[32] MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft. hf-search. https://huggingface.co/papers/2504.08388 (2025-04-11)
[33] Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model. hf-search. https://huggingface.co/papers/2508.13009 (2025-08-18)
[34] Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model. hf-search. https://huggingface.co/papers/2511.23429 (2025-11-28)
[35] GameGen-X: Interactive Open-world Game Video Generation. hf-search. https://huggingface.co/papers/2411.00769 (2024-11-01)
[36] World Simulation with Video Foundation Models for Physical AI. hf-search. https://huggingface.co/papers/2511.00062 (2025-10-28)
[37] Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control. hf-search. https://huggingface.co/papers/2503.14492 (2025-03-18)
[38] DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training. hf-daily. https://huggingface.co/papers/2610.12468 (2026-10-08)
[39] Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation. hf-search. https://huggingface.co/papers/2410.05363 (2024-10-07)
[40] PISA Experiments: Exploring Physics Post-Training for Video Diffusion Models by Watching Stuff Drop. hf-search. https://huggingface.co/papers/2503.09595 (2025-03-12)
[41] WorldMark: A Unified Benchmark Suite for Interactive Video World Models. hf-search. https://huggingface.co/papers/2604.21686 (2026-04-23)
[42] WorldOlympiad: Can Your World Model Survive a Triathlon?. web. https://arxiv.org/abs/2606.11129 (2026-06-09)
[43] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. web. https://arxiv.science/abs/2606.00133 (2026-05-28)
[44] Understanding World or Predicting Future? A Comprehensive Survey of World Models (ACM Computing Surveys). web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
