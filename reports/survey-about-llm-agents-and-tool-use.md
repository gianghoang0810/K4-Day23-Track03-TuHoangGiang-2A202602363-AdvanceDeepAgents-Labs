# LLM Agents and Tool Use: A Survey of Methods, Benchmarks, Security and open problems

## TL;DR

- Tool use began as a way to make language models call external functions and APIs; ReAct interleaved reasoning and acting [1], and Toolformer showed that models can learn tool calls in a self-supervised way [2]. Later work moved toward large API sets and retrieval-aware calling [3][4].
- Training methods have split into supervised fine-tuning on synthetic tool traces [5][6] and reinforcement learning for multi-turn tool calling, where turn-level credit assignment and reward design are the main levers [7][8][9][10][11].
- Benchmarks show a reliability gap: on τ-bench, gpt-4o succeeded on fewer than 50% of tasks and on pass^8 below 25% in retail [12], and on OSWorld the best model reached 12.24% against 72.36% for humans at publication [13].
- Security is the most active recent concern. MCP tool poisoning, rug pulls and 31 attack methods have been documented [14][15][16], and skill files and injected tools can steer agents [17][18].
- Open problems include reliable multi-turn tool use, least-privilege tool selection, standard protocols for enterprise deployment, and evaluation that goes beyond task success [19][20][21].

## Background

A tool-using LLM agent is a language model that decides when to call external functions, APIs, search engines or computer interfaces, and then folds the results into its next steps. The ReAct framework formalised this as interleaved reasoning and acting steps, and its experiments used PaLM-540B and GPT-3 on HotpotQA and ALFWorld [1]. Toolformer showed that a model can learn which APIs to call, when, with what arguments, and how to use the results, using only a few demonstrations per API and a filter that keeps calls that reduce perplexity on future tokens [2]. A 2024 survey describes tool learning as four stages: task planning, tool selection, tool calling and response generation [22].

The topic matters now because tool use has become the main interface between models and software. The newer skills-based view treats agents as using reusable procedural capabilities that package knowledge, applicability conditions and execution policies, and it raises security and governance questions that pure tool use did not [23]. The Model Context Protocol (MCP) is the most common standard for connecting models to tools, and the A2A protocol is positioned as complementary, covering agent-to-agent rather than agent-to-tool communication [19][24].

Foundational work also includes Gorilla, a fine-tuned LLaMA model that uses a document retriever and retriever-aware training to write API calls, and which the authors report surpasses GPT-4 at writing API calls [3][25]. ToolLLM extended open-source models to more than 16,000 real-world APIs with a decision-tree search and a neural API retriever [4]. API-Bank provided a runnable benchmark with 73 API tools and 314 annotated tool-use dialogues [26].

## Training and Prompting Approaches

Approaches to teaching models tool use fall into three families that are often combined. The first is prompting and planning. ReAct and its successors keep the model frozen and structure its reasoning around actions [1]. Its strength is that it needs no training, so it can be applied to any prompted model [1].

The second family is supervised fine-tuning on synthetic tool traces. ToolACE generates diverse tool-learning data through a self-evolution synthesis process and claims stronger function calling with limited parameters [5]. ToolLoop extends this with closed-loop generation that uses iterative self-feedback across three stages, reported to improve accuracy while reducing data needs [6]. Toolformer is an early example of the same idea, where the training data is produced by the model itself [2]. The main strength of this family is cheap data; its weakness is that the quality of the synthetic traces caps performance.

The third family is reinforcement learning for multi-turn tool use. VerlTool offers a modular framework for agentic RL with tool use across domains [7]. The main design problem is credit assignment across turns: one approach assigns credit at the turn level [9], Turn-PPO reports that PPO is more robust than GRPO for long-horizon multi-turn RL [10], and RC-GRPO treats exploration as a steering problem through discrete reward tokens in low-reward-variance settings [11]. Reward design is also contested. CM2 replaces verifiable outcome rewards with checklist rewards in LLM-simulated environments for multi-turn, multi-step tool use [8]. The RL papers in our notes are mostly summary-level, so the comparison of their numbers is limited; the qualitative trade-off is that RL targets long-horizon behaviour that SFT on short traces does not capture, at the cost of reward design and simulation fidelity.

## Tool Retrieval, Selection and Interfaces

As tool catalogues grow, selecting the right tool becomes a retrieval problem. COLT proposes a completeness-oriented retriever that aims to select a diverse set of tools, using a dual-view graph framework [27]. Tool-DE argues that tools are under-documented and shows that simple document expansion boosts tool retrieval, with dedicated Tool-Embed and Tool-Rank models [28]. ScaleMCP makes MCP tool discovery dynamic and auto-synchronising, with a tool retriever and an embedding strategy [29]. The retrieval papers in our notes do not compare documentation quality against retriever architecture directly, so the relative importance of the two remains untested here.

MCP-style interfaces standardise how tools are described and invoked. MCP-Atlas evaluates tool-use competency with real MCP servers, using 36 servers, 220 tools and 1,000 multi-step tasks with claims-based scoring [30]. The 2026 MCP roadmap reports that the November 2025 specification remained the latest version, that stateful Streamable HTTP sessions conflict with load balancers, and that the Tasks primitive shipped as experimental with open gaps in retry and expiry semantics [19].

## Benchmarks and Evaluation

Benchmarks have moved from single-call accuracy to reliability over repeated trials and to end-to-end environments. τ-bench simulates a user and an agent with domain APIs and policy guidelines, judges the final database state, and introduces pass^k to measure consistency across trials. Its abstract reports that state-of-the-art function-calling agents succeed on fewer than 50% of tasks and that pass^8 is below 25% in retail; in the airline domain even gpt-4o solves only 35.2% of tasks [12]. Its failure analysis attributes most failures to agent errors such as wrong argument filling and hallucinated IDs [12].

The Berkeley Function Calling Leaderboard has evolved from simple AST-checked calls to a holistic agentic evaluation. In V4, agentic tasks account for 40% of the score and multi-turn for 30%, and the agentic part includes web search and memory tests [31]. GAIA takes a different angle, with 466 real-world questions that humans answer at 92% against 15% for GPT-4 with plugins at publication [32]. AgentBench spans eight interactive environments and 29 LLMs, and finds a significant gap between top commercial models and open-source models up to 70B, with poor long-term reasoning and instruction following as main failure causes [33].

Computer-use benchmarks show the largest gaps. OSWorld reports that humans complete over 72.36% of its 369 tasks while the best model reached 12.24% at publication, with GUI grounding and operational knowledge as the main failures [13]. MacArena extends the setting to 421 macOS tasks across 50 applications and reports that current GUI agents perform worse on native Apple Silicon environments than on Linux-based benchmarks [34]. The gap between benchmark numbers and the reliability needed in production is a recurring theme across these papers.

## Security and Reliability

Security has become the most intensively studied theme in tool-using agents. A survey of MCP security describes the server lifecycle in four phases and sixteen threat scenarios, including tool poisoning, installer spoofing and unauthorised access; it notes that tool poisoning and rug pulls are introduced at creation but triggered during operation [14]. A systematic review of 18 studies from 2024 and 2025 reports that half of the research focuses on adversarial attacks, while the benign ecosystem already shows structural deficits, and it argues that Zero Trust and guardrails address symptoms rather than architectural flaws [15]. MCPTox measures tool poisoning on real-world MCP servers and finds it widespread [35], and a separate analysis identifies 31 attack methods against tool-integration protocols [16].

Attacks also target the selection and instruction channels. ToolHijacker inserts malicious tools into the tool library to manipulate tool selection and is reported to beat existing attacks [18]. Skill-Inject finds that frontier models show up to 80% vulnerability to harmful instructions delivered through skill files [17]. Indirect prompt injection into LLM-integrated applications was already demonstrated in 2023 [36]. Defences are emerging: Progent enforces fine-grained tool-call policies through a domain-specific language [20]. Its authors claim security and utility across scenarios, but the claim rests on the paper summary alone.

Reliability problems are structural. A study of multi-agent LLM systems identifies 14 failure modes grouped into specification issues, inter-agent conflicts and task-verification problems [37]. A diagnostic framework for tool-invocation reliability proposes a 12-category error taxonomy and reliability thresholds for model selection in enterprise settings [38]. Agents also need to handle conflicting evidence: a 2026 benchmark measures whether agents revise, acknowledge uncertainty or escalate when retrieved evidence contradicts their beliefs, operationalised as Identify, Solve and Escalate [21].

## Trends and open problems

Over the last two years the field has shifted in four ways. First, RL has become the main route to multi-turn tool competence, with work on turn-level advantages, information-gain rewards and non-verifiable checklist rewards [8][9][10][39]. Second, evaluation has moved from single calls to reliability over trials and to agentic settings with web search, memory and computer use [12][31][13][34]. Third, MCP has become a de facto interface, with security analyses and a 2026 roadmap that names transport scalability, enterprise audit, SSO-integrated authentication and gateway behaviour as gaps [14][15][19]. Fourth, agents are being trained as foundation models for scientific and agentic workflows, where tool-mediated interactions are tied to verifiable outcomes [40][41].

Several problems remain unsolved or disputed:

- **Reliability over repeated runs.** Pass^k results show that single-run success hides inconsistency, and it is unclear whether RL or better data closes that gap [12][8].
- **Least privilege and tool-selection security.** Tool poisoning and tool-selection hijacking show that the choice of tool is itself an attack surface, and current defences are partial [20][18][15].
- **Protocol gaps.** The MCP roadmap lists unresolved issues in transport, retry semantics, audit trails and authorisation, and the security literature argues that these are architectural rather than patchable [19][15].
- **Multi-agent failures.** Failure taxonomies exist, but attribution of failures to specific agents and interventions that reliably fix them remain open [37][38].
- **Memory and long-horizon state.** Memory for agents is framed as a write-manage-read loop with open challenges in continual consolidation, trustworthy reflection, learned forgetting and privacy governance [42].
- **Evaluation validity.** Benchmark numbers from leaderboards are snapshots, and the reported results for many recent papers are from abstracts only, so cross-paper comparisons are weak [12][31].

Limitations of this survey: arXiv search was rate-limited during the research run, so most sources come from Hugging Face paper pages, web pages and summaries of abstracts. Numbers for recent (2026) methods are taken from summaries and have not been checked against full papers.

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. hf-search. https://huggingface.co/papers/2210.03629 (2022-10-06)
[2] Toolformer: Language Models Can Teach Themselves to Use Tools. web. https://arxiv.org/abs/2302.04761 (2023-02-09)
[3] Gorilla: Large Language Model Connected with Massive APIs. hf-search. https://huggingface.co/papers/2305.15334 (2023-05-24)
[4] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. hf-search. https://huggingface.co/papers/2307.16789 (2023-07-31)
[5] ToolACE: Winning the Points of LLM Function Calling. hf-search. https://huggingface.co/papers/2409.00920 (2024-09-02)
[6] ToolLoop: Closed-Loop Tool-Use Data Synthesis via Decomposed Generation and Dynamic Self-Feedback. hf-search. https://huggingface.co/papers/2609.09072 (2026-09-08)
[7] VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use. hf-search. https://huggingface.co/papers/2509.01055 (2025-09-01)
[8] CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use. hf-search. https://huggingface.co/papers/2602.12268 (2026-02-12)
[9] Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment. hf-search. https://huggingface.co/papers/2505.11821 (2025-05-17)
[10] Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs. hf-search. https://huggingface.co/papers/2512.17008 (2025-12-18)
[11] RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents. hf-search. https://huggingface.co/papers/2602.03025 (2026-02-03)
[12] τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. web. https://proceedings.iclr.cc/paper_files/paper/2025/file/1b126cc38b8638e07bef37e7b2bb72bf-Paper-Conference.pdf (2025)
[13] OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. arxiv. https://arxiv.org/abs/2404.07972 (2024-04-11)
[14] MCP Landscape, Security Threats, and Future Research Directions. web. https://dl.acm.org/doi/pdf/10.1145/3796519 (n.d.)
[15] The Model Context Protocol Security Landscape: A Systematic Analysis of Inherent Vulnerabilities and Defensive Inadequacies. web. https://link.springer.com/chapter/10.1007/978-3-032-31998-2_13 (2026-10-06)
[16] Systematic Analysis of MCP Security. hf-search. https://huggingface.co/papers/2508.12538 (2025-08-18)
[17] Skill-Inject: Measuring Agent Vulnerability to Skill File Attacks. hf-search. https://huggingface.co/papers/2602.20156 (2026-02-23)
[18] Prompt Injection Attack to Tool Selection in LLM Agents (ToolHijacker). hf-search. https://huggingface.co/papers/2504.19793 (2025-08-24)
[19] The 2026 MCP Roadmap. web. https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/ (2026-03-09)
[20] Progent: Programmable Privilege Control for LLM Agents. hf-search. https://huggingface.co/papers/2504.11703 (2025-04-16)
[21] Accurate but Not Humble: Evaluating Epistemic Humility in LLM Agents under Knowledge Conflict. hf-daily. https://huggingface.co/papers/2610.12360 (2026-10-08)
[22] Tool Learning with Large Language Models: A Survey. hf-search. https://huggingface.co/papers/2405.17935 (2024-05-28)
[23] SoK: Agentic Skills -- Beyond Tool Use in LLM Agents. hf-search. https://huggingface.co/papers/2602.20867 (2026-02-24)
[24] A2A Protocol (Agent2Agent). web. https://a2a-protocol.org/latest/ (n.d.)
[25] Gorilla (NeurIPS 2024 proceedings). web. https://proceedings.neurips.cc/paper_files/paper/2024/file/e4c61f578ff07830f5c37378dd3ecb0d-Paper-Conference.pdf (n.d.)
[26] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. web. https://aclanthology.org/2023.emnlp-main.187/ (2023)
[27] Towards Completeness-Oriented Tool Retrieval for Large Language Models (COLT). hf-search. https://huggingface.co/papers/2405.16089 (2024-05-25)
[28] Tools are under-documented: Simple Document Expansion Boosts Tool Retrieval. hf-search. https://huggingface.co/papers/2510.22670 (2025-10-26)
[29] ScaleMCP: Dynamic and Auto-Synchronizing Model Context Protocol Tools for LLM Agents. hf-search. https://huggingface.co/papers/2505.06416 (2025-05-09)
[30] MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers. hf-search. https://huggingface.co/papers/2602.00933 (2026-01-31)
[31] Berkeley Function Calling Leaderboard (BFCL V4: From Tool Use to Agentic Evaluation). web. https://gorilla.cs.berkeley.edu/leaderboard (2026-04-12)
[32] GAIA: a benchmark for General AI Assistants. web. https://proceedings.iclr.cc/paper_files/paper/2024/hash/25ae35b5b1738d80f1f03a8713e405ec-Abstract-Conference.html (2024)
[33] AgentBench: Evaluating LLMs as Agents. web. https://arxiv.org/abs/2308.03688 (2023-08)
[34] MacArena (macOS computer-use benchmark). hf-search. https://huggingface.co/papers/2606.06560 (2026-06-04)
[35] MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers. hf-search. https://huggingface.co/papers/2508.14925 (2025-08-19)
[36] Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. hf-search. https://huggingface.co/papers/2302.12173 (2023-02-23)
[37] Why Do Multi-Agent LLM Systems Fail?. hf-search. https://huggingface.co/papers/2503.13657 (2025-03-17)
[38] When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems. hf-search. https://huggingface.co/papers/2601.16280 (2026-01-22)
[39] Information Gain-based Policy Optimization (IGPO) for Multi-Turn LLM Agents. hf-search. https://huggingface.co/papers/2510.14967 (2025-10-16)
[40] Atria Dawn: The Dawn of Agentic Superintelligence. hf-daily. https://huggingface.co/papers/2609.15818 (2026-09-14)
[41] ZGCM-1: A Fully Open and Extremely Efficient Foundation Model for Math and Agentic Search. hf-daily. https://huggingface.co/papers/2609.13356 (2026-09-11)
[42] Memory for LLM agents: survey (write-manage-read loop). web. https://arxiv.org/pdf/2603.07670 (n.d.)
