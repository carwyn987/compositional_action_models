# Related work

Where this project sits in the literature, per research question: the closest prior work, and what this
project does that it does not. The research questions are in `docs/research_focus.md`; full references are at
the end.

## The field in one paragraph

Five lines of work meet here and rarely cite each other. **Multi-task RL** conditions one policy on a task
identifier, almost always one-hot (Meta-World, soft modularization, PaCo, MOORE); CARE is the exception and
uses natural-language task metadata. **Specification-conditioned policies** take a structured task
description as input — an LTL formula (LTL2Action, DeepLTL), a reward machine, a program, a predicate goal
(DECSTR), or an instruction (BC-Z) — and generalize to unseen specifications. **Neuro-symbolic skill
learning** for task and motion planning uses symbolic operators to define skills (LEAGUE, Silver et al. 2022,
Ada), but the operator shapes reward, state abstraction or termination; it is not an input to the policy, and
each operator usually gets its own policy. **Planning-model repair** (Gragera et al., Lin & Bercher, model
reconciliation, SPOTTER) edits the symbolic model so plans succeed, and learns new executors from scratch when
needed. **Transfer machinery** — UVFAs, successor features, hindsight relabeling, hypernetworks, probes —
provides the tools.

This project conditions one continuous-control policy on the structure of a PDDL action model
(preconditions, add and delete effects, typed parameters bound to grounding slots) and asks whether that
structure buys sample efficiency, transfer to new skills, and repair when the action model is edited.
No prior work found does this; each neighbouring line stops one step short. Established nearby: structured,
entity- and goal-conditioned policies beat MLPs in manipulation (Zhou et al. 2024), symbolic operators can be
coupled to learned skills (Silver et al. 2022; Garrett et al. 2021), and hypernetworks generate controllers for
unseen task parameters (Rezaei-Shoshtari et al. 2023; Beck et al. 2022). Not established: a zero-shot mapping
from unseen combinations of preconditions and effects to the physical behaviour they require.

## RQ1: operator structure vs arbitrary identifiers

**Closest work.** CARE (Sodhani et al. 2021) encodes each task's natural-language description with a frozen
language model and uses it to compose shared encoders on Meta-World; it compares with and without metadata,
not against one-hot or random identifiers. BC-Z (Jang et al. 2021) finds one-hot and language conditioning
perform about the same on training tasks, with language helping only on unseen tasks. LIV (Ma et al. 2023)
finds language task specifications worse than one-hot on MetaWorld, because pretrained text embeddings alias
similar descriptions. LEAGUE (Cheng & Xu 2023) and Silver et al. (2022) learn one policy per operator, with
operator effects as reward and operator parameters as state abstraction. Ada (Wong et al. 2024) learns a
shared controller conditioned on subgoal predicate conjunctions (effects only) in discrete domains. DECSTR
(Akakzia et al. 2021) conditions a Fetch block-manipulation policy on binary predicate goal vectors.
Entity-structured policies (Deep Sets, graphs, self-attention) learn simulated manipulation with less data than
MLPs (Zhou et al. 2024), but their conditioning is entities and subgoals, not an action model. Policy sketches
reuse named subpolicies across tasks (Andreas et al. 2017), and compositional language abstractions beat
non-compositional ones (Jiang et al. 2019).

**This project.** The conditioning input is the full action model, including preconditions and delete
effects, and success is checked against its semantics (effects settle and the STRIPS frame holds). The
conditions form a controlled ladder on one architecture, budget and seed set: one-hot, random, learnable,
text embedding of the operator, multi-hot over literals, and learnable compositional encoders. The prior
results above predict small differences on training skills; RQ1 is the baseline that RQ2 and RQ3 are
measured against. One-hot and multi-hot are different priors, not equal-capacity embeddings: a learned layer
on multi-hot is an additive lookup over literal features, while one on a random code is a learned map from an
arbitrary identifier.

## RQ2: learning a new skill faster, or zero-shot

**Closest work.** Policies conditioned on compositional specifications generalize to unseen ones built from
seen parts: LTL formulas (Vaezipoor et al. 2021; Kuo et al. 2020; Jackermeier & Abate 2025), logical task
compositions (Nangue Tasse et al. 2020, 2024), parameterized instructions with analogy-making (Oh et al.
2017), programs (Sun et al. 2020). UVFAs and successor features generalize across goals and reward weights
(Schaul et al. 2015; Barreto et al. 2017; Borsa et al. 2019). Modular and compositional architectures transfer
to unseen task combinations in manipulation (Devin et al. 2017; Mendez et al. 2022a, 2022b). The multi-hot
literal encoding is an attribute representation in the sense of attribute-based zero-shot learning (Lampert
et al. 2009). Structured policies also generalize to more objects and new combinations of familiar
subtasks (Zhou et al. 2024). None of these shows generalization to new literals or unseen predicates: held-out
combinations of seen literals are the clean test, and an unseen predicate is beyond what a learned predicate
lookup can support.

**This project.** The specification is an action model rather than a goal, formula or instruction: it
carries the context in which the skill applies (preconditions) and what must not change (delete effects,
frame), and it describes continuous manipulation with physical failure conditions. The gain from structure
is measured as steps to threshold on a held-out skill against identifier conditions that can only help
through shared parameters.

## RQ3: repairing a skill when its operator changes

**Closest work.** Planning-model repair finds the edit to preconditions or effects that makes a task
solvable or a plan valid (Gragera et al. 2023; Lin & Bercher 2021); model reconciliation explains plans by
model edits (Chakraborti et al. 2017; Sreedharan et al. 2019). SPOTTER (Sarathy et al. 2021), RAPid-Learn
(Goel et al. 2022) and Thierauf & Scheutz (2024) learn new operators or executors with RL when the symbolic
model fails. Specification-conditioned policies can be re-targeted by a new specification, and language
corrections can steer a robot's behaviour (Sharma et al. 2022; Shi et al. 2024). LLM-generated PDDL models
need correction after the fact (Guan et al. 2023). In task and motion planning, symbolic operators are lossy
abstractions of continuous transitions (Garrett et al. 2021). Successor features transfer to a changed reward
under fixed dynamics (Barreto et al. 2017), but an edited precondition or effect is not a change of reward
weights.

**This project.** The missing direction: given an edited action model, does the edit reach the trained
low-level skill through its conditioning, zero-shot or with fewer steps? Repair pairs differ by one or two
literals (`unstack` → `unstack-raised`, `push-left` → `push-left-front`). Controls separate "the structure
carried the edit" from "the task was easy": repair-hidden (the policy sees the original operator), identifier
conditions, and the invalid-repair check (the original behaviour must fail the repaired operator); the
comparison with learning the repaired operator from scratch gives the transfer gain. A repair that only corrects
the model (e.g. a precondition that changes when the skill applies, not how it moves) should leave behaviour
unchanged, so only repairs that need physically different behaviour can show the edit reaching the skill.

## RQ4: generating policies from operator embeddings

**Closest work.** FiLM (Perez et al. 2018) is a standard multi-task conditioning baseline and BC-Z's
conditioning mechanism. Hypernetworks generate weights from a task embedding (Ha et al. 2017; von Oswald et
al. 2020), from MDP parameters for zero-shot transfer (Rezaei-Shoshtari et al. 2023), from task
specifications (HyPoGen, Ren et al. 2025), and from text or instructions (TeNet, Bighashdel & Luck 2026; DISC,
Ren et al. 2026). DISC argues that a generated policy cannot shortcut the instruction through the
observation. Initialization strongly affects how reliably RL hypernetworks train (Beck et al. 2022).

**This project.** Only the conditioning source (an action model) and the comparison against individually
trained per-skill policies are specific to it. FiLM and hypernetwork conditioning are best treated as
ablations of how the operator embedding enters the policy in RQ1–RQ3, not as a contribution of their own.
Generated and individually trained policies are compared by their actions on shared probe states and their
returns, not their weights, which are not identifiable under hidden-unit permutations.

## RQ5: do learned embeddings encode operator structure?

**Closest work.** Linear probes (Alain & Bengio 2017) with control tasks for selectivity (Hewitt & Liang
2019; Belinkov 2022); amnesic probing, which separates information present in a representation from
information used, by intervention (Elazar et al. 2021); probes for human concepts in an RL agent (McGrath et al. 2022); analyses of learned
task and skill embedding spaces (von Oswald et al. 2020; Hausman et al. 2018). Konidaris et al. (2018) learn
symbols from skills.

**This project.** Probing a compositional embedding for the literals it was built from is trivially
positive. The informative probe targets embeddings that never saw literals — the learnable codes of the
identifier conditions — and asks whether RL on behaviour alone recovers the action model, with leave-one-
operator-out probes and permutation baselines.

## RQ6: behavioural sensitivity as a diagnostic of literals

**Closest work.** Perturbation and gradient attribution for policies and networks (Greydanus et al. 2018;
Sundararajan et al. 2017); action-model learning from executions (Stern & Juba 2017; Aineto et al. 2019);
operator learning that models only the effects needed for planning (Kumar et al. 2023); model repair (RQ3).

**This project.** The operator-conditioned policy is used as an instrument to audit its own action model:
remove or alter one literal and measure the change in behaviour on fixed probe states. Sensitivity shows what
the policy uses, not what the world requires, so the method is evaluated on injected faults with known
ground truth (an implied precondition, a wrong effect). Following Elazar et al. (2021), sensitivity is measured
by intervening on the symbolic input with the observation and grounding fixed, and compared with irrelevant
edits: RQ5 asks whether operator structure is recoverable, RQ6 whether it is used.

## Operator-level hindsight relabeling

**Closest work.** Relabeling experience for goals or tasks other than the one pursued is standard: Kaelbling
(1993), HER (Andrychowicz et al. 2017), SAC-X (Riedmiller et al. 2018), generalized hindsight (Li et al.
2020), HIPI (Eysenbach et al. 2020), hindsight task relabelling (Packer et al. 2021), counterfactual
experiences for reward machines (Toro Icarte et al. 2022), hindsight instruction relabeling (Jiang et al.
2019), and relabeling over predicate conjunctions (Wong et al. 2024).

**This project.** Operator relabeling (`docs/next_steps.md`) counts an episode as a success for another
operator when that operator's preconditions held at the start and its effects and the frame held at the
end. It is an operator-specific form of a known technique: a sample-efficiency lever, not a contribution in
itself.

## Evaluation practice

Two standards from the literature apply to every comparison above. Few-seed RL comparisons need per-task
results with stratified bootstrap confidence intervals (Agarwal et al. 2021). And a conditioned policy may
ignore its conditioning and identify the task from the observation (shortcut learning, Geirhos et al. 2020;
the observation leakage that motivates DISC), which the wrong-embedding control in `docs/next_steps.md`
tests: a policy that identifies the skill from the state or the grounding does not change behaviour when
given another skill's operator. For training, SAC (Haarnoja et al. 2018) with hindsight relabeling
(Andrychowicz et al. 2017) is the strong baseline for sparse, goal-relabelable Fetch tasks, though not evidence
that SAC beats PPO on every skill.

## References

- Agarwal, Schwarzer, Castro, Courville, Bellemare. Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021. https://arxiv.org/abs/2108.13264
- Aineto, Jiménez, Onaindia. Learning action models with minimal observability. Artificial Intelligence 275, 2019. https://doi.org/10.1016/j.artint.2019.05.003
- Akakzia, Colas, Oudeyer, Chetouani, Sigaud. Grounding Language to Autonomously-Acquired Skills via Goal Generation (DECSTR). ICLR 2021. https://arxiv.org/abs/2006.07185
- Alain, Bengio. Understanding intermediate layers using linear classifier probes. ICLR 2017 workshop. https://arxiv.org/abs/1610.01644
- Andreas, Klein, Levine. Modular Multitask Reinforcement Learning with Policy Sketches. ICML 2017. https://proceedings.mlr.press/v70/andreas17a.html
- Andrychowicz et al. Hindsight Experience Replay. NeurIPS 2017. https://arxiv.org/abs/1707.01495
- Barreto et al. Successor Features for Transfer in Reinforcement Learning. NeurIPS 2017. https://arxiv.org/abs/1606.05312
- Beck, Jackson, Vuorio, Whiteson. Hypernetworks in Meta-Reinforcement Learning. CoRL 2022. https://proceedings.mlr.press/v205/beck23a.html
- Belinkov. Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics 48(1), 2022. https://doi.org/10.1162/coli_a_00422
- Bighashdel, Luck. TeNet: Text-to-Network for Compact Policy Synthesis. arXiv 2026. https://arxiv.org/abs/2601.15912
- Borsa et al. Universal Successor Features Approximators. ICLR 2019. https://arxiv.org/abs/1812.07626
- Chakraborti, Sreedharan, Zhang, Kambhampati. Plan Explanations as Model Reconciliation. IJCAI 2017. https://www.ijcai.org/proceedings/2017/23
- Cheng, Xu. LEAGUE: Guided Skill Learning and Abstraction for Long-Horizon Manipulation. RA-L 2023. https://arxiv.org/abs/2210.12631
- Devin, Gupta, Darrell, Abbeel, Levine. Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer. ICRA 2017. https://doi.org/10.1109/ICRA.2017.7989250
- Elazar, Ravfogel, Jacovi, Goldberg. Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals. TACL 9, 2021. https://doi.org/10.1162/tacl_a_00359
- Eysenbach, Geng, Levine, Salakhutdinov. Rewriting History with Inverse RL: Hindsight Inference for Policy Improvement. NeurIPS 2020. https://arxiv.org/abs/2002.11089
- Garrett, Chitnis, Holladay, Kim, Silver, Kaelbling, Lozano-Pérez. Integrated Task and Motion Planning. Annual Review of Control, Robotics, and Autonomous Systems 4, 2021. https://arxiv.org/abs/2010.01083
- Geirhos et al. Shortcut learning in deep neural networks. Nature Machine Intelligence 2020. https://doi.org/10.1038/s42256-020-00257-z
- Goel, Shukla, Sarathy, Scheutz, Sinapov. RAPid-Learn: A Framework for Learning to Recover for Handling Novelties in Open-World Environments. ICDL 2022. https://arxiv.org/abs/2206.12493
- Gragera, Fuentetaja, García-Olaya, Fernández. A Planning Approach to Repair Domains with Incomplete Action Effects. ICAPS 2023. https://ojs.aaai.org/index.php/ICAPS/article/view/27190
- Greydanus, Koul, Dodge, Fern. Visualizing and Understanding Atari Agents. ICML 2018. https://proceedings.mlr.press/v80/greydanus18a.html
- Guan, Valmeekam, Sreedharan, Kambhampati. Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning. NeurIPS 2023. https://arxiv.org/abs/2305.14909
- Ha, Dai, Le. HyperNetworks. ICLR 2017. https://arxiv.org/abs/1609.09106
- Haarnoja, Zhou, Abbeel, Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML 2018. https://proceedings.mlr.press/v80/haarnoja18b.html
- Hausman, Springenberg, Wang, Heess, Riedmiller. Learning an Embedding Space for Transferable Robot Skills. ICLR 2018. https://mlanthology.org/iclr/2018/hausman2018iclr-learning
- Hendawy, Peters, D'Eramo. Multi-Task Reinforcement Learning with Mixture of Orthogonal Experts (MOORE). ICLR 2024. https://arxiv.org/abs/2311.11385
- Hewitt, Liang. Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP 2019. https://doi.org/10.18653/v1/D19-1275
- Jackermeier, Abate. DeepLTL: Learning to Efficiently Satisfy Complex LTL Specifications for Multi-Task RL. ICLR 2025. https://arxiv.org/abs/2410.04631
- Jang et al. BC-Z: Zero-Shot Task Generalization with Robotic Imitation Learning. CoRL 2021. https://proceedings.mlr.press/v164/jang22a.html
- Jiang, Gu, Murphy, Finn. Language as an Abstraction for Hierarchical Deep Reinforcement Learning. NeurIPS 2019. https://proceedings.nips.cc/paper_files/paper/2019/file/0af787945872196b42c9f73ead2565c8-Paper.pdf
- Kaelbling. Learning to Achieve Goals. IJCAI 1993. https://mlanthology.org/ijcai/1993/kaelbling1993ijcai-learning
- Konidaris, Kaelbling, Lozano-Pérez. From Skills to Symbols: Learning Symbolic Representations for Abstract High-Level Planning. JAIR 61, 2018. https://jair.org/index.php/jair/article/view/11175
- Kumar, McClinton, Chitnis, Silver, Lozano-Pérez, Kaelbling. Learning Efficient Abstract Planning Models that Choose What to Predict. CoRL 2023. https://proceedings.mlr.press/v229/kumar23a.html
- Kuo, Katz, Barbu. Encoding formulas as deep networks: Reinforcement learning for zero-shot execution of LTL formulas. IROS 2020. https://arxiv.org/abs/2006.01110
- Lampert, Nickisch, Harmeling. Learning to detect unseen object classes by between-class attribute transfer. CVPR 2009. https://doi.org/10.1109/CVPR.2009.5206594
- Li, Pinto, Abbeel. Generalized Hindsight for Reinforcement Learning. NeurIPS 2020. https://arxiv.org/abs/2002.11708
- Lin, Bercher. Change the World — How Hard Can that Be? On the Computational Complexity of Fixing Planning Models. IJCAI 2021. https://www.ijcai.org/proceedings/2021/571
- Ma et al. LIV: Language-Image Representations and Rewards for Robotic Control. ICML 2023. https://arxiv.org/abs/2306.00958
- McGrath et al. Acquisition of chess knowledge in AlphaZero. PNAS 2022. https://doi.org/10.1073/pnas.2206625119
- Mendez, Hussing, Gummadi, Eaton. CompoSuite: A Compositional Reinforcement Learning Benchmark. CoLLAs 2022 (2022a). https://proceedings.mlr.press/v199/mendez22a.html
- Mendez, van Seijen, Eaton. Modular Lifelong Reinforcement Learning via Neural Composition. ICLR 2022 (2022b). https://arxiv.org/abs/2207.00429
- Nangue Tasse, James, Rosman. A Boolean Task Algebra for Reinforcement Learning. NeurIPS 2020. https://arxiv.org/abs/2001.01394
- Nangue Tasse, Jarvis, James, Rosman. Skill Machines: Temporal Logic Skill Composition in Reinforcement Learning. ICLR 2024. https://arxiv.org/abs/2205.12532
- Oh, Singh, Lee, Kohli. Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning. ICML 2017. https://proceedings.mlr.press/v70/oh17a.html
- Packer, Abbeel, Gonzalez. Hindsight Task Relabelling: Experience Replay for Sparse Reward Meta-RL. NeurIPS 2021. https://arxiv.org/abs/2112.00901
- Perez, Strub, de Vries, Dumoulin, Courville. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018. https://arxiv.org/abs/1709.07871
- Ren et al. HyPoGen: Optimization-Biased Hypernetworks for Generalizable Policy Generation. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/afe52e5ddaa88badea593b306df483d9-Abstract-Conference.html
- Ren, Zhou, Zhou, Yang. DISC: Decoupling Instruction from State-Conditioned Control via Policy Generation. arXiv 2026. https://arxiv.org/abs/2605.20856
- Rezaei-Shoshtari, Morissette, Hogan, Dudek, Meger. Hypernetworks for Zero-Shot Transfer in Reinforcement Learning. AAAI 2023. https://doi.org/10.1609/aaai.v37i8.26146
- Riedmiller et al. Learning by Playing — Solving Sparse Reward Tasks from Scratch (SAC-X). ICML 2018. https://proceedings.mlr.press/v80/riedmiller18a.html
- Sarathy, Kasenberg, Goel, Sinapov, Scheutz. SPOTTER: Extending Symbolic Planning Operators through Targeted Reinforcement Learning. AAMAS 2021. https://arxiv.org/abs/2012.13037
- Schaul, Horgan, Gregor, Silver. Universal Value Function Approximators. ICML 2015. https://proceedings.mlr.press/v37/schaul15.html
- Sharma et al. Correcting Robot Plans with Natural Language Feedback. RSS 2022. https://roboticsproceedings.org/rss18/p065.html
- Shi et al. Yell At Your Robot: Improving On-the-Fly from Language Corrections. RSS 2024. https://roboticsproceedings.org/rss20/p025.html
- Silver, Athalye, Tenenbaum, Lozano-Pérez, Kaelbling. Learning Neuro-Symbolic Skills for Bilevel Planning. CoRL 2022. https://proceedings.mlr.press/v205/silver23a.html
- Sodhani, Zhang, Pineau. Multi-Task Reinforcement Learning with Context-based Representations (CARE). ICML 2021. https://proceedings.mlr.press/v139/sodhani21a.html
- Sreedharan, Olmo, Mishra, Kambhampati. Model-Free Model Reconciliation. IJCAI 2019. https://doi.org/10.24963/ijcai.2019/83
- Stern, Juba. Efficient, Safe, and Probably Approximately Complete Learning of Action Models. IJCAI 2017. https://www.ijcai.org/proceedings/2017/615
- Sun, Wu, Lim. Program Guided Agent. ICLR 2020. https://iclr.cc/virtual/2020/poster/1662
- Sun, Zhang, Xu, Tomizuka. PaCo: Parameter-Compositional Multi-Task Reinforcement Learning. NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/86b8ad667206fb9a52ae575fbf1cd6be-Abstract.html
- Sundararajan, Taly, Yan. Axiomatic Attribution for Deep Networks. ICML 2017. https://proceedings.mlr.press/v70/sundararajan17a.html
- Thierauf, Scheutz. Fixing symbolic plans with reinforcement learning in object-based action spaces. IROS 2024. https://doi.org/10.1109/IROS58592.2024.10801362
- Toro Icarte, Klassen, Valenzano, McIlraith. Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning. JAIR 73, 2022. https://arxiv.org/abs/2010.03950
- Vaezipoor, Li, Toro Icarte, McIlraith. LTL2Action: Generalizing LTL Instructions for Multi-Task RL. ICML 2021. https://proceedings.mlr.press/v139/vaezipoor21a.html
- von Oswald, Henning, Grewe, Sacramento. Continual learning with hypernetworks. ICLR 2020. https://arxiv.org/abs/1906.00695
- Wong, Mao, Sharma, Siegel, Feng, Korneev, Tenenbaum, Andreas. Learning adaptive planning representations with natural language guidance (Ada). ICLR 2024. https://arxiv.org/abs/2312.08566
- Yang, Xu, Wu, Wang. Multi-Task Reinforcement Learning with Soft Modularization. NeurIPS 2020. https://papers.nips.cc/paper/2020/hash/32cfdce9631d8c7906e8e9d6e68b514b-Abstract.html
- Yu et al. Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning. CoRL 2019. https://arxiv.org/abs/1910.10897
- Zhou, Kumar, Finn, Rajeswaran. Policy Architectures for Compositional Generalization in Control. Reinforcement Learning Journal 5, 2024 (RLC 2024). https://rlj.cs.umass.edu/2024/papers/Paper327.html
