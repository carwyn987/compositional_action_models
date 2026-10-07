# RQ : Experiment : Metric triples

Can a neural network effectively map embedded symbolic models to low-level policies, and does this generalize to unseen symbolic model embeddings (such as those from new skills or repairs)?


## RQ1 Does conditioning a shared multi-skill policy on symbolic action-model representations (compositional encodings) improve sample efficiency or final performance relative to arbitrary skill identifiers?

Dimensions
 - Symbolic model embedding type (aka high level structural skill information provided)

Experiment
 - Train the same shared policy on skillset; Compare metrics across symbolic model embedding types: one hot / random hash, text embedding, compositional embedding, per skill & aggregate (careful, easy skills can dominate).
    uninformative:
    - one-hot skill ID
    - fixed random hash/vector
    - learnable vector
    symbolic:
    - whole-string/text embedding
    - compositional embedding (learnable)
    - compositional embeddings (not learnable)
 - Ensure same shared policy architecture, same skillset, same sampling schedule, same RL algorithm, same training budget, same seeds

Metrics
 - Steps-to-success threshold
 - Learning AUC
 - Final success rate

Interpretation
 - Asks - is there information we can glean from symbolic action models in order to improve training speed of low-level partner policies
 - In both cases, the shared policy network can learn from all skills being trained. The only difference is that in one, symbolic models are provided, with the intent to be performative
 - If the step-to-success speed of the symbolic model is higher, then we conclude that the symbols effectively improved learning speed (not necessarily due to generalization, but joint learning helps fs)
 - If the final success rate is higher in symb model, then we conclude that training with symb model embeddings leads to the development of shared pathways (...?) which improves performance?
 - A comparison between the learnable vector and compositional embedding answers
  - If the compositional / symbolic stucturing of learnable embeddings, we can say that the symbolic representations / relationships are meaningful priors (?)

## RQ2 How (much) does leveraging symbolic embeddings for shared policy networks speed up next-policy learning (zero-shot)?

Experiment
Compare the following over text/whole-model embedding vs aggregate compositional embedding vs slot compositional embedding
 - Repeat the following
  - Shuffle skills
   - Train in order, appending each next skill to the training set to ensure we don't catestrophically forget any learned skills. e.g. for each "training run": [1], [1,2], [1,2,3], ....
   - Measure steps-to-success on new learned skill ; consider a success threshold as the end signal
 - Aggregate results to get statistically significant

Interpretation
 - Measures how much the shared policy network generalizes to new skills - including zero-shot-learning potential

## RQ3 Does leveraging symbolic embeddings as conditioners enable on-the-fly domain repair?

Experiment
 - Learn a bunch of skills. Repair one of them (or just change an effect, e.g.)
 - Test zero shot performance
 - Train this skill until success threshold and measure steps-to-success
 - Compare against one hot / random hash embedding type

### Together, RQ2,RQ3 answer the impact on learning - both repair, and new skills

## RQ4 Does leveraging symbolic action models to condition individual policy networks (via hypernetworks) to perform multiple actions outperform or provide useful initializations over training individual policy networks from scratch?

Experiment
 - Build pipeline: operator → embedding → hypernetwork → policy parameters. 
 - Compare generated policies against individually trained policies.

Metrics
 - steps-to-success

# Discussion-based RQS

## RQ5. Does the learned representation actually encode meaningful operator structure?

Experiment: Build (autoregressive) autoencoder (?) may not be possible with our amount of data, so probes may have to suffice
See if we can actually regenerate the original PDDL symbolic action model
 - If so, this implies we can vectorize action models
 - Is this meaningful? We have already showed we know how to do this with hardcoded ohe/mhe methods, assuming arity / predicate limits...

## RQ6. Can behavioral sensitivity help identify unnecessary or incorrect symbolic model components?

Experiment
 - Remove individual preconditions/effects, regenerate/recondition policy, evaluate behavior; then test candidate simplified models.
 - This is more for my own curiousity - although I'm very curious to see what happens ...
 - Will similar symbolic action models with significantly different behavior be close?
 - Why would this let us identify incorrect symbolic model components? Like unneccessary preconditions?

# NOT TO BE PURSUED (FUTURE WORK POTENTIAL?)

## RQN (Optional) Does leveraging symbolic action models to condition a single policy network to perform multiple actions outperform individual policy networks?

## RQN+1. Does concurrent multi-skill training actually improve stability/transfer versus sequential skill training?

# Dimensions

Symbolic model embedding type (aka high level structural skill information provided)
 - One hot / random hash --> text embedding --> compositional embedding --> per skill

Domain action model format (? Declarative vs Procedural)
 - PDDL --> Refinement (operational) Action Models
 - How do other world model types fit in? E.g. could we use individually trained networks as "model" specs?
 - What about FSM and other world model types?