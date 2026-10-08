"""Learnable compositional embeddings of PDDL operators, trained end to end with the policy.

Two steps, in two places:

1. Serializing the operator into the observation (structure.py). StructuredOperatorEncoder writes
   the operator as a fixed-size *structure array*, because Stable-Baselines3 observations must be
   fixed-size numeric arrays. PolicyObservationWrapper computes it once per operator and puts it in
   obs["operator_embedding"], where the other conditions put their fixed embeddings.
2. Composing the embedding in the policy network (extractor.py). CompositionalOperatorExtractor, the
   first module of the policy, unpacks the structure array and composes learnable component embeddings
   (components.py) into one operator embedding of size --operator-embedding-dim, appended to the other
   observation parts exactly like every other condition. Being part of the policy, every embedding and
   function is trained by the RL losses.

Components (components.py; each embedded in R^component_dim):
    type            every object type (e.g. block) has an embedding
    variable        every parameter *position* has an embedding: variables are identified by position,
                    not by name, so renaming ?o to ?x changes nothing and variable i lines up with
                    grounding slot i (docs/policy_inputs.md)
    typed variable  TYPED(variable, type): the type acts on the variable through a learned function
    predicate       every predicate the environment can evaluate (e.g. on-table) has an embedding
    literal         LITERAL(predicate, argument_1, ..., argument_A): the predicate applied to its
                    typed-variable arguments in argument order, so (on ?a ?b) differs from (on ?b ?a)
    name            the operator name's fixed text embedding (--text-backend), through a learned map

Architectures, i.e. how the components are composed (--compositional-architecture):
    tree       tree.py: learned functions following the PDDL syntax
                   NOT(literal)                       negated preconditions and delete effects
                   AND({literals})                    conjunction: a DeepSets sum, order-invariant
                   PRE(AND(preconditions)), EFF(AND(effects)), PARAMETERS({typed variables})
                   OPERATOR(NAME, PARAMETERS, PRE, EFF) -> the operator embedding
    slots      slots.py: no hand-written composition. Tokens for the name, typed variables and literals
               (each plus an embedding of its kind) are grouped by slot attention (Locatello et al.,
               2020) into learned slots, flattened into the operator embedding
    geometric  geometric.py: the components laid out on a grid, read by a CNN and slot attention
"""
