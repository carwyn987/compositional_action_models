# Embedding explorer (RQ5)

`scripts/explore_embeddings.py` shows what the operator embeddings contain, for RQ5 (`docs/research_focus.md`: do
learned embeddings encode the operator's structure?). It takes any embedding condition, fixed or learned, and the
parts the embedding is built from, projects them to 2-D or 3-D in an interactive chart, and compares them in tables
on the full vectors. It is separate from `main.py` and trains nothing.

```bash
./scripts/explore_embeddings.py                                        # one-hot, random, multi-hot, text (mock)
./scripts/explore_embeddings.py --run outputs/<run> --level all --dims 2 3
./scripts/explore_embeddings.py --run outputs/<run> --serve            # the app, at http://127.0.0.1:8765
./scripts/explore_embeddings.py --static multi-hot compositional-tree --text-backend openai --offline
```

The code: `src/cam/experiments/embedding_points.py` (sources, levels, points), `embedding_geometry.py` (projections
and comparisons), the script (tables, chart, workspace, app) and `scripts/explore_embeddings.html` (the page).

## Sources

A source is one embedding function, built the way training builds it (`OPERATOR_ENCODERS` from a config), so a run
is explored exactly as its policy saw its operators.

| source | what it is | its control |
| --- | --- | --- |
| `--static one-hot`, `random` | identifiers: no structure | (they are the control) |
| `--static multi-hot` | the literal features, nothing learned | (structure by construction) |
| `--static text` | the PDDL or a paragraph through a text model (`--text-backend`, `--operator-text`) | multi-hot |
| `--static compositional-tree`, `-slots`, `-geometric` | an untrained composition (`--seed`) | its other seeds |
| `--run DIR` with `--trainable-operator-embedding` | W x: the learned matrix on the encoder's vector | level `input`: x |
| `--run DIR`, compositional | the trained composition | untrained twins |
| `--run DIR` with a fixed encoder | that encoder, with the run's settings | |

A run's points come from its `model.zip` and `config.json` (defaults filled in as `main.py` does on resume; a stale
run whose policy reads another embedding size is refused). `--extractor` picks the features extractor: the actor's
by default; `critic`, `critic-target` or `all`. With `main.py`'s SAC settings the critic and the actor share one
extractor module (shown once, as the actor); the target critic has its own copy, which follows by Polyak averaging.
For PPO, actor is the policy's extractor (pi) and critic the value function's (vf), one module when shared.

**Untrained twins.** A composition encodes some structure before any training: operators that share literals share
inputs. So "the learned embedding of pickup is near pickup-raised" says little on its own; the question is whether
it is nearer than in the same architecture untrained. Each compositional source gets twins (`--untrained-seeds`,
default 0 1 2): its policy rebuilt with the same settings from a fresh seed. The tables give their mean ± std next
to each number. A trainable run's control is its input x (level `input`): W starts at the identity, so
`operator-change`, (W - I) x, is what training added. PPO's orthogonal initialization replaces the identity at the
start (W = sqrt(2) Q), so a PPO run has no `operator-change` level and x is not where it started; the experiments
use multi-task SAC.

**The operator name.** The name is an input to one-hot and random embeddings, to text ones with
`--operator-text-name include` and to compositional ones with `--compositional-name text`. Runs follow their config.
Static text and compositional sources leave the name out, as the experiments do, unless `--operator-name include`
(with `--static-config RUN`, the run's setting).

**Text without the API.** Text embeddings from OpenAI are cached on disk (`--text-embedding-cache`; relative paths
are the repository's). `--offline` never calls the API: a text missing from the cache is listed as not encodable.
The mock backend needs neither.

## Levels

`--level` picks what to plot (default `operator`; `all` for every level a source has). Each level is a kind of
vector; two levels of a source share a **space** only if they are inputs to the same function or one is a linear
combination of the other, and only points in one space are projected together, compared or added up. The same size
is not enough: the tree's `pre` and `eff` are both 32 numbers but come out of different functions.

| level | sources | size | space | points |
| --- | --- | --- | --- | --- |
| operator | all | 256 | operator | one per operator: what the policy sees |
| literal | multi-hot, text, trainable | 256 | operator | each literal as its one-literal operator |
| feature | multi-hot, trainable multi-hot | 256 | operator | each literal feature (multi-hot's basis; W's columns) |
| predicate (text) | text | 256 | operator | each predicate's name as a text |
| input, operator-change | trainable | 256 | operator | x and (W - I) x (operator-change: SAC only) |
| type, variable, typed-variable, predicate | compositional | 32 | one each | the component embeddings; TYPED for 3 positions x types |
| atom | compositional | 32 | literal | LITERAL(predicate, arguments): the same in any section and sign |
| name | compositional with the name | 32 | name | the name's text embedding through its learned map |
| signed-literal | tree | 32 | literal | NOT(atom) for negated preconditions and delete effects |
| conjunct | tree | 32 | conjunct-sum | phi(signed literal), the term AND sums |
| and-sum | tree | 32 | conjunct-sum | each section's sum of conjuncts (`<op> pre`, `<op> eff`) |
| and | tree | 32 | and | AND: rho of the sum |
| pre, eff, parameters | tree | 32 | one each | PRE(AND(...)), EFF(AND(...)), PARAMETERS(typed variables) |
| token | slots | 32 | literal | each component token plus its kind: slot attention's inputs |
| slot, slots | slots, geometric | 32, 128 | slot, slots | each of the 4 slots, and the 4 flattened |
| operator-raw | compositional | 256 | operator-raw | the composition's output before normalization |

(Sizes for the default `--component-embedding-dim 32`, `--operator-embedding-dim 256`, `--slots 4`.) Components are
collected once each from the operators, with variables by parameter position, as every encoder sees them, and named
as in the first operator that has them. `--all-atoms` adds every predicate on every ordering of parameter positions
(37 atoms for Fetch). `--probe literals` adds each literal as a one-literal operator to the operator level, which
for a policy is an operator it never saw.

## The chart

Each source and space is a panel with its own projection: compare points within a panel, never coordinates across
panels. PCA (default) is fitted on the registered operators (`--fit-on registered`; or the trained ones, or every
point) and every other point is placed in it, so adding a custom point never moves the rest. The title gives each
component's share of the variance; a point's hover gives its residual, the share of its distance from the mean the
plane does not show (a point far off the plane can look close to others). `--unit-norm` projects directions only.
t-SNE (`--method tsne`) keeps neighbourhoods rather than distances, needs a few dozen points (it warns below 20 and
shows PCA below 5) and refits every point, so no position or distance between clusters means anything by itself;
never read an analogy from it.

Colour is the operator's family (grasp: pickup, unstack and their raised variants; place: putdown, stack,
place-beside; push) or, for components, what their predicate is about (region: at-left/right/front/back; relation:
on, beside; object state: the rest). Each colour has its own marker shape. Filled markers are trained operators,
open ones held out (runs only), diamonds custom points, stars expressions, crosses probes. Clicking pair, analogy
or x -> W x in the legend draws a line for each pair, arrows for each analogy's two directions, and arrows from a
trainable run's x to W x (with `input` drawn). Every point is labelled (the app can turn labels off) and its hover
gives its level, source, the operators it belongs to and its PDDL. The page follows the system's light or dark mode.

## The tables

Printed, saved as `tables.txt` and shown under the chart. All use the full vectors, within one source and level.

- **pairs** (`--pair A B`): cosine; its percentile among every pair of registered operators (how unusual this
  similarity is for this source); B's rank among A's nearest and A's among B's (ties share their mean rank, so for
  one-hot every rank is the middle); the distance; the untrained twins' cosine; and the structural reference, the
  operators' literal Jaccard and multi-hot cosine. `pairs.csv` adds each distance relative to the median distance.
  The default pairs ask: is pickup near pickup-raised (one effect differs) and near unstack (the same grasp from
  another block)? unstack near unstack-raised, the two raised variants near each other, a push near its diagonal,
  and putdown and stack near place-beside (all release the block).
- **analogies** (`--analogy A B C D`, A:B::C:D): the cosine between B - A and D - C, their length ratio, and D's
  rank among the points nearest C + B - A. The defaults: pickup:pickup-raised::unstack:unstack-raised (the same
  edit, delete raised -> add raised, on two operators), the two diagonal pushes per direction,
  pickup:putdown::unstack:stack (grasp to place; pickup -> putdown points the opposite way to stack -> unstack), and
  at-left:at-right::at-front:at-back at the predicate levels. At linear levels the first is exact by construction
  (multi-hot, W x of multi-hot, the tree's and-sum), which makes them the controls: the operator levels after the
  composition's nonlinear functions are where an analogy is learned, or not.
- **structure agreement**: Spearman correlation between the operators' pairwise cosines and their literal Jaccard
  (`pre` and `eff` levels: precondition and effect Jaccard), i.e. representational similarity analysis. One-hot is
  n/a (every cosine is 0), random about 0, multi-hot high; a learned embedding above its untrained twins has learned
  structure, below them has reduced towards an identity.
- **between levels**: the same correlation between two levels of a source, e.g. how much of `and-sum`'s similarity
  survives to `operator`.
- **custom and expression points**: the three nearest registered points and the distance to the nearest.
- **not encodable**: what a source could not encode, and why.

## Custom points and expressions

Custom points are PDDL operators in the workspace (`--workspace`, default `outputs/embedding_explorer/workspace.json`;
`--add LABEL=PDDL` or `LABEL=@file.pddl`, `--remove LABEL`, or the app's form, which can start from a registered
operator's PDDL). The workspace keeps the PDDL, not vectors: every source embeds it afresh. Invalid PDDL is refused
with the parser's message and not saved. The label is separate from the operator's name; to isolate a change of
content, keep the original's name (otherwise one-hot cannot encode it, and random, text and compositional-with-name
move for the name alone). A custom copy of a registered operator lands exactly on it, also with its literals
reordered and variables renamed, except in text, which embeds the PDDL as written.

Expressions (`--expression LABEL="pickup - eff (not (raised ?o)) + eff (raised ?o)"`) add up operator labels and
literals (`pre` or `eff`, then the literal; spaces around + and -). Each literal is its one-literal operator on the
first operator's name and parameters, so they apply where literals share the operators' space: multi-hot (exact:
the example is pickup-raised), W x of multi-hot (exact), text (empirical). Compositional sources refuse them (their
literals are in another space): there, add the edited operator as a custom point, which the composition embeds
directly.

## Files and the app

`--out` (default `outputs/embedding_explorer/<time>/`) gets `embeddings.html` (self-contained, with plotly.js;
`--plotlyjs cdn` for a small file needing the network; `--dims 2 3` adds a 3-D tab), `points.csv`, `pairs.csv`,
`vectors.npz` (one array per source and level, and its labels), `projection.json` (each panel's PCA mean, components
and explained variance) and `tables.txt`.

`--serve` runs the same as an app on 127.0.0.1 (`--port`, default 8765; `--no-browser`): choose sources, levels,
PCA or t-SNE, 2-D or 3-D, add and remove custom points, edit the pairs and analogies, and export. Sources load once;
every chart is computed by the same Python as the export. From another machine, forward the port:
`ssh -L 8765:127.0.0.1:8765 <machine>`. The JSON API: `GET /api/options`; `POST /api/chart` with `{sources, levels,
method, dims, fit_on, unit_norm, labels}`; `POST /api/points` with `{label, pddl, note}`; `DELETE /api/points/<label>`;
`POST /api/pairs`, `POST /api/analogies` (the whole list); `POST /api/export`.
