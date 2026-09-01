# Design notes

## The tick

Six phases, in order. The expensive ones iterate *cells*, not organisms.

1. **Index.** Counting-sort both populations into grid cells, and accumulate the
   per-cell fields sensing reads: total plant biomass, edible plant biomass,
   prey mass, threat mass, animal count.
2. **Environment.** Recompute per-row temperature and light for this tick.
3. **Plants.** Photosynthesis, respiration, seeding intents, death.
4. **Animals.** Sense, think, move, feed, emit reproduction intents, die.
5. **Births.** Turn intents into organisms.
6. **Compact and census.** Drop the dead, recount species, gather statistics.

Every organism belongs to exactly one cell, so a cell owns its plants, its
animals and its patch of soil exclusively. That is what makes grazing and
predation free of read-modify-write races without any locking, and it leaves the
door open to running cells in parallel. Interaction reaches across a block of
cells rather than one — see below — and blocks are disjoint for the same reason.

## Sensing is field-based

At equilibrium density a 3×3 neighbourhood holds on the order of thirty
organisms. Having each animal walk its neighbours would cost tens of millions of
distance computations per tick and put a million organisms out of reach
entirely.

So sensing does not walk anything. The index pass accumulates a handful of
scalar fields per cell, and an animal reads those fields and their
central-difference gradients: a fixed ~27 array loads, regardless of how crowded
the world gets. Gradients are rotated into the animal's own heading frame before
they reach the network, so a brain learns "food is ahead" rather than having to
rediscover steering separately for each compass direction.

Every half-saturation constant in the model is a **density**, not a per-cell
amount. An earlier version used per-cell quantities, which meant changing the
grid resolution silently changed the ecology — a parameter that is supposed to
control only how finely the world is sampled.

## Interaction blocks

Sensing and interaction want opposite things from the grid. Gradients need fine
cells to point anywhere useful. Grazing and predation need cells with somebody in
them, and at equilibrium a sensing cell holds well under one animal — so
cell-local predation is effectively impossible and carnivores can never
establish.

Interaction blocks decouple them. Animals are updated a block at a time
(4×4 cells by default) and can reach anything inside their block, while still
sensing at cell resolution. Blocks tile the grid exactly and stay disjoint, so
the exclusivity that makes the phase race-free is preserved.

## Memory

Plants and animals live in separate struct-of-arrays pools. Their footprints
differ by an order of magnitude and most of the world is plants, so merging them
would drag every plant update through animal-sized cache lines for nothing.

| | bytes | at 1M organisms |
|---|---:|---:|
| plant | 26 | 18 MB at 700k |
| animal | 246 | 74 MB at 300k |
| grid fields and buckets | — | ~15 MB |

Brain weights dominate the animal record at 194 bytes. As `f32` a million
animals would spend 780 MB on weights alone, which is the difference between
fitting comfortably in a browser tab and not running at all. The `i8`
quantisation noise is far below what a mutating genome introduces anyway, and
the dequantisation scale factors out of the inner loop entirely: because the
accumulation is linear, `sum(w · s · x) == s · sum(w · x)`, so it is one
multiply per neuron rather than one per connection.

Removal is swap-remove. Stable compaction copies every survivor after the first
hole, which at a million organisms is a ~240 MB memcpy triggered by a single
death near index zero.

## Contingency

There is one RNG for the whole world, and draws are taken in update order.

An earlier version gave every organism its own stream keyed on its identity and
the tick, specifically so that the outcome did not depend on the order organisms
were updated in. That is a tidy property and a wrong one. Whether you or your
neighbour reaches the last plant first is not noise to be engineered away; it is
the contingency that decides which lineage persists, and removing it removes
something real. Seeds are an initial condition, not a replay, and runs default to
a seed drawn from system entropy.

Snapshots still carry the generator's state, because a save that omitted it
would be a lossy copy rather than a state save.

The float maths is still hand-rolled — `exp`, `ln`, `sin`, `cos` and `tanh` from
`+ - * /`, comparisons and bit casts only — but for speed and to avoid libm's
per-platform variation, not to promise reproducibility.

## Death

Mortality is Gompertz–Makeham: a constant age-independent hazard plus one that
rises exponentially with age. There is no maximum age.

A hard cutoff makes every animal immortal right up to a birthday, which is not
how anything dies, and it had a specific pathology. Reproduction used to be
gated on a neural output, so a lineage whose network always voted against
breeding was sterile *and* immortal at once: it could not die out, and it
produced no offspring for selection to act on, so nothing could remove it. Such
lineages sat at a few dozen individuals for thousands of ticks.

The fix for that is not a floor under reproduction — that is overriding
selection to get an outcome. It is that reproduction is physiological, driven by
condition with life-history genes setting the strategy, and that nothing is
immortal.

## Ecology

Matter cycles on a closed budget between soil, plant biomass and animal bodies.
Energy is separate: sunlight enters free and dissipates on death. Total biomass
is therefore capped by a physical stock rather than a tuned constant, and matter
drift is a correctness test rather than an ecological outcome.

Temperature varies with latitude and season. The gradient is the reason
speciation happens at all — a uniform world has one optimum, so every lineage
converges on it and the run becomes a beige monoculture.

### Environmental variance

Two AR(1) processes sit on top of the deterministic backbone: a global
temperature anomaly, and a *regional* productivity field interpolated from a
coarse grid. Both are reddened deliberately. White environmental noise averages
out over a lifetime and barely perturbs a population; autocorrelated noise
produces runs of bad years, and runs of bad years are what drive populations to
extinction. AR(1) is the standard minimal model for a reddened environment, and
scaling the innovation by `sqrt(1 - phi^2)` keeps the long-run spread fixed
instead of growing with the autocorrelation.

Productivity is regional rather than global because a world where every place
has a bad year simultaneously has no refuges, and refuges are where populations
actually persist. The coarse field is interpolated bilinearly and wrapped, so
droughts have soft edges rather than rectangular boundaries organisms could
visibly evolve against.

Disturbance — fire, storm, flood — kills a patch at random with severity falling
off toward the edge. It is a structuring force in most real ecosystems rather
than an interruption to one: it clears space, resets succession, and kills
without regard to fitness, which is a different selective regime from starvation
and predation.

### Diet is two genes, not a dial

There is no "diet" trait. There are two gut investments, one for plants and one
for flesh, and gut is tissue with upkeep proportional to how much of it you
carry. Specialisation emerges because a generalist pays for both.

The single-dial version needed a hand-picked curve to say how a half-carnivore
fared, and every choice of curve is an answer decided in advance. The one I
chose — linear plant digestion, concave meat digestion — put the midpoint of the
range at a constructed fitness minimum, which is exactly where uniform-random
founders start. Founding populations were therefore initialised at the worst
point on that axis for reasons that had nothing to do with biology.

### Organ costs are fractions of basal rate

Upkeep is `basal x kleiber x (1 + organ load)`, where vision, combat, guts and
slow ageing each contribute a fraction of basal. They used to be independent
additive terms of the same magnitude as basal itself, so an animal with middling
genes paid over two and a half times its basal rate in organ costs alone and
income never cleared upkeep by any margin. Expressing organ maintenance as a
fraction of BMR is also how physiologists measure it.

### Things that killed early versions

Nearly every problem was a modelling error, not a coding one. They are listed
because the current parameters only make sense against them.

**The temperature-generalist penalty was normalised against the minimum
tolerance**, with a square root, leaving a mid-range plant at 0.43 of peak
fitness. Photosynthesis could not outrun maintenance, plants never reached
seeding size, and the food web starved from the bottom up.

**Grazing had a type I functional response and could eat plants to death.**
Herbivores ate just as efficiently at low plant density as at high, stripped the
world bare and then starved — plant populations swung by an order of magnitude.
A saturating response plus a refuge the grazers cannot reach (real grazed plants
survive because the crown and roots are out of reach) damped it.

**That response keyed on total biomass including the refuge**, so a fully
cropped stand still read as abundant and intake never fell off before the food
was gone. Edible mass is now tracked as its own field.

**Per-capita intake did not fall with consumer density.** The animal population
had no intermediate equilibrium: growth was either positive until it hit the
hard cap or negative until extinction, and runs alternated between the two. A
Beddington–DeAngelis interference term gives it somewhere to settle.

**Animals drew the matter for their offspring from the soil beneath them.**
Plants sit at their nutrient-limited equilibrium and hold nearly all the matter,
so soil stays near zero and births fail however well-fed the animal is — and
fewer animals liberate less matter, permitting still fewer animals. A textbook
Allee trap. Animals now build offspring from what they have eaten.

**Reproduction was gated on a brain output.** A lineage whose network always
voted no became sterile *and* immortal at once: it could not die out, and it
produced no offspring for selection to act on, so nothing could ever fix it.
Such lineages sat at a few dozen individuals for thousands of ticks. The output
now modulates timing above a floor rather than holding a veto.

**Carnivory was unreachable.** Two separate causes. An aborted hunt cost the
animal its entire feeding turn, taxing every intermediate diet; hunts now fall
through to grazing. And meat digestion fell linearly to zero with diet, so at low
diet a hunt returned less than the grazing turn it cost and every step toward
predation was selected against. Meat digestion is now concave — which is also the
better biology, since living on plants needs machinery proportional to the
investment while flesh is nutritionally easy. Predators appearing brought
speciation with them: animal species went from a single lineage to more than
twenty.

**Founders were seeded with a little carnivory**, and the concave curve makes
even a nominally herbivorous founder a 52%-efficient carnivore. With no
established plants to graze, the founding cohort ate itself — every early death
was a kill, three quarters of the animals gone before the first birth. Founders
are strict herbivores now, and their ages are spread across their lifespans
rather than all starting at zero, which had left several hundred ticks in which
the population could only shrink.

## Diploidy and sex

Two alleles per locus, stored interleaved so a locus is one cache line.
Expression is the allele mean; because decoding is linear in the byte, averaging
the alleles then decoding equals decoding both then averaging, so expression
costs one add and a shift. Gametes take one allele per locus independently — no
linkage, which is the right simplification for unlinked quantitative traits.

Mate search walks outward ring by ring from where the animal stands, bounded by a
cell budget. It deliberately is *not* confined to the interaction block: the block
exists so feeding writes stay disjoint, and finding a mate only reads the
partner, so it can range further. It must, too — at equilibrium density a block
usually holds nobody else, and confining the search to one made every animal
population fail for want of a mate regardless of how healthy it was.

### Things adding sex broke, and why

**Animals had no directional sense of their own kind.** The sensory inputs
included crowding as a scalar but no conspecific gradient, so nothing requiring
an animal to *approach* another could evolve. Harmless under clonal
reproduction; fatal with sex.

**Indexing a diploid genome by a gene constant reads the wrong locus.** A genome
is now twice as wide, so `child[ag::SIZE]` is a raw byte in the middle of another
gene. It survived compilation and every type check, and showed up only as a slow
matter leak in the conservation test — which is exactly the sort of thing that
invariant is for.

**Founders arrived starving.** Zero reserve meant the founding cohort spent its
whole breeding window accumulating matter and thinned past the density at which
mates can be found. A fed adult in breeding condition carries stores worth more
than one offspring; anything less is not "in condition".

**Too many founding populations fragment the propagule.** Separate stocks are
reproductively isolated from the first tick, so two dozen of them means two dozen
populations each below the density at which mates can be found. One is no better:
the whole run then turns on a single random genotype. A few is right.

## The WebAssembly boundary

Raw `extern "C"`, no wasm-bindgen. The interface is entirely numeric, so
generated glue buys nothing while its toolchain and its version coupling to the
compiler cost plenty.

Two hazards are worth naming. Growing WebAssembly memory allocates a new
`ArrayBuffer` and **detaches every typed-array view over the old one** — and a
detached view does not throw on read, it reports zero length, so a stale view
shows up as an empty world rather than an error. Every accessor in `borscht.js`
re-creates its views when the buffer identity changes. And a view into
WebAssembly memory cannot be transferred to another thread, so the worker copies
each frame into a plain `ArrayBuffer` before posting it, recycling two buffers so
the steady state does not allocate.

`web/params.js` is generated from `config.rs`, so adding a parameter is a
one-file change and the browser table cannot drift from the Rust definitions.
