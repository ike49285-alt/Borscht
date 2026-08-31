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

## Determinism

`borscht-core` has no dependencies. The RNG is a hand-rolled PCG32 whose bit
stream is pinned by test, and each organism draws from a stream derived from its
own id and the tick, so update order cannot affect the outcome.

The float maths is hand-rolled for the same reason. `exp`, `ln`, `sin`, `cos`
and `tanh` are built from `+ - * /`, comparisons and bit casts only — all
exactly specified by IEEE 754 and implemented identically by x86 SSE and the
wasm32 float instructions. Calling `f32::sin` would hand the result to a platform
libm and quietly break the guarantee that a browser run and a CLI run are
bit-identical. That guarantee is what makes snapshots portable between them.

## Ecology

Matter cycles on a closed budget between soil, plant biomass and animal bodies.
Energy is separate: sunlight enters free and dissipates on death. Total biomass
is therefore capped by a physical stock rather than a tuned constant, and matter
drift is a correctness test rather than an ecological outcome.

Temperature varies with latitude and season. The gradient is the reason
speciation happens at all — a uniform world has one optimum, so every lineage
converges on it and the run becomes a beige monoculture.

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
