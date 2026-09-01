# Borscht

An evolution simulator. Plants and animals share a world with a closed nutrient
budget, a climate that varies with latitude and season, and no scripted
behaviour at all. Everything you see — where things live, what they eat, whether
predators exist — is the outcome of selection on mutable genomes.

The default world holds about a hundred thousand organisms; it scales up to a
million if you want one. It runs in the browser and headless from the command
line for long experiments.

```
tools/build-web.sh
python3 -m http.server -d web 8080     # then open http://localhost:8080
```

## What actually evolves

Each organism carries a genome of bytes. Every byte maps onto a trait, mutates
when it is copied, and — this is the part that matters — **costs something**.

Animals have sixteen loci: body size, top speed, sensory reach, two independent
gut investments, attack, defence, maturity age, offspring investment, mutation
rate, senescence rate, temperature preference and tolerance, colour, energy
storage, and breeding threshold — each carrying two alleles. Plants have eight: growth rate, maximum size,
seed dispersal range, seed investment, toxicity, temperature preference and
tolerance, and colour.

Diet is not a dial. There are two genes — one for digesting plants, one for
digesting flesh — and a gut is tissue you pay upkeep on. Specialisation emerges
because carrying both guts means paying for both. A single "diet" gene needs a
hand-picked curve to say how a half-carnivore fares, and whatever curve you
choose is the answer you wanted rather than one the model produced.

A gene with only upside is not an evolutionary pressure — the population pins it
to the maximum and stops being interesting. So size raises attack and storage
but also metabolism, following Kleiber's `mass^0.75`. Speed costs energy with
the square of velocity. Vision, weapons, guts and slow ageing are charged as
*fractions of basal metabolic rate*, which is how organ maintenance is actually
measured. A wide temperature tolerance lowers the peak, so a generalist never
beats a specialist on the specialist's home ground.

Animals also carry a small neural network — fourteen senses, ten hidden units,
three actions — whose weights recombine and mutate alongside the genes. It sees local plant,
prey and predator density and their gradients, its own energy and age, crowding,
temperature mismatch, and an internal oscillator; it decides how to turn, how
hard to move, whether to feed, and whether to breed. Gradients are rotated into
the animal's own frame of reference, so a brain learns "food is ahead" rather
than having to rediscover steering separately for every compass direction.

Founders are drawn at random and get no help. Their diets are whatever they are,
which means a founding cohort can and does eat itself; they begin with no store
of matter and have to eat before they can build anything. Reproduction is
physiological rather than a decision — an animal that is mature, fed and
carrying enough matter breeds — because a neural veto on breeding is not how
organisms work, and it produces a lineage that is fit in every other respect but
never reproduces, and so never produces the offspring selection would need to
remove it.

Nothing dies on a birthday either. Mortality is Gompertz–Makeham: a constant
hazard plus one rising exponentially with age. Nothing is immortal, so a lineage
that fails is gone within a few lifetimes.

## Sex

Organisms are diploid. Every locus carries two alleles, expression is their mean
(additive gene action, the standard model for quantitative traits), and
reproduction goes through gametes: one allele per locus, assorted independently,
fused into a zygote. Animal brains recombine by uniform crossover between the
parents. Heterozygosity is tracked, and it falls over a run — drift removes
alleles and only mutation puts them back.

Animals must **find a mate**, and there is no selfing fallback: two animals breed
only if their genetic distance is below a threshold, so reproductive isolation —
not a label in a registry — is what makes a species a species. Plants outcross
where they can and self when they cannot, which is what mixed mating systems do.

Sex is expensive, and the model shows it. Clonal reproduction gave 4–5 of 6
worlds a persistent animal population; with sex it is closer to 2–3 of 6. That is
not a defect to be tuned away — the two-fold cost of sex and Allee effects in
small sexual populations are exactly what the literature describes. Adding sex
also exposed a real gap: animals could sense *crowding* but had no directional
sense of their own kind, so mate-seeking and breeding aggregation could not
evolve at all. They now have a conspecific gradient.

## The environment is not a backdrop

Temperature and light vary with latitude and season, and on top of that sit two
stochastic processes. Both are AR(1), because environmental variance only
matters if it is **autocorrelated**: white noise averages out within a lifetime
and barely perturbs a population, while reddened noise produces runs of bad
years, and runs of bad years are what actually drive populations to extinction.

Productivity varies *regionally* rather than globally, so a drought leaves
refuges — and refuges are where populations survive bad years. Temperature has a
global anomaly on top of the seasonal cycle. Disturbance — fire, storm, flood —
clears patches at random, killing without regard to fitness, which is a
different selective regime from starvation and predation.

## Matter is conserved, energy is not

The world runs on a fixed stock of matter that cycles between soil, plant
biomass and animal bodies. Sunlight enters for free, flows up the food chain and
dissipates when anything dies.

This asymmetry is the main thing keeping the ecosystem from either collapsing or
exploding: total biomass is capped by a physical budget rather than by a tuned
constant. It is also a sharp correctness test. Matter is checked every tick in
the test suite, and any drift means a bug in one of the transfer paths rather
than an interesting ecological outcome.

Animals build their offspring out of what they have eaten. That sounds
equivalent to drawing from the soil under a closed budget, but it is not: plants
sit at their nutrient-limited equilibrium and hold nearly all the matter, so
soil stays pinned near zero and births fail however well-fed the animal is.

The **biomass** control moves that budget. It is an intervention rather than a
parameter — a step change in how much stuff there is to go round, of the kind an
experimenter makes and an ecosystem does not. Taking matter out works up from the
bottom of the food web: the soil first, because in this model the soil *is* the
dead-organism pool and everything that dies goes to the soil where it fell; then
the standing crop, thinned evenly; then animal reserves, which are fat and can be
given up; and only then whole animals, because an animal cannot be partly
removed. Adding matter puts it back into the soil in proportion to what is
already there, so the spatial structure that makes some places worth being in
survives the intervention.

Conservation is not relaxed for the control. Every gram moved is recorded, and
the invariant becomes an equality against founding stock plus that ledger — so an
intervention still cannot hide a leak.

## Layout

```
crates/borscht-core/    the simulation: no dependencies, no I/O
crates/borscht-wasm/    raw extern "C" surface for the browser
crates/borscht-cli/     headless runner, benchmarks, PNG and CSV output
web/                    the viewer: worker, WebGL2 renderer, UI
tools/                  build script, benchmarks, browser check
```

`borscht-core` has no dependencies at all. The RNG is a hand-rolled PCG32 and
`exp`, `ln`, `sin`, `cos` and `tanh` are built from IEEE-exact primitives rather
than handed to a platform libm — they are faster than libm here and avoid its
per-platform variation, but reproducibility is not promised.

There is one RNG for the whole world and draws are taken in update order, so
**who reaches the last plant first depends on iteration order**. That is
deliberate. An earlier version gave each organism its own stream keyed on its id
and the tick, which made outcomes independent of update order — a property no
real ecosystem has, and one that quietly removes exactly the contingency that
decides which lineage persists. Seeds are an initial condition, not a replay:
runs default to a seed drawn from system entropy.

The browser build uses no wasm-bindgen and no wasm-pack. Everything crossing the
boundary is a number or a pointer into linear memory, so the whole binding is
one 200-line JavaScript file, and building needs `rustup target add
wasm32-unknown-unknown` and nothing else.

## Command line

```
borscht run     [--population N] [--ticks N] [--seed N] [--out DIR]
                [--frames N] [--image-size N] [--color MODE] [--set key=value]
borscht bench   [--population N]... [--ticks N]
borscht params  [--json]
```

`--population` scales the world at constant density, so the ecology behaves the
same at every size and only the map changes; omit it for the default world. `--out` writes `stats.csv` with 27
per-tick measurements plus PNG frames. Any of the 53 parameters can be
overridden with `--set`; `borscht params` lists them with ranges and
descriptions.

```
borscht run --ticks 20000 --out runs/one --frames 40
borscht run --population 400000 --ticks 20000 --set temp_stress=2.0 --quiet
```

## Performance

Measured on four cores. WebAssembly figures come from running the same module
the browser runs, under Node.

The world is sized for a target population, but the population it actually
carries is an ecological outcome, so both columns are given. Living organisms is
the number that sets the cost.

| target | living organisms | native ms/tick | wasm ms/tick |
|--------|-----------------:|---------------:|-------------:|
| 10,000 | 2,400            | —              | 0.24         |
| 25,000 | 5,900            | 0.3            | 0.51         |
| 60,000 | 14,100           | —              | 1.28         |
| 83,000 | 17,700           | 1.2            | —            |

Cost is linear in the living population at roughly 65 ns per organism natively
and 90 ns in WebAssembly. Every scale the viewer offers runs faster than display
rate, so the timeline speed control rather than the machine is what decides how
fast the world advances. Diploidy and sex account for much of the per-organism cost — the genome
doubled, the brain widened to take a conspecific sense, and mate search walks
outward until it finds someone. Rendering is decoupled from that: the simulation runs in
a worker and the page draws at display rate from whatever frame is current, so
panning and zooming stay smooth no matter how slowly the world is advancing.

Three things make that possible. Animals sense through per-cell aggregate fields
rather than walking their neighbours, which keeps cost flat at ~27 loads per
animal however crowded the world gets. Brain weights are `i8`, so a million
animals cost 194 MB of weights instead of 780 MB. And brains run on a stagger,
each animal thinking every fourth tick on its own offset, while movement
integrates every tick.

## Watching it

The viewer has two views. **World** draws every organism as a point, coloured by
species, diet, energy, age or size. **Tree of life** draws the phylogeny: one
line per lineage from the tick it split off to the tick it died out, with a
connector back to the lineage it came from. Colour is inherited with drift, so
relatives sit near each other in hue and a radiation reads as a block.

Extinct branches are most of the tree, and keeping them is what makes it a
history rather than a snapshot. They also cost something: registry slots are
recycled the moment a species dies, so lineages are recorded separately in a
history that outlives the slot.

That history is bounded, and how it is bounded matters. It used to stop recording
once it was full, which is first-come-first-kept: it preserved the deep past and
threw away the recent radiation, so in a long run most of the *living* lineages
were missing from their own phylogeny. It now prunes instead. Nothing alive is
ever evicted, and neither is anything that is an ancestor of something alive —
that is the backbone, and losing it shatters the tree rather than thinning it.
What goes is the smallest, longest-dead leaves, which is exactly what the
viewer's minimum-peak control already hides. Survivors whose parent was pruned
are re-pointed at their nearest surviving recorded ancestor. Whatever is lost is
counted and shown, so the tree says it is incomplete instead of implying it is
the whole record.

The timeline **rewinds**. The worker keeps periodic snapshots in a ring bounded
by bytes rather than by count — a snapshot scales with the population, so a
fixed count would quietly eat a gigabyte in a large world — and seeking loads
the newest checkpoint at or before the target and re-ticks forward. Because a
snapshot carries the generator state, replaying reproduces the run you already
watched rather than a new one.

### One file, no server

`node tools/build-artifact.mjs` writes the whole viewer as a single HTML file:
the WebAssembly module embedded as base64, and the engine running on the main
thread rather than in a Worker. The worker existed for million-organism worlds;
at the sizes the viewer now offers a tick is a fraction of a frame, so there is
nothing left to decouple.

It is generated from `web/`, never hand-maintained, and every substitution in
the builder asserts that it matched — a silent no-op replacement produces a page
that looks right and quietly runs the wrong world. `node tools/check-web.mjs
--bundle` runs the same browser assertions against the bundle as against the
multi-file app, and `node tools/watch-bundle.mjs` answers the separate question
of whether the result is worth watching.

## Small worlds die

Establishment depends sharply on how big the world is. Six seeds each, 6,000
ticks, counting a world as established if it still has more than 100 animals:

| organisms | established | animals in survivors | animal species |
|-----------|------------:|---------------------:|----------------|
| 2,500     | 1 / 6       | 156                  | 1              |
| 10,000    | 0 / 6       | —                    | —              |
| 25,000    | 1 / 6       | 2,762                | 63             |
| 60,000    | 3 / 6       | 902 – 6,373          | 8 – 109        |
| 120,000   | 5 / 6       | —                    | up to 43       |
| 300,000   | 5 / 6       | —                    | 123 – 692      |

This is minimum-viable-population behaviour and it fell out of the model rather
than being put in. Read the top of that table before drawing conclusions from a
small run: **a default world usually loses its animals**, and every one of the
scales the viewer offers is at or below the threshold where a sexual population
reliably holds on. That is a result, not a fault — but it does mean the small
scales are for watching mechanism and the large ones for watching an ecosystem.
Most of this project's debugging happened at 40,000 organisms, where extinction
dominates and it is easy to mistake an island too small to hold a sexual
population for a bug.

The plants are unaffected: they can self, so a plant flora establishes and
persists at every scale. What the small worlds cannot support is animals that
need to find a compatible mate.

## Tests

```
cargo test                                        # unit tests
cargo test --release --test ecology -- --nocapture # invariants, and a census
node tools/bench-wasm.mjs                          # wasm throughput
node tools/check-web.mjs                           # drives the real page
```

The unit tests check the machinery. The ecology gate checks what must be true
*whatever happens*: matter is conserved in a thriving world and a dead one
alike, state stays physically meaningful, selection narrows the founding
variation, and populations are limited by ecology rather than by the memory cap.

One test is a genuine validation rather than an invariant:
`establishment_rises_with_propagule_size` checks that more founders means a
better chance of establishing, which is the strongest single predictor of
invasion success in the literature and was not tuned for. It caught a real
fault — founders had been small mutations around a few templates, so a larger
propagule brought more individuals but no more genotypes, and establishment was
flat across a sixty-fold range.

It deliberately does **not** assert that populations survive. An earlier version
did, and that assertion was a tuning target dressed as a test — it passed only
because the model had been fitted until it did. Ecosystems collapse;
colonisation by a handful of random genotypes usually fails, and a model in
which it never does is describing its author rather than ecology. Outcomes are
measured and printed by `census_across_seeds`, not required. As of writing, five
of six worlds still have animals after 6,000 ticks and one does not.

`tools/check-web.mjs` drives the real page in headless Chromium and decodes the
screenshot to confirm organisms were actually drawn.

## Design notes

`docs/design.md` covers the tick pipeline, the memory layout, and the ecological
modelling — including the several ways early versions died, which is most of
what the design is shaped by.
