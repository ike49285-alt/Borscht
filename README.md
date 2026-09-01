# Borscht

A mass battle simulator: two armies, hundreds of thousands of men each, every
one of them an individual body with a position, a facing, health and nerve.

It grew out of an evolution simulator, and kept the parts that turned out to be
about *scale* rather than about ecology: struct-of-arrays pools, a counting-sort
spatial grid, sensing through per-cell fields rather than neighbour lists,
quantised neural networks, a raw `extern "C"` WebAssembly boundary with no
bindgen, and an instanced renderer that draws bodies with a heading rather than
points. The ecology itself is in the git history.

## Where it is

Two armies muster, march at each other, fight, and one of them breaks. Behaviour
is still hard-coded -- walk toward the enemy strength you can sense, keep out of
your own side's crush, hit whatever comes into reach -- so nothing manoeuvres
yet. Commanders and doctrine are next.

## Morale is what makes it a battle

Without it two lines stand and hack until one is annihilated, which is not how
mass engagements are decided. Nerve is a **rate**, not a lookup: it moves a
little each tick according to a man's circumstances rather than being recomputed
from them, so it has memory and cannot flicker. Everything it reads is a
per-cell field the grid already accumulated, so it costs a handful of array
reads per man and nothing walks a neighbour list.

A man is steadied by getting the better of a fight and by having formed men at
his shoulder; he is shaken by casualties near him, by his own wounds, and above
all by friends running past. That last term is the carrier: one broken company
frightens the next, which is the whole phenomenon. Below his archetype's nerve
he breaks, stops fighting, and runs. A blow that lands on a man who is running
does far more damage, because he cannot turn and defend himself.

Rallying needs three things at once, and each was added because leaving it out
produced a specific wrong behaviour: nerve past a margin above the threshold,
no enemy in the cell he is standing in, and more formed men than fugitives where
he has stopped. Men rally on a colour party, not in the middle of a stampede.

The measured shape, at twenty thousand men, five seeds:

| | |
|---|---|
| line holds for | 420 – 480 ticks |
| 10% of survivors running | tick ~430 |
| half of them | tick ~520 |
| nine in ten | tick ~580 |
| cut down fighting | ~50% |
| cut down running | ~50% |

The collapse accelerates as it spreads, which is what a real one does, and the
pursuit accounts for about half the dead, which is roughly what history says.

### Four ways this was wrong first

Every one of them came out of the rout trace rather than from looking at it.

**Shock read a raw count, not a share.** The casualty field is a decaying sum,
so at a decay of 0.92 it settles at twelve and a half times the local death
rate; one man dying per tick drove the shock term to 0.69 a tick and both armies
were annihilated in nerve within seconds of contact. What matters is the
*fraction* of the men around you who are falling: a company of six that loses
two is shattered, a mass of five hundred that loses two has not noticed.

**A fugitive was rewarded for running.** Local odds were `own / (own + enemy)`,
which reads 1.0 -- a crushing victory -- when there is no enemy nearby at all. A
man who had simply run out of contact was therefore being steadied for it. The
term is gated on the enemy actually being present now.

**A broken man was charged for the other fugitives around him.** A routing mob
is almost entirely routing, so the panic term pinned his nerve at zero and
rallying was arithmetically impossible.

**And then rallying became automatic.** Fixing the two above let men re-form in
the open the instant they were clear, break again seconds later, and repeat
forever: an army of ten thousand logged three hundred thousand breaks and half a
million rallies. It takes a delay before a man will re-form at all, and somebody
formed to re-form *on*.

## The measured ceiling

Both sides together, on one core, including the phase where the armies are
actually in contact:

| men | native ms/tick | native ticks/s | wasm ms/tick | wasm ticks/s |
|-----|---------------:|---------------:|-------------:|-------------:|
| 20,000 | 2.1 | 484 | 1.8 | 572 |
| 100,000 | 9.5 | 105 | 9.0 | 111 |
| 500,000 | 39 | 26 | 46 | 22 |
| 1,000,000 | 104 | 10 | 132 | 8 |

A million men run, and they are not a slideshow by accident: at eight ticks a
second plus thirty milliseconds of frame packing, that is about six frames a
second in a browser. **500,000 is the practical ceiling for something worth
watching**, and 100,000 leaves real headroom for the morale and command work
that has not been done yet. The viewer offers all of them and says which is
which.

Two measurements were worth the trouble of taking:

**The first benchmark lied.** It ran two hundred ticks from deployment, which at
these distances is entirely marching. Coarsening the grid looked like a clean
win, because coarse cells make the per-cell fields smaller. Timing a *whole
battle* reversed it completely -- 128 cells a side took 333 seconds against 80
for 512 -- because target selection scans a neighbourhood, and a neighbourhood
holding four times as many men costs four times as much.

**Compaction was clearing every target in the army, every tick.** Swap-remove
moves a unit's index, so compacting has to drop every stored target; and in a
battle somebody dies almost every tick. Every man in contact was therefore
re-scanning his surroundings continuously to find the enemy he was already
fighting. Bodies now wait on the field until there are enough to be worth
clearing, which halved the cost of a battle.

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

| organisms | established | animals in survivors |
|-----------|------------:|---------------------:|
| 5,000     | 1 / 6       | 112                  |
| 10,000    | 2 / 6       | 121 – 164            |
| 15,000    | 2 / 6       | 483 – 654            |
| 25,000    | 3 / 6       | 123 – 1,841          |
| 120,000   | 5 / 6       | —                    |
| 300,000   | 5 / 6       | —                    |

This is minimum-viable-population behaviour and it fell out of the model rather
than being put in. Read it before drawing conclusions from a small run: at these
sizes **a world is as likely to lose its animals as to keep them**, and the
scales the viewer offers all sit below the threshold where a sexual population
reliably holds on. That is a result, not a fault.

The floor did come down, though, and by mechanism rather than tuning. Making
turning cost speed, making speciation a population event, and raising metabolism
to the point where an animal cannot live off one patch together took 10,000
organisms from zero worlds in six to two, and 25,000 from one to three. Most of
this project's debugging happened at 40,000 organisms, where extinction dominates
and it is easy to mistake an island too small to hold a sexual population for a
bug.

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
