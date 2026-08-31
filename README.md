# Borscht

An evolution simulator. Plants and animals share a world with a closed nutrient
budget, a climate that varies with latitude and season, and no scripted
behaviour at all. Everything you see — where things live, what they eat, whether
predators exist — is the outcome of selection on mutable genomes.

It runs in the browser at up to a million organisms, and headless from the
command line for long experiments.

```
tools/build-web.sh
python3 -m http.server -d web 8080     # then open http://localhost:8080
```

## What actually evolves

Each organism carries a genome of bytes. Every byte maps onto a trait, mutates
when it is copied, and — this is the part that matters — **costs something**.

Animals have sixteen genes: body size, top speed, sensory reach, diet, attack,
defence, maturity age, offspring investment, mutation rate, lifespan,
temperature preference and tolerance, colour, energy storage, breeding
threshold, and aggression. Plants have eight: growth rate, maximum size, seed
dispersal range, seed investment, toxicity, temperature preference and
tolerance, and colour.

A gene with only upside is not an evolutionary pressure — the population pins it
to the maximum and stops being interesting. So size raises attack and storage
but also metabolism, following Kleiber's `mass^0.75`. Speed costs energy with
the square of velocity. Vision, weapons, armour and long life are all charged
for as metabolic upkeep. A wide temperature tolerance lowers the peak, so a
generalist never beats a specialist on the specialist's home ground.

Animals also carry a small neural network — fourteen senses, ten hidden units,
four actions — whose weights mutate alongside the genes. It sees local plant,
prey and predator density and their gradients, its own energy and age, crowding,
temperature mismatch, and an internal oscillator; it decides how to turn, how
hard to move, whether to feed, and whether to breed. Gradients are rotated into
the animal's own frame of reference, so a brain learns "food is ahead" rather
than having to rediscover steering separately for every compass direction.

Founding animals are strict herbivores. Predators are not seeded — they evolve,
and they usually take several thousand ticks to appear.

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

## Layout

```
crates/borscht-core/    the simulation: no dependencies, no I/O
crates/borscht-wasm/    raw extern "C" surface for the browser
crates/borscht-cli/     headless runner, benchmarks, PNG and CSV output
web/                    the viewer: worker, WebGL2 renderer, UI
tools/                  build script, benchmarks, browser check
```

`borscht-core` has no dependencies at all. Determinism matters more here than
convenience, so the RNG is a hand-rolled PCG32 with its bit stream pinned by
test, and `exp`, `ln`, `sin`, `cos` and `tanh` are built from IEEE-exact
primitives rather than handed to a platform libm. A run in the browser and a run
from the CLI produce bit-identical results, which is what makes snapshots
portable between them.

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
same at every size and only the map changes. `--out` writes `stats.csv` with 27
per-tick measurements plus PNG frames. Any of the 53 parameters can be
overridden with `--set`; `borscht params` lists them with ranges and
descriptions.

```
borscht run --population 200000 --ticks 20000 --out runs/one --frames 40
borscht run --population 60000 --ticks 20000 --set temp_stress=2.0 --quiet
```

## Performance

Measured on four cores. WebAssembly figures come from running the same module
the browser runs, under Node.

| organisms | native ms/tick | wasm ms/tick | wasm ticks/s |
|-----------|---------------:|-------------:|-------------:|
| 35,000    | 1.9            | 2.5          | 407          |
| 177,000   | 10.6           | 15.1         | 66           |
| 356,000   | 21.9           | 32.7         | 31           |

Cost is linear in population at roughly 52 ns per organism natively and 94 ns in
WebAssembly, so a full million organisms is about 95 ms per tick in the browser,
or ten ticks a second. Rendering is decoupled from that: the simulation runs in
a worker and the page draws at display rate from whatever frame is current, so
panning and zooming stay smooth no matter how slowly the world is advancing.

Three things make that possible. Animals sense through per-cell aggregate fields
rather than walking their neighbours, which keeps cost flat at ~27 loads per
animal however crowded the world gets. Brain weights are `i8`, so a million
animals cost 194 MB of weights instead of 780 MB. And brains run on a stagger,
each animal thinking every fourth tick on its own offset, while movement
integrates every tick.

## Tests

```
cargo test                                        # 138 tests
cargo test --release --test ecology               # the ecology gate
cargo test --release --test ecology -- --ignored  # predator emergence (slow)
node tools/bench-wasm.mjs                         # wasm throughput
node tools/check-web.mjs                          # drives the real page
```

The unit tests check the machinery. The ecology gate checks the thing that
actually matters: across several seeds, both kingdoms persist, several species
coexist, plant biomass moves under grazing, matter is conserved, and traits
drift away from the founders in directions selection explains. A build can pass
every unit test and still produce a world where everything starves by tick 500 —
the first working one did.

`tools/check-web.mjs` drives the real page in headless Chromium and decodes the
screenshot to confirm organisms were actually drawn.

## Design notes

`docs/design.md` covers the tick pipeline, the memory layout, and the ecological
modelling — including the several ways early versions died, which is most of
what the design is shaped by.
