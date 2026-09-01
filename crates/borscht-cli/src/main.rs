//! Headless runner for the Borscht evolution simulator.
//!
//! Exists for the two things a browser cannot do well: run long unattended
//! experiments, and measure honestly. `bench` reports ticks per second at a
//! range of populations; `run` produces a stats CSV and PNG frames.

mod png;
mod render;

use borscht_core::config::{Config, PARAMS};
use borscht_core::stats::STAT_NAMES;
use borscht_core::{ColorMode, World};
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "borscht - evolution simulator

USAGE:
    borscht bench   [--population N]... [--ticks N] [--seed N]
    borscht diag    [--population N] [--ticks N] [--seed N] [--set key=value]...
    borscht run     [--population N] [--ticks N] [--seed N] [--out DIR]
                    [--frames N] [--image-size N] [--color MODE] [--set key=value]...
    borscht params [--json]

OPTIONS:
    --population N   total organisms; the world is scaled to hold them at
                     constant density. Omit for the default world, which holds
                     about a hundred thousand.
    --ticks N        ticks to simulate (default 2000)
    --seed N         world seed (default: drawn from system entropy, so runs
                     differ; pass one to repeat a particular run)
    --out DIR        write stats.csv and frames here (default: no output)
    --frames N       number of PNG frames to write, spread over the run
    --image-size N   frame edge in pixels (default: matched to the population,
                     so organisms are dense enough to read)
    --color MODE     species | diet | energy | age | size (default species)
    --set key=value  override any simulation parameter; repeatable
                     (`borscht params` lists them)
    --quiet          suppress the per-tick table and print one summary line of
                     key=value pairs, for parameter sweeps

DIAG:
    `diag` answers questions about *why* a world looks the way it does, which
    the stats table does not: whether the tree is recording real lineages or
    sampling noise, whether the herd is a foraging population or a single
    advancing front, and what is actually stopping births."
    );
    std::process::exit(2)
}

struct Args {
    command: String,
    seed_given: bool,
    populations: Vec<u32>,
    ticks: u32,
    seed: u64,
    out: Option<String>,
    frames: u32,
    image_size: u32,
    color: ColorMode,
    overrides: Vec<(String, f32)>,
    quiet: bool,
    json: bool,
}

fn parse_args() -> Args {
    let mut argv = std::env::args().skip(1);
    let command = argv.next().unwrap_or_else(|| usage());
    let mut args = Args {
        command,
        seed_given: false,
        populations: Vec::new(),
        ticks: 2000,
        seed: 1,
        out: None,
        frames: 0,
        image_size: 0,
        color: ColorMode::Species,
        overrides: Vec::new(),
        quiet: false,
        json: false,
    };
    let rest: Vec<String> = argv.collect();
    let mut i = 0;
    while i < rest.len() {
        let key = rest[i].as_str();
        let mut value = || {
            i += 1;
            rest.get(i).cloned().unwrap_or_else(|| {
                eprintln!("error: {key} needs a value");
                std::process::exit(2)
            })
        };
        match key {
            "--population" => {
                let v = value();
                args.populations.push(v.parse().unwrap_or_else(|_| {
                    eprintln!("error: --population wants an integer, got {v:?}");
                    std::process::exit(2)
                }));
            }
            "--ticks" => args.ticks = value().parse().unwrap_or(2000),
            "--seed" => {
                args.seed = value().parse().unwrap_or(1);
                args.seed_given = true;
            }
            "--out" => args.out = Some(value()),
            "--frames" => args.frames = value().parse().unwrap_or(0),
            "--image-size" => args.image_size = value().parse().unwrap_or(1024),
            "--color" => {
                args.color = match value().as_str() {
                    "diet" => ColorMode::Diet,
                    "energy" => ColorMode::Energy,
                    "age" => ColorMode::Age,
                    "size" => ColorMode::Size,
                    _ => ColorMode::Species,
                }
            }
            "--set" => {
                let v = value();
                let (name, val) = v.split_once('=').unwrap_or_else(|| {
                    eprintln!("error: --set wants key=value, got {v:?}");
                    std::process::exit(2)
                });
                let parsed: f32 = val.parse().unwrap_or_else(|_| {
                    eprintln!("error: {name} wants a number, got {val:?}");
                    std::process::exit(2)
                });
                if Config::param_id(name).is_none() {
                    eprintln!("error: unknown parameter {name:?}; try `borscht params`");
                    std::process::exit(2);
                }
                args.overrides.push((name.to_string(), parsed));
            }
            "--quiet" | "-q" => {
                args.quiet = true;
                i += 1;
                continue;
            }
            "--json" => {
                args.json = true;
                i += 1;
                continue;
            }
            "--help" | "-h" => usage(),
            other => {
                eprintln!("error: unknown option {other:?}");
                usage()
            }
        }
        i += 1;
    }
    args
}

/// `None` means the default world, which is already about a hundred thousand
/// organisms; a value rescales it at constant density.
fn build_config(population: Option<u32>, overrides: &[(String, f32)]) -> Config {
    let mut cfg = match population {
        Some(target) => Config::for_population(target),
        None => Config::default(),
    };
    for (name, value) in overrides {
        if let Some(id) = Config::param_id(name) {
            cfg.set_param(id, *value);
        }
    }
    cfg.sanitize();
    cfg
}

fn main() {
    let mut args = parse_args();
    if !args.seed_given {
        args.seed = borscht_core::rng::entropy_seed();
    }
    match args.command.as_str() {
        "bench" => bench(&args),
        "diag" => diag(&args),
        "run" => run(&args),
        "params" if args.json => emit_params_js(),
        "params" => list_params(),
        other => {
            eprintln!("error: unknown command {other:?}");
            usage()
        }
    }
}

/// Emit `web/params.js`.
///
/// Generated rather than hand-written so the browser's parameter table cannot
/// drift from the Rust definitions: adding a parameter in `config.rs` is the
/// only edit needed for it to appear in the UI with the right range and help
/// text.
fn emit_params_js() {
    let cfg = Config::default();
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    println!("// Generated by `borscht params --json`. Do not edit.");
    println!("export const PARAMS = [");
    for (i, p) in PARAMS.iter().enumerate() {
        println!(
            "  {{ id: {i}, name: \"{}\", group: \"{}\", lo: {}, hi: {}, value: {}, doc: \"{}\" }},",
            esc(p.name),
            esc(p.group),
            p.lo,
            p.hi,
            cfg.get_param(i as u32),
            esc(&p.description())
        );
    }
    println!("];");
    println!("export const STAT_NAMES = [");
    for name in STAT_NAMES {
        println!("  \"{}\",", esc(name));
    }
    println!("];");

    // Gene names are generated for the same reason the parameters are: the
    // inspector labels traits positionally, so a hand-written list silently
    // mislabels every gene the moment one is renamed or reordered.
    println!("export const ANIMAL_GENES = [");
    for spec in borscht_core::genome::ANIMAL_GENES {
        println!("  \"{}\",", esc(spec.name));
    }
    println!("];");
    println!("export const PLANT_GENES = [");
    for spec in borscht_core::genome::PLANT_GENES {
        println!("  \"{}\",", esc(spec.name));
    }
    println!("];");
}

fn list_params() {
    let cfg = Config::default();
    let mut group = "";
    for (i, p) in PARAMS.iter().enumerate() {
        if p.group != group {
            group = p.group;
            println!("\n[{group}]");
        }
        println!(
            "  {:<24} {:>12}   {:>10} .. {:<10}  {}",
            p.name,
            format!("{}", cfg.get_param(i as u32)),
            p.lo,
            p.hi,
            p.description()
        );
    }
}

/// Measure *why* a world looks the way it does.
///
/// The stats table says what the populations are; this says whether the tree is
/// recording lineages or noise, whether the animals are a foraging population or
/// one advancing front, and what is actually stopping births. Every number here
/// exists because a specific complaint about the simulation needed evidence
/// rather than an opinion.
fn diag(args: &Args) {
    let cfg = build_config(args.populations.first().copied(), &args.overrides);
    let seed = args.seed;
    let mut world = World::new(cfg, seed);
    let ticks = if args.ticks == 2000 { 6000 } else { args.ticks };

    // Displacement needs a before, so positions are sampled and re-read a fixed
    // interval later rather than integrated over the whole run: what matters is
    // how far an animal gets in a hundred ticks, not how far it wanders in six
    // thousand.
    const WINDOW: u32 = 100;
    let mut marks: Vec<(u32, f32, f32)> = Vec::new();
    let mut displacement = Vec::new();

    for tick in 0..ticks {
        world.tick();
        if tick % WINDOW == 0 {
            if !marks.is_empty() {
                let size = world.cfg.world_size;
                for &(id, x0, y0) in &marks {
                    if let Some(i) = (0..world.animals.len()).find(|&i| world.animals.id[i] == id) {
                        let d2 = borscht_core::grid::wrap_dist_sq(
                            x0,
                            y0,
                            world.animals.x[i],
                            world.animals.y[i],
                            size,
                        );
                        displacement.push(d2.sqrt());
                    }
                }
            }
            marks.clear();
            // A sample, not the whole population: the lookup below is linear.
            let step = (world.animals.len() / 200).max(1);
            for i in (0..world.animals.len()).step_by(step) {
                marks.push((world.animals.id[i], world.animals.x[i], world.animals.y[i]));
            }
        }
    }

    let tables = borscht_core::genome::tables();
    use borscht_core::genome::{ag, pg};

    println!(
        "seed {seed}, {ticks} ticks, {} organisms",
        world.population()
    );
    println!(
        "  plants {}   animals {}",
        world.plants.len(),
        world.animals.len()
    );

    // ---- the tree: lineages, or sampling noise? ----
    let history = &world.animal_species.history;
    let total = history.len().max(1);
    let above = |n: u32| history.iter().filter(|l| l.peak_population > n).count();
    println!("\nTREE");
    println!("  lineages recorded          {}", history.len());
    println!(
        "  never exceeded 1 individual {:>6}  ({:.1}%)",
        history.len() - above(1),
        100.0 * (history.len() - above(1)) as f32 / total as f32
    );
    println!(
        "  reached 8                   {:>6}  ({:.1}%)",
        above(8),
        100.0 * above(8) as f32 / total as f32
    );
    println!(
        "  reached 50                  {:>6}  ({:.1}%)",
        above(50),
        100.0 * above(50) as f32 / total as f32
    );
    let births = world.stats.get("animal_births").unwrap_or(0.0).max(1.0);
    println!(
        "  splits per 1000 births      {:>6.1}",
        1000.0 * history.len() as f32 / (births * ticks as f32)
    );

    // ---- the grazing front ----
    // A plant held at the refuge floor is one grazing has pinned: it cannot
    // regrow, because logistic growth is proportional to the biomass it has
    // left.
    let refuge = world.cfg.graze_refuge;
    let mut pinned = 0usize;
    let mut alive = 0usize;
    for i in 0..world.plants.len() {
        if !world.plants.alive[i] {
            continue;
        }
        alive += 1;
        let max_size = tables.plant[pg::MAX_SIZE][world.plants.gene(i, pg::MAX_SIZE) as usize];
        if world.plants.biomass[i] <= refuge * max_size * 1.05 {
            pinned += 1;
        }
    }
    // Circular concentration of headings. One is a column marching in step -- a
    // front; zero is a population milling in every direction.
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    for i in 0..world.animals.len() {
        sx += borscht_core::fastmath::cos(world.animals.heading[i]);
        sy += borscht_core::fastmath::sin(world.animals.heading[i]);
    }
    let n = world.animals.len().max(1) as f32;
    let concentration = (sx * sx + sy * sy).sqrt() / n;
    displacement.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = displacement
        .get(displacement.len() / 2)
        .copied()
        .unwrap_or(0.0);
    println!("\nGRAZING");
    println!(
        "  plants pinned at refuge     {:>6.1}%  ({pinned} of {alive})",
        100.0 * pinned as f32 / alive.max(1) as f32
    );
    println!(
        "  heading concentration       {:>6.2}   (1 = one column in step, 0 = milling)",
        concentration
    );
    println!(
        "  median travel / {WINDOW} ticks  {:>6.1}   ({:.1}% of the world)",
        median,
        100.0 * median / world.cfg.world_size
    );
    // Net displacement alone cannot tell a stationary animal from one that
    // swims hard in circles, and the two want opposite fixes. Realised speed
    // against the speed the genes allow separates them.
    let mut realised = 0.0f64;
    let mut capable = 0.0f64;
    let mut thrust = 0.0f64;
    let mut turn = 0.0f64;
    for i in 0..world.animals.len() {
        realised += world.animals.speed[i] as f64;
        capable +=
            tables.animal[ag::MAX_SPEED][world.animals.gene(i, ag::MAX_SPEED) as usize] as f64;
        thrust += world
            .animals
            .action_of(i, borscht_core::brain::output::THRUST) as f64;
        turn += world
            .animals
            .action_of(i, borscht_core::brain::output::TURN)
            .abs() as f64;
    }
    let n64 = world.animals.len().max(1) as f64;
    println!(
        "  mean speed                  {:>6.3}   of {:.3} the genes allow ({:.0}% of capacity)",
        realised / n64,
        capable / n64,
        100.0 * realised / capable.max(1e-9)
    );
    println!(
        "  path length / {WINDOW} ticks    {:>6.1}   versus {median:.1} of net travel",
        100.0 * realised / n64
    );
    println!(
        "  mean thrust {:>6.2}   mean |turn| {:>6.2}",
        thrust / n64,
        turn / n64
    );

    // ---- satiation ----
    // An animal at capacity that keeps cropping wastes the plant and never has
    // a reason to stop and do something else.
    let mut full = 0usize;
    for i in 0..world.animals.len() {
        let size = tables.animal[ag::SIZE][world.animals.gene(i, ag::SIZE) as usize];
        let cap = world.cfg.mass_per_size * size * world.cfg.reserve_capacity;
        if world.animals.reserve[i] >= cap * 0.98 {
            full += 1;
        }
    }
    println!("\nSATIATION");
    println!(
        "  animals at full reserves    {:>6.1}%  ({full} of {})",
        100.0 * full as f32 / world.animals.len().max(1) as f32,
        world.animals.len()
    );

    // ---- what is stopping births ----
    let stat = |n: &str| world.stats.get(n).unwrap_or(0.0);
    println!("\nBIRTHS BLOCKED (per tick, at the end of the run)");
    println!(
        "  on finding a mate           {:>8.2}",
        stat("animal_births_blocked_mate")
    );
    println!(
        "  on matter                   {:>8.2}",
        stat("animal_births_blocked_matter")
    );
    println!(
        "  on space                    {:>8.2}",
        stat("animal_births_blocked_space")
    );
    println!(
        "  actual births               {:>8.2}",
        stat("animal_births")
    );
}

fn bench(args: &Args) {
    let populations = if args.populations.is_empty() {
        vec![25_000, 100_000, 400_000]
    } else {
        args.populations.clone()
    };
    let ticks = if args.ticks == 2000 { 200 } else { args.ticks };

    println!(
        "{:>12}  {:>10}  {:>10}  {:>10}  {:>9}  {:>8}",
        "target", "organisms", "ms/tick", "ticks/s", "build ms", "MB"
    );
    for target in populations {
        let cfg = build_config(Some(target), &args.overrides);
        let build_start = Instant::now();
        let mut world = World::new(cfg, args.seed);
        let build_ms = build_start.elapsed().as_secs_f64() * 1e3;

        // Warm up so the measurement is not dominated by first-touch page
        // faults on freshly allocated pools.
        world.tick_many(10.min(ticks));

        let start = Instant::now();
        world.tick_many(ticks);
        let elapsed = start.elapsed().as_secs_f64();
        let per_tick = elapsed / ticks as f64;

        println!(
            "{:>12}  {:>10}  {:>10.2}  {:>10.1}  {:>9.0}  {:>8.0}",
            target,
            world.population(),
            per_tick * 1e3,
            1.0 / per_tick,
            build_ms,
            estimate_memory_mb(&world),
        );
    }
}

/// Population extremes over the back half of a run, used to tell a stable
/// ecosystem from one that merely happened to be alive at the final tick.
struct Tail {
    plant_min: f32,
    plant_max: f32,
    animal_min: f32,
    animal_max: f32,
    peak_animal_species: f32,
    kills: f64,
    ticks: f64,
    energy_per_animal: f64,
}

impl Default for Tail {
    fn default() -> Self {
        Tail {
            plant_min: f32::MAX,
            plant_max: 0.0,
            animal_min: f32::MAX,
            animal_max: 0.0,
            peak_animal_species: 0.0,
            kills: 0.0,
            ticks: 0.0,
            energy_per_animal: 0.0,
        }
    }
}

impl Tail {
    fn observe(&mut self, s: &borscht_core::stats::Stats) {
        self.plant_min = self.plant_min.min(s.plants);
        self.plant_max = self.plant_max.max(s.plants);
        self.animal_min = self.animal_min.min(s.animals);
        self.animal_max = self.animal_max.max(s.animals);
        self.peak_animal_species = self.peak_animal_species.max(s.animal_species);
        self.kills += s.kills as f64;
        self.ticks += 1.0;
        if s.animals > 0.0 {
            self.energy_per_animal += (s.animal_energy / s.animals) as f64;
        }
    }

    /// Peak-to-trough ratio. 1.0 is flat; large values mean boom and bust.
    fn swing(&self, min: f32, max: f32) -> f32 {
        if min <= 0.0 {
            return f32::INFINITY;
        }
        max / min
    }
}

fn estimate_memory_mb(world: &World) -> f64 {
    let animals = world.animals.capacity() as f64
        * (borscht_core::brain::BRAIN_LEN + borscht_core::genome::ANIMAL_GENE_COUNT + 32) as f64;
    let plants =
        world.plants.capacity() as f64 * (borscht_core::genome::PLANT_GENE_COUNT + 24) as f64;
    let grid = world.grid.cells() as f64 * 6.0 * 4.0;
    (animals + plants + grid) / (1024.0 * 1024.0)
}

fn run(args: &Args) {
    let cfg = build_config(args.populations.first().copied(), &args.overrides);
    let mut world = World::new(cfg, args.seed);
    if !args.quiet {
        println!("seed {}", args.seed);
    }

    let out = args.out.as_deref();
    if let Some(dir) = out {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| {
            eprintln!("error: cannot create {dir}: {e}");
            std::process::exit(1)
        });
    }

    let mut csv = String::new();
    csv.push_str(&STAT_NAMES.join(","));
    csv.push('\n');

    // A point cloud drawn one organism per pixel reads as noise when the image
    // is much larger than the population. Size the frame so a good fraction of
    // pixels are occupied unless the caller says otherwise.
    let image_size = if args.image_size > 0 {
        args.image_size
    } else {
        ((world.population() as f64 / 1.5).sqrt() as u32).clamp(256, 2048)
    };
    let mut canvas = args.frames.gt(&0).then(|| render::Canvas::new(image_size));
    let frame_every = if args.frames > 0 {
        (args.ticks / args.frames).max(1)
    } else {
        u32::MAX
    };
    let mut frame_index = 0u32;

    if !args.quiet {
        println!(
            "{:>8}  {:>9}  {:>8}  {:>7}  {:>7}  {:>7}  {:>6}  {:>6}  {:>7}",
            "tick", "plants", "animals", "spp", "a.born", "a.died", "soil", "diet", "carn%"
        );
    }

    // Stability is judged over the back half of the run, after the founding
    // transient has washed out.
    let settle = args.ticks / 2;
    let mut tail = Tail::default();

    let start = Instant::now();
    let report_every = (args.ticks / 20).max(1);
    for tick in 0..args.ticks {
        world.tick();
        let s = world.stats;
        csv.push_str(
            &s.as_slice()
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');

        if tick >= settle {
            tail.observe(&s);
        }

        if !args.quiet && (tick % report_every == 0 || tick == args.ticks - 1) {
            println!(
                "{:>8}  {:>9.0}  {:>8.0}  {:>7.0}  {:>7.0}  {:>7.0}  {:>7.0}  {:>6.2}  {:>6.1}%",
                s.tick,
                s.plants,
                s.animals,
                s.animal_species,
                s.animal_births,
                s.animal_deaths,
                s.soil,
                s.mean_diet,
                s.carnivore_fraction * 100.0
            );
        }

        if let (Some(canvas), Some(dir)) = (canvas.as_mut(), out) {
            if tick % frame_every == 0 {
                canvas.draw(&mut world, args.color);
                let path = format!("{dir}/frame_{frame_index:04}.png");
                if let Err(e) = std::fs::write(&path, canvas.encode()) {
                    eprintln!("warning: could not write {path}: {e}");
                }
                frame_index += 1;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    if args.quiet {
        let s = world.stats;
        println!(
            "plants={:.0} animals={:.0} plant_spp={:.0} animal_spp={:.0} \
carn={:.3} diet={:.3} size={:.2} speed={:.2} mut={:.4} \
plant_min={:.0} plant_max={:.0} animal_min={:.0} animal_max={:.0} \
plant_swing={:.2} animal_swing={:.2} peak_spp={:.0} kills_per_tick={:.1} energy={:.1} ticks_per_s={:.0}",
            s.plants, s.animals, s.plant_species, s.animal_species,
            s.carnivore_fraction, s.mean_diet, s.mean_size, s.mean_max_speed,
            s.mean_mutation_rate,
            tail.plant_min, tail.plant_max, tail.animal_min, tail.animal_max,
            tail.swing(tail.plant_min, tail.plant_max),
            tail.swing(tail.animal_min, tail.animal_max),
            tail.peak_animal_species,
            tail.kills / tail.ticks.max(1.0),
            tail.energy_per_animal / tail.ticks.max(1.0),
            args.ticks as f64 / elapsed,
        );
    } else {
        println!(
            "\n{} ticks in {:.1}s ({:.1} ticks/s), {} organisms alive",
            args.ticks,
            elapsed,
            args.ticks as f64 / elapsed,
            world.population()
        );
    }

    if let Some(dir) = out {
        let path = format!("{dir}/stats.csv");
        match std::fs::write(&path, csv) {
            Ok(()) => println!(
                "wrote {path}{}",
                if frame_index > 0 {
                    format!(" and {frame_index} frames")
                } else {
                    String::new()
                }
            ),
            Err(e) => eprintln!("warning: could not write {path}: {e}"),
        }
    }
}
