//! Headless runner for the battle simulator.
//!
//! Exists for the two things a browser cannot do well: run long unattended
//! experiments, and measure honestly. `bench` reports ticks per second across a
//! range of musters; `battle` fights one engagement and reports how it went.

mod png;
mod render;

use borscht_core::config::{Config, PARAMS};
use borscht_core::stats::STAT_NAMES;
use borscht_core::{Battle, ColorMode, Outcome};
use std::time::Instant;

mod sweep;

fn usage() -> ! {
    eprintln!(
        "borscht - mass battle simulator

USAGE:
    borscht bench   [--muster N]... [--ticks N] [--seed N]
    borscht battle  [--muster N] [--ticks N] [--seed N] [--out DIR]
                    [--frames N] [--image-size N] [--color MODE] [--set key=value]...
    borscht nerve   [--muster N] [--ticks N] [--seed N] [--set key=value]...
    borscht orders  [--muster N] [--ticks N] [--seed N] [--doctrine]
                    [--set key=value]...
    borscht sweep   [--muster N] [--ticks N] [--seeds N] [--set key=value]...
    borscht train   [--muster N] [--ticks N] [--seed N] [--generations N]
                    [--population N] [--sigma F] [--out FILE] [--log DIR]
                    [--verdicts PATH]
    borscht replay  DIR --match ID [--out DIR] [--frames N] [--color MODE]
    borscht params  [--json]
    borscht match    [--muster N] [--seeds N] [--ticks N]
    borscht commander

OPTIONS:
    --muster N       total units on the field, both sides; the field is scaled
                     to hold them at constant density
    --ticks N        ticks to simulate (default 2000)
    --seed N         battle seed (default 1)
    --out DIR        write stats.csv and frames here (default: no output)
    --frames N       number of PNG frames to write, spread over the battle
    --image-size N   frame edge in pixels
    --color MODE     team | kind | health | morale | division (default team)
    --set key=value  override any parameter; repeatable (`borscht params` lists them)
    --log DIR        record every battle a training run plays, so any of them
                     can be replayed afterwards
    --verdicts PATH  judgements on watched battles, as a .jsonl file or a
                     directory of documents; flagged seeds are trained on and
                     approved seeds are reported on separately
    --match ID       which recorded battle to replay
    --seeds N        battles in a sweep, run across cores (default 24)
    --doctrine       sweep with the hand-written commander instead of the
                     trained one, for comparing the two
    --generations N  training generations (default 30)
    --population N   commanders per generation (default 16)
    --sigma F        mutation size, per weight (default 0.08)
    --quiet          print one summary line of key=value pairs, for sweeps"
    );
    std::process::exit(2)
}

struct Args {
    command: String,
    musters: Vec<u32>,
    ticks: u32,
    seed: u64,
    out: Option<String>,
    frames: u32,
    image_size: u32,
    color: ColorMode,
    overrides: Vec<(String, f32)>,
    quiet: bool,
    json: bool,
    generations: u32,
    population: usize,
    sigma: f32,
    seeds: u64,
    doctrine: bool,
    log: Option<String>,
    verdicts: Option<String>,
    match_id: Option<u64>,
}

fn parse() -> Args {
    let mut args = Args {
        command: String::new(),
        musters: Vec::new(),
        ticks: 2000,
        seed: 1,
        out: None,
        frames: 0,
        image_size: 900,
        color: ColorMode::Team,
        overrides: Vec::new(),
        quiet: false,
        json: false,
        generations: 30,
        population: 16,
        sigma: 0.08,
        seeds: 24,
        doctrine: false,
        log: None,
        verdicts: None,
        match_id: None,
    };
    let mut it = std::env::args().skip(1);
    args.command = it.next().unwrap_or_else(|| usage());
    while let Some(key) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage());
        match key.as_str() {
            "--muster" => {
                let v = value();
                args.musters.push(v.parse().unwrap_or_else(|_| {
                    eprintln!("error: --muster wants an integer, got {v:?}");
                    std::process::exit(2)
                }));
            }
            "--ticks" => args.ticks = value().parse().unwrap_or(2000),
            "--seed" => args.seed = value().parse().unwrap_or(1),
            "--out" => args.out = Some(value()),
            "--frames" => args.frames = value().parse().unwrap_or(0),
            "--image-size" => args.image_size = value().parse().unwrap_or(900),
            "--color" => {
                args.color = match value().as_str() {
                    "kind" => ColorMode::Kind,
                    "health" => ColorMode::Health,
                    _ => ColorMode::Team,
                }
            }
            "--set" => {
                let raw = value();
                let Some((k, v)) = raw.split_once('=') else {
                    eprintln!("error: --set wants key=value, got {raw:?}");
                    std::process::exit(2)
                };
                let parsed = v.parse().unwrap_or_else(|_| {
                    eprintln!("error: {k} wants a number, got {v:?}");
                    std::process::exit(2)
                });
                args.overrides.push((k.to_string(), parsed));
            }
            "--seeds" => args.seeds = value().parse().unwrap_or(24),
            "--generations" => args.generations = value().parse().unwrap_or(30),
            "--population" => args.population = value().parse().unwrap_or(16),
            "--sigma" => args.sigma = value().parse().unwrap_or(0.08),
            "--log" => args.log = Some(value()),
            "--verdicts" => args.verdicts = Some(value()),
            "--match" => args.match_id = value().parse().ok(),
            "--doctrine" => args.doctrine = true,
            "--quiet" => args.quiet = true,
            "--json" => args.json = true,
            "--help" | "-h" => usage(),
            // `replay` names its log directory positionally, the way one
            // reaches for a path: `borscht replay runs/today --match 12`.
            other if !other.starts_with("--") && args.log.is_none() => {
                args.log = Some(other.to_string())
            }
            other => {
                eprintln!("error: unknown option {other:?}");
                usage()
            }
        }
    }
    args
}

fn build_config(muster: Option<u32>, overrides: &[(String, f32)]) -> Config {
    let mut cfg = match muster {
        Some(n) => Config::for_muster(n),
        None => Config::default(),
    };
    for (name, value) in overrides {
        match Config::param_id(name) {
            Some(id) => {
                if !cfg.set_param(id, *value) {
                    eprintln!("error: {name} rejected the value {value}");
                    std::process::exit(2);
                }
            }
            None => {
                eprintln!("error: no parameter named {name:?}; try `borscht params`");
                std::process::exit(2);
            }
        }
    }
    cfg.sanitize();
    cfg
}

fn main() {
    let args = parse();
    match args.command.as_str() {
        "bench" => bench(&args),
        "battle" | "run" => battle(&args),
        "sweep" => sweep::run(
            &build_config(
                args.musters.first().copied().or(Some(8_000)),
                &args.overrides,
            ),
            args.seeds,
            // Battles stop the moment they are decided, so a generous cap costs
            // nothing on the ones that end and prevents the ones that do not
            // from being scored as though they had. A cap that truncates
            // quietly turns "still fighting" into "nobody won".
            if args.ticks == 2000 { 8000 } else { args.ticks },
        ),
        "params" if args.json => emit_params_js(),
        "params" => list_params(),
        other => {
            eprintln!("error: unknown command {other:?}");
            usage()
        }
    }
}

// ------------------------------------------------------------------ params --

fn list_params() {
    let cfg = Config::default();
    let mut group = "";
    for (i, p) in PARAMS.iter().enumerate() {
        if p.group != group {
            group = p.group;
            println!("\n[{group}]");
        }
        println!(
            "  {:<20} {:>10}   range {} to {}",
            p.name,
            cfg.get_param(i as u32),
            p.lo,
            p.hi
        );
        for line in textwrap(&p.description(), 74) {
            println!("      {line}");
        }
    }
}

/// Wrap prose to a width without pulling in a dependency for it.
fn textwrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// The parameter table as an ES module, for the browser build.
fn emit_params_js() {
    let cfg = Config::default();
    println!("// Generated by `borscht params --json`. Do not edit.");
    println!("export const PARAMS = [");
    for (i, p) in PARAMS.iter().enumerate() {
        println!(
            "  {{ id: {i}, name: {:?}, group: {:?}, lo: {}, hi: {}, value: {}, doc: {:?} }},",
            p.name,
            p.group,
            p.lo,
            p.hi,
            cfg.get_param(i as u32),
            p.description()
        );
    }
    println!("];");
    println!("export const STAT_NAMES = [");
    for name in STAT_NAMES {
        println!("  {name:?},");
    }
    println!("];");
    // What each step of the build ramp is called. Generated rather than typed
    // into the page for the same reason the parameters are: one definition, in
    // the core, and no second copy to fall behind it.
    // The roster, so the page's key names the arms and says what each is for
    // without a second copy of either.
    println!("export const BUILD_NAMES = [");
    for build in borscht_core::army::ROSTER {
        println!("  {:?},", build.name);
    }
    println!("];");
    println!("export const BUILD_NOTES = [");
    for build in borscht_core::army::ROSTER {
        println!("  {:?},", build.note);
    }
    println!("];");
}

// ------------------------------------------------------------------- bench --

fn bench(args: &Args) {
    let musters = if args.musters.is_empty() {
        vec![20_000, 100_000, 500_000]
    } else {
        args.musters.clone()
    };
    let ticks = if args.ticks == 2000 { 200 } else { args.ticks };

    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "muster", "on field", "ms/tick", "ticks/s", "render ms", "MB"
    );
    for target in musters {
        let cfg = build_config(Some(target), &args.overrides);
        let mut b = Battle::new(cfg, args.seed);
        // A few ticks first so the buckets and fields are warm and the timing
        // is of the steady state rather than of the first allocation.
        b.tick_many(10);
        let on_field = b.units();

        let start = Instant::now();
        b.tick_many(ticks);
        let elapsed = start.elapsed().as_secs_f64();
        let ms = elapsed * 1000.0 / ticks as f64;

        let start = Instant::now();
        for _ in 0..10 {
            b.prepare_render(args.color);
        }
        let render_ms = start.elapsed().as_secs_f64() * 100.0;

        let bytes = on_field * (32 + borscht_core::battle::RENDER_STRIDE) + b.grid.cells() * 4 * 8;
        println!(
            "{target:>10} {on_field:>10} {ms:>10.2} {:>10.1} {render_ms:>10.2} {:>8}",
            1000.0 / ms,
            bytes / (1 << 20)
        );
    }
}

// ------------------------------------------------------------------ replay --

// ------------------------------------------------------------------- match --

// ------------------------------------------------------------------ orders --

// ------------------------------------------------------------------- nerve --

// ------------------------------------------------------------------ battle --

fn battle(args: &Args) {
    let cfg = build_config(args.musters.first().copied(), &args.overrides);
    let mut b = Battle::new(cfg, args.seed);
    let started = b.started();
    // Taken before a blow is struck: what each arm brought to the field.
    let mut mustered_by = [[0u32; 8]; 2];
    for i in 0..b.army.len() {
        mustered_by[b.army.team[i] as usize][b.army.kind[i] as usize] += 1;
    }

    let mut csv = String::from("tick,");
    csv.push_str(&STAT_NAMES.join(","));
    csv.push('\n');

    let frame_every = if args.frames > 0 {
        (args.ticks / args.frames).max(1)
    } else {
        u32::MAX
    };
    let mut frame = 0;

    // The ground each side is standing on when the fighting starts.
    //
    // This is the measurement that says whether terrain decides anything or is
    // merely decorative: if the side that came to the fight holding the higher
    // ground does not win more often than chance, the hills are scenery and the
    // honest thing is to say so rather than to turn the constants up.
    let mut ground = [0.0f32; 2];
    let mut contact_at = None;

    for tick in 0..args.ticks {
        b.tick();
        {
            if contact_at.is_none() && b.stats.red_killed + b.stats.blue_killed > 0.0 {
                contact_at = Some(tick);
                let mut sum = [0.0f64; 2];
                let mut men = [0u32; 2];
                for i in 0..b.army.len() {
                    if !b.army.alive(i) {
                        continue;
                    }
                    let t = b.army.team[i] as usize;
                    sum[t] += b.grid.height[b.grid.units.cell_of[i] as usize] as f64;
                    men[t] += 1;
                }
                for t in 0..2 {
                    ground[t] = (sum[t] / men[t].max(1) as f64) as f32;
                }
            }
        }
        if tick % 20 == 0 {
            csv.push_str(&format!("{tick},"));
            let row: Vec<String> = b.stats.as_slice().iter().map(|v| format!("{v}")).collect();
            csv.push_str(&row.join(","));
            csv.push('\n');
        }
        if tick % frame_every == 0 {
            if let Some(dir) = &args.out {
                let mut canvas = render::Canvas::new(args.image_size);
                canvas.draw(&mut b, args.color);
                let path = format!("{dir}/frame{frame:04}.png");
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(&path, canvas.encode());
                frame += 1;
            }
        }
        match b.outcome() {
            // A decision is the end of the battle; the pursuit after it is a
            // separate thing and the readout does not claim to measure it.
            Outcome::RedHolds | Outcome::BlueHolds => break,
            // A mutual break is not a decision, so it does not stop the clock.
            // It runs until the field empties of fugitives.
            Outcome::MutualBreak if b.units() == 0 => break,
            _ => {}
        }
    }

    let end = b.army.muster();
    // Decided on who is still holding, not on who is still breathing: the
    // battle ends when one army stops contesting the ground, and the men
    // streaming away from it are no longer part of that argument.
    let winner = match b.outcome() {
        Outcome::MutualBreak => "both armies broke",
        Outcome::BlueHolds => "blue holds the field",
        Outcome::RedHolds => "red holds the field",
        Outcome::Undecided => "undecided",
    };
    if args.quiet {
        println!(
            "ticks={} red={} blue={} red_started={} blue_started={} red_killed={} blue_killed={} red_ground={:.4} blue_ground={:.4} red_slope={:.5} blue_slope={:.5} contact={} winner={winner}",
            b.tick, end[0], end[1],
            started[0], started[1],
            b.stats.red_killed, b.stats.blue_killed,
            ground[0], ground[1],
            b.counters.blow_slope[0] / b.counters.blows[0].max(1) as f32,
            b.counters.blow_slope[1] / b.counters.blows[1].max(1) as f32,
            contact_at.map_or(-1i64, |t| t as i64)
        );
    } else {
        println!("after {} ticks: {winner}", b.tick);
        println!("  red   {:>8} of {:>8}", end[0], started[0]);
        println!("  blue  {:>8} of {:>8}", end[1], started[1]);
        println!(
            "  mean downhill on blows struck   red {:+.4}   blue {:+.4}",
            b.counters.blow_slope[0] / b.counters.blows[0].max(1) as f32,
            b.counters.blow_slope[1] / b.counters.blows[1].max(1) as f32,
        );
        println!(
            "  ground held at contact (tick {})   red {:.3}   blue {:.3}",
            contact_at.map_or("never".to_string(), |t| t.to_string()),
            ground[0],
            ground[1]
        );

        let c = b.counters;
        // What each arm did and what it cost, which is the only way to see
        // whether combined arms is combined arms or five kinds of swordsman.
        //
        // The question a roster cannot answer on its own: are the cavalry
        // getting among the archers, are the spears where the horse is, is
        // anything being decided at a distance? An arm that musters and dies at
        // the same rate as every other arm is not playing a different game, it
        // is wearing a different colour.
        let arms = borscht_core::army::arms_in_play(b.cfg.kinds);
        if arms > 1 {
            let shot: u32 = c.shot_kills.iter().sum();
            let total = (c.killed_fighting[0] + c.killed_fighting[1]).max(1);
            println!("\n  ARMS");
            println!(
                "    volleys loosed {} + {}, killed {shot} ({:.0}% of the dead)",
                c.volleys[0],
                c.volleys[1],
                100.0 * shot as f32 / total as f32
            );
            println!(
                "    {:<10} {:>9} {:>9} {:>8}",
                "arm", "mustered", "standing", "lost"
            );
            let mut alive_by = [[0u32; 8]; 2];
            for i in 0..b.army.len() {
                if b.army.alive(i) {
                    alive_by[b.army.team[i] as usize][b.army.kind[i] as usize] += 1;
                }
            }
            for kind in 0..arms {
                let mustered: u32 = mustered_by[0][kind] + mustered_by[1][kind];
                let standing: u32 = alive_by[0][kind] + alive_by[1][kind];
                println!(
                    "    {:<10} {mustered:>9} {standing:>9} {:>7.0}%",
                    borscht_core::army::build_name(kind),
                    100.0 * (mustered.saturating_sub(standing)) as f32 / mustered.max(1) as f32
                );
            }
        }
    }

    if let Some(dir) = &args.out {
        let _ = std::fs::create_dir_all(dir);
        let path = format!("{dir}/stats.csv");
        match std::fs::write(&path, csv) {
            Ok(()) => println!("wrote {path}"),
            Err(e) => eprintln!("could not write {path}: {e}"),
        }
    }
}
