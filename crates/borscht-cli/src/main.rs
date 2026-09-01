//! Headless runner for the battle simulator.
//!
//! Exists for the two things a browser cannot do well: run long unattended
//! experiments, and measure honestly. `bench` reports ticks per second across a
//! range of musters; `battle` fights one engagement and reports how it went.

mod png;
mod render;

use borscht_core::config::{Config, PARAMS};
use borscht_core::stats::STAT_NAMES;
use borscht_core::{Battle, ColorMode};
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "borscht - mass battle simulator

USAGE:
    borscht bench   [--muster N]... [--ticks N] [--seed N]
    borscht battle  [--muster N] [--ticks N] [--seed N] [--out DIR]
                    [--frames N] [--image-size N] [--color MODE] [--set key=value]...
    borscht params  [--json]

OPTIONS:
    --muster N       total units on the field, both sides; the field is scaled
                     to hold them at constant density
    --ticks N        ticks to simulate (default 2000)
    --seed N         battle seed (default 1)
    --out DIR        write stats.csv and frames here (default: no output)
    --frames N       number of PNG frames to write, spread over the battle
    --image-size N   frame edge in pixels
    --color MODE     team | kind | health | morale (default team)
    --set key=value  override any parameter; repeatable (`borscht params` lists them)
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
    };
    let mut it = std::env::args().skip(1);
    args.command = it.next().unwrap_or_else(|| usage());
    while let Some(key) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage());
        match key.as_str() {
            "--muster" | "--population" => {
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
                    "morale" => ColorMode::Morale,
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
            "--quiet" => args.quiet = true,
            "--json" => args.json = true,
            "--help" | "-h" => usage(),
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

// ------------------------------------------------------------------ battle --

fn battle(args: &Args) {
    let cfg = build_config(args.musters.first().copied(), &args.overrides);
    let mut b = Battle::new(cfg, args.seed);
    let started = b.started();

    let mut csv = String::from("tick,");
    csv.push_str(&STAT_NAMES.join(","));
    csv.push('\n');

    let frame_every = if args.frames > 0 {
        (args.ticks / args.frames).max(1)
    } else {
        u32::MAX
    };
    let mut frame = 0;

    // The rout trace. The spread between these is the measurement that matters:
    // a real collapse is progressive and accelerating, so if 10%, 50% and 90%
    // land on nearly the same tick the whole army broke at once, which is a
    // flash rout and looks fake however good every other number is.
    let mut milestones = [[None::<u32>; 3]; 2];
    let mut peak_routing = 0u32;

    for tick in 0..args.ticks {
        b.tick();
        {
            let alive = b.army.muster();
            let mut routing_now = 0u32;
            for team in 0..2 {
                let holding = b.army.holding(team as u8);
                let routing = alive[team].saturating_sub(holding);
                routing_now += routing;
                // Against the men still on the field, not against the muster:
                // a router who is cut down stops counting as routing, so a
                // share of the starting strength can never reach the top of
                // the scale and the last milestone would never fire.
                let share = routing as f32 / alive[team].max(1) as f32;
                for (slot, want) in [0.10f32, 0.50, 0.90].iter().enumerate() {
                    if milestones[team][slot].is_none() && share >= *want {
                        milestones[team][slot] = Some(tick);
                    }
                }
            }
            peak_routing = peak_routing.max(routing_now);
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
        if b.decided() {
            break;
        }
    }

    let end = b.army.muster();
    // Decided on who is still holding, not on who is still breathing: the
    // battle ends when one army stops contesting the ground, and the men
    // streaming away from it are no longer part of that argument.
    let holding = [b.army.holding(0), b.army.holding(1)];
    let winner = match (holding[0], holding[1]) {
        (0, 0) => "both armies broke",
        (0, _) => "blue holds the field",
        (_, 0) => "red holds the field",
        _ => "undecided",
    };
    if args.quiet {
        println!(
            "ticks={} red={} blue={} red_started={} blue_started={} red_killed={} blue_killed={} winner={winner}",
            b.tick, end[0], end[1], started[0], started[1],
            b.stats.red_killed, b.stats.blue_killed
        );
    } else {
        println!("after {} ticks: {winner}", b.tick);
        println!(
            "  red   {:>8} of {:>8} ({} holding)",
            end[0], started[0], b.stats.red_holding
        );
        println!(
            "  blue  {:>8} of {:>8} ({} holding)",
            end[1], started[1], b.stats.blue_holding
        );
        println!(
            "  mean nerve   red {:.2}   blue {:.2}",
            b.stats.red_morale, b.stats.blue_morale
        );

        let c = b.counters;
        let at = |m: Option<u32>| m.map_or("never".to_string(), |t| t.to_string());
        println!("\n  ROUT");
        for (team, name) in ["red", "blue"].iter().enumerate() {
            println!(
                "    {name:<5} broke {:>8}   10% at {:>6}   50% at {:>6}   90% at {:>6}",
                c.broke[team],
                at(milestones[team][0]),
                at(milestones[team][1]),
                at(milestones[team][2]),
            );
        }
        println!(
            "    peak running at once {peak_routing}, rallied {}",
            c.rallied
        );
        let fighting: u32 = c.killed_fighting.iter().sum();
        let running: u32 = c.killed_routing.iter().sum();
        let total = (fighting + running).max(1);
        println!(
            "    cut down fighting {fighting} ({:.0}%), running {running} ({:.0}%)",
            100.0 * fighting as f32 / total as f32,
            100.0 * running as f32 / total as f32
        );
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
