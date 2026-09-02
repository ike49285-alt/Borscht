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

mod matchlog;
mod sweep;
mod train;
mod verdict;

fn usage() -> ! {
    eprintln!(
        "borscht - mass battle simulator

USAGE:
    borscht bench   [--muster N]... [--ticks N] [--seed N]
    borscht battle  [--muster N] [--ticks N] [--seed N] [--out DIR]
                    [--frames N] [--image-size N] [--color MODE] [--set key=value]...
    borscht nerve   [--muster N] [--ticks N] [--seed N] [--set key=value]...
    borscht orders  [--muster N] [--ticks N] [--seed N] [--set key=value]...
    borscht sweep   [--muster N] [--ticks N] [--seeds N] [--set key=value]...
    borscht train   [--muster N] [--ticks N] [--seed N] [--generations N]
                    [--population N] [--sigma F] [--out FILE] [--log DIR]
                    [--verdicts PATH]
    borscht replay  DIR --match ID [--out DIR] [--frames N] [--color MODE]
    borscht params  [--json]
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
                    "morale" => ColorMode::Morale,
                    "division" => ColorMode::Division,
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
        "nerve" => nerve(&args),
        "orders" => orders(&args),
        "replay" => replay(&args),
        "sweep" => sweep::run(
            &build_config(args.musters.first().copied().or(Some(8_000)), &args.overrides),
            args.seeds,
            // Battles stop the moment they are decided, so a generous cap costs
            // nothing on the ones that end and prevents the ones that do not
            // from being scored as though they had. A cap that truncates
            // quietly turns "still fighting" into "nobody won".
            if args.ticks == 2000 { 8000 } else { args.ticks },
            args.doctrine,
        ),
        "train" => train::run(&train::Plan {
            cfg: build_config(Some(args.musters.first().copied().unwrap_or(4_000)), &args.overrides),
            seed: args.seed,
            ticks: if args.ticks == 2000 { 1500 } else { args.ticks },
            generations: args.generations,
            population: args.population,
            sigma: args.sigma,
            out: args.out.clone(),
            log: args.log.clone(),
            verdicts: args.verdicts.clone(),
        }),
        // Which commander does this build actually ship? The page reports the
        // same name from the WebAssembly module, and a verdict recorded there
        // is worthless if the two ever disagree -- so it is checkable from the
        // shell rather than only inferable.
        "commander" => println!("{}", matchlog::name_of(&borscht_core::brain::Net::trained())),
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

// ------------------------------------------------------------------ replay --

/// Fight a recorded battle again and report what happened.
///
/// The point of the match log: a training run's twenty thousand battles stop
/// being a number and become a list you can pull any line out of and watch.
fn replay(args: &Args) {
    let Some(dir) = &args.log else {
        eprintln!("error: replay needs a log directory, e.g. `borscht replay runs/today --match 12`");
        std::process::exit(2);
    };
    let Some(id) = args.match_id else {
        eprintln!("error: replay needs --match ID");
        std::process::exit(2);
    };
    let recorded = match matchlog::find(std::path::Path::new(dir), id) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    let row = recorded.row.clone();
    println!(
        "match {id} from {}: seed {}, stream {}, red {} against blue {}",
        row.phase, row.seed, row.stream, row.red, row.blue
    );

    // Frames are spread over the battle's recorded length rather than a tick
    // cap: a replay is the one case where how long the fight ran is known
    // before it is fought again.
    let mut canvas = args
        .out
        .as_ref()
        .filter(|_| args.frames > 0)
        .map(|out| {
            let _ = std::fs::create_dir_all(out);
            render::Canvas::new(args.image_size)
        });
    let frame_every = (row.ticks / args.frames.max(1) as u64).max(1);
    let mut frames_written = 0u32;

    let mut b = matchlog::replay(&recorded, |b| {
        let Some(canvas) = canvas.as_mut() else {
            return;
        };
        if b.tick % frame_every != 0 || frames_written >= args.frames {
            return;
        }
        canvas.draw(b, args.color);
        let out = args.out.as_ref().expect("a canvas implies an output dir");
        let _ = std::fs::write(
            format!("{out}/match{id:06}-{frames_written:03}.png"),
            canvas.encode(),
        );
        frames_written += 1;
    });
    let alive = b.army.muster();
    // The claim the whole record rests on. If a replay disagrees with the row it
    // came from, the log is decoration and had better say so out loud.
    let faithful = alive == row.alive && b.tick == row.ticks;
    println!(
        "  recorded: {} ticks, red {} blue {}",
        row.ticks, row.alive[0], row.alive[1]
    );
    println!(
        "  replayed: {} ticks, red {} blue {}   {}",
        b.tick,
        alive[0],
        alive[1],
        if faithful {
            "-- identical"
        } else {
            "-- DIFFERENT, the log cannot be trusted"
        }
    );

    // The last frame is the one worth having whatever else was asked for: it
    // is how the battle actually ended.
    if let Some(out) = &args.out {
        let _ = std::fs::create_dir_all(out);
        let mut canvas = render::Canvas::new(args.image_size);
        canvas.draw(&mut b, args.color);
        let path = format!("{out}/match{id:06}-end.png");
        let _ = std::fs::write(&path, canvas.encode());
        println!("  wrote {frames_written} frames and {path}");
    }
    if !faithful {
        std::process::exit(1);
    }
}

// ------------------------------------------------------------------ orders --

/// What the commanders decided, and whether the divisions are still divisions.
///
/// Two questions this exists to answer, and neither is visible in a battle
/// summary. Does the doctrine send every division to the same sector -- in
/// which case this is the old single order point wearing six times the
/// machinery? And is a reserve ever actually committed, or does it stand behind
/// the line for the whole battle?
fn orders(args: &Args) {
    let cfg = build_config(args.musters.first().copied(), &args.overrides);
    let divisions = (cfg.divisions as usize).min(borscht_core::army::MAX_DIVISIONS);
    let mut b = Battle::new(cfg, args.seed);
    let every = (args.ticks / 16).max(1);

    // When each division first stopped being a reserve.
    let mut committed = [[None::<u32>; borscht_core::army::MAX_DIVISIONS]; 2];

    println!(
        "{:>6} {:>5} {:>7} {:>28} {:>9}",
        "tick", "side", "spread", "postures", "sectors"
    );
    for tick in 0..args.ticks {
        b.tick();
        for (team, marks) in committed.iter_mut().enumerate() {
            for (d, mark) in marks.iter_mut().enumerate().take(divisions) {
                if mark.is_none()
                    && b.orders[team][d].posture != borscht_core::Posture::Reserve
                    && tick > 0
                {
                    *mark = Some(tick);
                }
            }
        }
        if tick % every != 0 {
            continue;
        }
        for (team, name) in ["red", "blue"].iter().enumerate() {
            // Mean distance between division centroids, against the field. This
            // is the number that says whether the army is still an army of
            // bodies or has melted into one crowd.
            let s = &b.divisions[team];
            let (mut sum, mut pairs) = (0.0f32, 0u32);
            for a in 0..divisions {
                for c in a + 1..divisions {
                    if s[a].men == 0 || s[c].men == 0 {
                        continue;
                    }
                    let (dx, dy) = (s[a].x - s[c].x, s[a].y - s[c].y);
                    sum += (dx * dx + dy * dy).sqrt();
                    pairs += 1;
                }
            }
            let spread = sum / pairs.max(1) as f32 / b.field_size();
            let postures: Vec<&str> = (0..divisions)
                .map(|d| &b.orders[team][d].posture.name()[..4])
                .collect();
            let sectors: Vec<String> = (0..divisions)
                .map(|d| b.orders[team][d].sector.to_string())
                .collect();
            let distinct: std::collections::HashSet<u8> =
                (0..divisions).map(|d| b.orders[team][d].sector).collect();
            println!(
                "{tick:>6} {name:>5} {spread:>7.3} {:>28} {:>3} of {}",
                postures.join(","),
                distinct.len(),
                divisions
            );
            let _ = sectors;
        }
    }

    println!("
  RESERVES");
    for (team, name) in ["red", "blue"].iter().enumerate() {
        let when: Vec<String> = committed[team]
            .iter()
            .take(divisions)
            .map(|m| m.map_or("never".to_string(), |t| t.to_string()))
            .collect();
        println!("    {name:<5} committed at {}", when.join(", "));
    }
}

// ------------------------------------------------------------------- nerve --

/// Where a man's nerve is actually going, term by term.
///
/// The morale rule is six terms pulling against each other and the only visible
/// output is whether the army broke, which is not enough to tell a rule that is
/// wrong from one that is merely mistuned. This prints the mean of each term
/// per tick, split by whether the man is in contact with the enemy -- because
/// the whole point of giving the formation depth was that those two groups
/// should no longer be the same group.
fn nerve(args: &Args) {
    let cfg = build_config(args.musters.first().copied(), &args.overrides);
    let mut b = Battle::new(cfg, args.seed);
    let every = (args.ticks / 20).max(1);

    println!(
        "{:>6} {:>5} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "tick", "where", "men", "odds", "cohes", "ascend", "shock", "panic", "wound", "net/tick"
    );
    for tick in 0..args.ticks {
        b.tick();
        if tick % every != 0 {
            continue;
        }
        // [front, rear] x [terms]
        let mut sum = [[0.0f64; 6]; 2];
        let mut men = [0u32; 2];
        let mut worst = [Vec::new(), Vec::new()];
        let mut nerve_now: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        // Where men are actually breaking, which is the question the mean of
        // the delta cannot answer.
        let mut broke = [0u32; 2];
        for i in 0..b.army.len() {
            if !b.army.alive(i) || b.army.routing(i) {
                continue;
            }
            let team = b.army.team[i] as usize;
            let a = b.archetypes[team][b.army.kind[i] as usize];
            let cell = b.grid.units.cell_of[i] as usize;
            let in_contact =
                b.grid.count[borscht_core::grid::foe(b.army.team[i])][cell] > 0.0;
            let p = borscht_core::morale::pressure_on(&b.army, &b.grid, &b.cfg, i, a.hp, b.host[team]);
            let g = usize::from(!in_contact);
            let c = &b.cfg;
            let terms = [
                (c.morale_odds * (p.odds - 0.5) * 2.0) as f64,
                (c.morale_cohesion * p.cohesion) as f64,
                (c.morale_ascendancy * p.ascendancy) as f64,
                (-c.morale_shock * p.losses) as f64,
                (-c.morale_panic * p.routing) as f64,
                (-c.morale_wound * p.hurt) as f64,
            ];
            for (slot, v) in sum[g].iter_mut().zip(terms) {
                *slot += v;
            }
            men[g] += 1;
            worst[g].push(terms.iter().sum::<f64>());
            nerve_now[g].push(b.army.morale[i]);
            if b.army.morale[i] < a.nerve + 0.05 {
                broke[g] += 1;
            }
        }
        for (g, name) in ["front", "rear"].iter().enumerate() {
            let n = men[g].max(1) as f64;
            let t: Vec<f64> = sum[g].iter().map(|v| v / n).collect();
            println!(
                "{tick:>6} {name:>5} {:>7} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>8.4}",
                men[g],
                t[0],
                t[1],
                t[2],
                t[3],
                t[4],
                t[5],
                t.iter().sum::<f64>()
            );
            // The mean hides the men it is actually happening to. A rule whose
            // average is comfortably positive can still be sending a tenth of
            // the army over the edge.
            let w = &mut worst[g];
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pick = |q: f64| w.get(((w.len() as f64 * q) as usize).min(w.len().saturating_sub(1)));
            let m = &mut nerve_now[g];
            m.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mpick = |q: f64| {
                m.get(((m.len() as f64 * q) as usize).min(m.len().saturating_sub(1)))
                    .copied()
                    .unwrap_or(0.0)
            };
            if let (Some(p1), Some(p10)) = (pick(0.01), pick(0.10)) {
                println!(
                    "{:>12} delta worst 1% {p1:>8.4} 10% {p10:>8.4} | nerve 10% {:>5.2} 50% {:>5.2} | broke here {}",
                    "",
                    mpick(0.10),
                    mpick(0.50),
                    broke[g]
                );
            }
        }
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
            "ticks={} red={} blue={} red_holding={} blue_holding={} red_started={} blue_started={} red_killed={} blue_killed={} red_ground={:.4} blue_ground={:.4} red_slope={:.5} blue_slope={:.5} contact={} winner={winner}",
            b.tick, end[0], end[1],
            b.stats.red_holding, b.stats.blue_holding,
            started[0], started[1],
            b.stats.red_killed, b.stats.blue_killed,
            ground[0], ground[1],
            b.counters.blow_slope[0] / b.counters.blows[0].max(1) as f32,
            b.counters.blow_slope[1] / b.counters.blows[1].max(1) as f32,
            contact_at.map_or(-1i64, |t| t as i64)
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
            "    peak running at once {peak_routing}, rallied in the field {}, re-formed at the rear {}",
            c.rallied,
            c.regrouped[0] + c.regrouped[1]
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
