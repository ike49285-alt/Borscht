//! A battle as a timeline, rather than as a verdict.
//!
//! The sweep answers "who won"; this answers "what happened", which is a
//! different question and the one that catches a battle that is not being
//! fought. It found a real bug: two armies drawn up four hundred and fifty
//! units apart were five hundred and sixty apart two hundred ticks later,
//! both of them backing away, and every man who died was shot by an engine.
//! No aggregate says that. A column of the distance between the two centres
//! of mass, falling or not falling, says it in one glance.
//!
//!     cargo run --release -p borscht-core --example probe -- 100000
//!     cargo run --release -p borscht-core --example probe -- 8000 guard_recall=0

use borscht_core::battle::Battle;
use borscht_core::config::Config;

/// Each side's centre of mass along the axis the armies face down.
fn centres(b: &Battle) -> [f64; 2] {
    let (mut sum, mut men) = ([0.0f64; 2], [0.0f64; 2]);
    for i in 0..b.army.len() {
        if b.army.alive(i) {
            let t = b.army.team[i] as usize;
            sum[t] += b.army.x[i] as f64;
            men[t] += 1.0;
        }
    }
    [sum[0] / men[0].max(1.0), sum[1] / men[1].max(1.0)]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let muster: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let mut cfg = Config::for_muster(muster);
    for set in args.iter().skip(2) {
        let (k, v) = set.split_once('=').expect("expected name=value");
        let id = Config::param_id(k).unwrap_or_else(|| panic!("no parameter {k:?}"));
        cfg.set_param(id, v.parse().expect("expected a number"));
    }
    cfg.sanitize();

    let mut b = Battle::new(cfg, 7);
    println!(
        "{} men a side on a field of {:.0}, drawn up {:.0} apart",
        cfg.units_per_side,
        cfg.field_size,
        cfg.field_size * cfg.deploy_separation
    );
    println!("  tick   red_x  blue_x   apart     red    blue    hand  arrows");
    let (mut was_hand, mut was_shot) = (0u32, 0u32);
    for step in 1..=40 {
        b.tick_many(200);
        let at = centres(&b);
        let alive = b.army.muster();
        let shot: u32 = (0..2).map(|t| b.counters.shot_kills[t]).sum();
        let dead: u32 = (0..2).map(|t| b.counters.killed_fighting[t]).sum();
        let hand = dead - shot;
        println!(
            "{:6}  {:6.0}  {:6.0}  {:6.0}  {:6}  {:6}  {:6}  {:6}",
            step * 200,
            at[0],
            at[1],
            (at[1] - at[0]).abs(),
            alive[0],
            alive[1],
            hand - was_hand,
            shot - was_shot,
        );
        (was_hand, was_shot) = (hand, shot);
        if b.decided() {
            println!("decided at tick {}", b.tick);
            break;
        }
    }
}
