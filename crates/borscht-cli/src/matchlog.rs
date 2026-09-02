//! A record of every battle a training run plays, and a way back to any of them.
//!
//! A thirty-generation run fights about twenty thousand battles and prints
//! thirty lines about them. Everything else — which pairings on which ground,
//! what happened, whether the run's conclusion rests on six lucky seeds — is
//! thrown away the moment it is summarised.
//!
//! It does not have to be. A battle here is exactly reproducible from its
//! configuration, its seed, its stream and the two commanders' weights, so a row
//! holding those is not a summary of a battle: it *is* the battle, and
//! [`replay`] rebuilds it tick for tick. The whole record of a run is then a few
//! megabytes and any line of it can be watched.
//!
//! # Why weights live beside the log rather than in it
//!
//! Three hundred floats per row times twenty thousand rows is most of a
//! gigabyte of mostly-repeated numbers: a generation's twelve candidates play
//! hundreds of battles each. Rows carry a content hash instead and each distinct
//! commander is written once, which keeps the log small without making any row
//! less resolvable.

use borscht_core::brain::{Net, LEN};
use borscht_core::{Battle, Config};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// What a commander is called, here and everywhere else.
///
/// The name comes from the core rather than from this file so that a log row, a
/// replay and the build playing in a browser all say the same thing about the
/// same commander — which is what lets a verdict passed on a watched battle be
/// checked against the commander that was actually on the field.
pub fn name_of(net: &Net) -> String {
    format!("{:016x}", net.fingerprint())
}

/// One battle, in the form that can rebuild it.
#[derive(Clone, Debug)]
pub struct Row {
    pub id: u64,
    /// What the run was doing when it played this: a generation number, the
    /// noise floor, the final verdict.
    pub phase: String,
    pub seed: u64,
    pub stream: u64,
    pub red: String,
    pub blue: String,
    pub ticks: u64,
    pub alive: [u32; 2],
    pub holding: [f32; 2],
    pub killed: [f32; 2],
    /// What this battle was worth to red, by the trainer's own scoring.
    pub margin: f32,
}

impl Row {
    fn to_json(&self) -> String {
        let mut s = String::with_capacity(256);
        let _ = write!(
            s,
            r#"{{"id":{},"phase":"{}","seed":{},"stream":{},"red":"{}","blue":"{}","ticks":{},"#,
            self.id, self.phase, self.seed, self.stream, self.red, self.blue, self.ticks
        );
        let _ = write!(
            s,
            r#""red_alive":{},"blue_alive":{},"red_holding":{},"blue_holding":{},"#,
            self.alive[0], self.alive[1], self.holding[0], self.holding[1]
        );
        let _ = write!(
            s,
            r#""red_killed":{},"blue_killed":{},"margin":{}}}"#,
            self.killed[0], self.killed[1], self.margin
        );
        s
    }

    /// Parse a row back. Deliberately a small hand-rolled reader rather than a
    /// dependency: the writer above is the only thing that produces these, the
    /// shape is flat, and the core of this project carries no dependencies at
    /// all.
    pub fn from_json(line: &str) -> Option<Row> {
        let field = |key: &str| -> Option<&str> {
            let at = line.find(&format!("\"{key}\":"))? + key.len() + 3;
            let rest = &line[at..];
            let end = rest.find([',', '}'])?;
            Some(rest[..end].trim_matches('"'))
        };
        Some(Row {
            id: field("id")?.parse().ok()?,
            phase: field("phase")?.to_string(),
            seed: field("seed")?.parse().ok()?,
            stream: field("stream")?.parse().ok()?,
            red: field("red")?.to_string(),
            blue: field("blue")?.to_string(),
            ticks: field("ticks")?.parse().ok()?,
            alive: [
                field("red_alive")?.parse().ok()?,
                field("blue_alive")?.parse().ok()?,
            ],
            holding: [
                field("red_holding")?.parse().ok()?,
                field("blue_holding")?.parse().ok()?,
            ],
            killed: [
                field("red_killed")?.parse().ok()?,
                field("blue_killed")?.parse().ok()?,
            ],
            margin: field("margin")?.parse().ok()?,
        })
    }
}

/// Where a run's record is kept.
pub struct Log {
    dir: PathBuf,
    matches: fs::File,
    /// Weight sets already written, so each is stored once however many battles
    /// it fights.
    seen: HashSet<String>,
    next: u64,
}

impl Log {
    pub fn create(dir: &Path, cfg: &Config, ticks: u32) -> std::io::Result<Log> {
        fs::create_dir_all(dir.join("weights"))?;
        // The configuration the whole run was fought under, so a replay a month
        // later is not quietly fought under different parameters.
        let mut params = String::from("{\n");
        for (i, p) in borscht_core::config::PARAMS.iter().enumerate() {
            let _ = writeln!(
                params,
                "  \"{}\": {}{}",
                p.name,
                cfg.get_param(i as u32),
                if i + 1 == borscht_core::config::PARAMS.len() {
                    ""
                } else {
                    ","
                }
            );
        }
        let _ = writeln!(params, "}}");
        fs::write(dir.join("config.json"), params)?;
        fs::write(dir.join("ticks.txt"), ticks.to_string())?;

        Ok(Log {
            dir: dir.to_path_buf(),
            matches: fs::File::create(dir.join("matches.jsonl"))?,
            seen: HashSet::new(),
            next: 0,
        })
    }

    /// Store a commander if it is not already stored, and return its name.
    pub fn remember(&mut self, net: &Net) -> String {
        let id = name_of(net);
        if self.seen.insert(id.clone()) {
            let body: Vec<String> = net.w.iter().map(|v| format!("{v:?}")).collect();
            let _ = fs::write(
                self.dir.join("weights").join(format!("{id}.json")),
                format!("[{}]", body.join(",")),
            );
        }
        id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        phase: &str,
        seed: u64,
        stream: u64,
        red: &Net,
        blue: &Net,
        b: &Battle,
        margin: f32,
    ) -> u64 {
        let id = self.next;
        self.next += 1;
        let row = Row {
            id,
            phase: phase.to_string(),
            seed,
            stream,
            red: self.remember(red),
            blue: self.remember(blue),
            ticks: b.tick,
            alive: b.army.muster(),
            holding: [b.stats.red_holding, b.stats.blue_holding],
            killed: [b.stats.red_killed, b.stats.blue_killed],
            margin,
        };
        let _ = writeln!(self.matches, "{}", row.to_json());
        id
    }

    pub fn matches_written(&self) -> u64 {
        self.next
    }
}

/// Everything needed to fight a logged battle again.
pub struct Recorded {
    pub row: Row,
    pub cfg: Config,
    pub red: Net,
    pub blue: Net,
    pub ticks: u32,
}

fn load_net(dir: &Path, id: &str) -> Option<Net> {
    let text = fs::read_to_string(dir.join("weights").join(format!("{id}.json"))).ok()?;
    let mut net = Net::zeroed();
    let values: Vec<f32> = text
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect();
    if values.len() != LEN {
        return None;
    }
    net.w.copy_from_slice(&values);
    Some(net)
}

/// Find a logged battle and everything needed to fight it again.
pub fn find(dir: &Path, id: u64) -> Result<Recorded, String> {
    let text = fs::read_to_string(dir.join("matches.jsonl"))
        .map_err(|e| format!("cannot read {}/matches.jsonl: {e}", dir.display()))?;
    let row = text
        .lines()
        .filter_map(Row::from_json)
        .find(|r| r.id == id)
        .ok_or_else(|| format!("no match {id} in {}", dir.display()))?;

    let params = fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("cannot read the run's config: {e}"))?;
    let mut cfg = Config::default();
    for line in params.lines() {
        let Some((name, value)) = line.trim().trim_end_matches(',').split_once(':') else {
            continue;
        };
        let name = name.trim().trim_matches('"');
        let Ok(value) = value.trim().parse::<f32>() else {
            continue;
        };
        if let Some(param) = Config::param_id(name) {
            cfg.set_param(param, value);
        }
    }
    cfg.sanitize();

    let ticks = fs::read_to_string(dir.join("ticks.txt"))
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(20_000);

    let red = load_net(dir, &row.red).ok_or_else(|| format!("missing weights {}", row.red))?;
    let blue = load_net(dir, &row.blue).ok_or_else(|| format!("missing weights {}", row.blue))?;
    Ok(Recorded {
        row,
        cfg,
        red,
        blue,
        ticks,
    })
}

/// Fight a logged battle again, exactly.
///
/// The property the whole record rests on: the same configuration, seed, stream
/// and weights give the same battle byte for byte. If this ever stops being
/// true the log is decoration, which is why it is asserted in a test rather
/// than assumed here.
/// `watch` sees the field after every tick, which is what makes a logged row
/// watchable rather than merely countable: a frame can be drawn, or a statistic
/// traced, at any point of a battle fought weeks ago. It gets the battle mutably
/// only because drawing fills a render buffer — advancing it, or altering
/// anything a tick reads, would make this a different battle from the one
/// recorded, which the caller then sees reported as a disagreement with the row.
pub fn replay(r: &Recorded, mut watch: impl FnMut(&mut Battle)) -> Battle {
    let mut b = Battle::new(r.cfg, r.row.seed);
    b.doctrine = [r.red, r.blue];
    b.restream(r.row.stream);
    for _ in 0..r.ticks {
        b.tick();
        watch(&mut b);
        if b.decided() {
            break;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_survives_being_written_and_read() {
        let row = Row {
            id: 4711,
            phase: "gen3".into(),
            seed: 90_210,
            stream: 7,
            red: "abc123".into(),
            blue: "def456".into(),
            ticks: 8484,
            alive: [1524, 0],
            holding: [1524.0, 0.0],
            killed: [8472.0, 9996.0],
            margin: -0.375,
        };
        let back = Row::from_json(&row.to_json()).expect("a row this file wrote is unreadable");
        assert_eq!(back.id, row.id);
        assert_eq!(back.phase, row.phase);
        assert_eq!(back.seed, row.seed);
        assert_eq!(back.stream, row.stream);
        assert_eq!(back.red, row.red);
        assert_eq!(back.ticks, row.ticks);
        assert_eq!(back.alive, row.alive);
        assert_eq!(back.killed, row.killed);
        assert!((back.margin - row.margin).abs() < 1e-6);
    }

    /// The property the record rests on: a row is not a summary of a battle, it
    /// *is* the battle. Write one, read it back off disk, fight it again, and
    /// the outcome must match to the tick and to the man. If this ever fails
    /// the log is decoration and every replay is a different battle wearing the
    /// same id.
    #[test]
    fn a_replayed_row_is_the_same_battle() {
        let dir = std::env::temp_dir().join(format!(
            "borscht-matchlog-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        // Small enough to fight twice inside a unit test, big enough that the
        // battle is not decided by two men bumping into each other — and far
        // enough from the default muster that a config that failed to round-trip
        // would be caught here rather than passing by luck.
        let mut cfg = Config::for_muster(1_200);
        cfg.sanitize();
        let ticks = 4_000u32;

        let mut rng = borscht_core::rng::Rng::new(11, 4);
        let red = Net::doctrine();
        let mut blue = Net::doctrine();
        blue.mutate(&mut rng, 0.4);

        let (seed, stream) = (0xB0_1234u64, 3u64);
        let mut played = Battle::new(cfg, seed);
        played.doctrine = [red, blue];
        played.restream(stream);
        for _ in 0..ticks {
            played.tick();
            if played.decided() {
                break;
            }
        }

        let mut log = Log::create(&dir, &cfg, ticks).expect("cannot open a log");
        let id = log.record("test", seed, stream, &red, &blue, &played, 0.25);
        drop(log);

        let found = find(&dir, id).expect("the row this test just wrote is unreadable");
        let again = replay(&found, |_| {});

        assert_eq!(
            again.tick, found.row.ticks,
            "a replay ran for a different number of ticks than the row records"
        );
        assert_eq!(
            again.army.muster(),
            found.row.alive,
            "a replay ended with a different number of men standing"
        );
        // And the row that came back off disk is the battle that was fought,
        // not merely self-consistent with its own replay.
        assert_eq!(found.row.ticks, played.tick);
        assert_eq!(found.row.alive, played.army.muster());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_different_commanders_get_two_different_names() {
        let mut rng = borscht_core::rng::Rng::new(3, 9);
        let a = Net::doctrine();
        let mut b = a;
        b.mutate(&mut rng, 0.1);
        assert_ne!(name_of(&a), name_of(&b));
        assert_eq!(name_of(&a), name_of(&Net::doctrine()));
    }
}
