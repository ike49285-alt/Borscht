//! What a person thought of a battle they watched, and what the trainer does
//! with it.
//!
//! # Why a verdict steers rather than scores
//!
//! The obvious thing to do with "this commander played badly" is to subtract
//! something from that commander's score. It does not work, and not for a
//! subtle reason: a verdict attaches to the weights that were *shipped*, and no
//! candidate in a later generation is those weights any more. There is nothing
//! left to attach the penalty to.
//!
//! What a verdict does identify exactly is a **situation** — a seed and a
//! configuration in which play was watched and judged. Situations outlive the
//! commander that was in them, and choosing which ones a search spends its
//! evaluations on is the one real piece of leverage a person has over a process
//! that can already generate infinite samples of its own choosing. So:
//!
//! - a **badly fought** battle puts its seed in the training set, weighted up,
//!   and the search is made to work on the ground somebody found wanting;
//! - a **well fought** battle puts its seed in a held-out set the final report
//!   also covers, so a champion that lifts the average by wrecking play someone
//!   liked is visible rather than silent.
//!
//! # What this cannot do
//!
//! A handful of verdicts is a handful of seeds, and a search told to concentrate
//! on six situations will overfit to them. That is why the flagged seeds and the
//! approved seeds are both reported separately at the end rather than folded
//! into one number: if a champion improves on the flagged seeds and regresses
//! everywhere else, the honest reading is that the verdicts were too few to
//! train on, and the report has to make that readable rather than hide it.

use crate::matchlog::field;
use borscht_core::Config;
use std::fs;
use std::path::Path;

/// One judgement, as the page records it.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub seed: u64,
    /// True when the play was found wanting: this is ground to work on.
    pub badly: bool,
    pub side: String,
    pub note: String,
    pub commander: String,
    /// The muster the battle was watched at. A verdict passed on ten thousand a
    /// side says nothing reliable about the same seed at fifty thousand — the
    /// two are measurably different battles — so this is kept in order to be
    /// checked rather than assumed to match.
    pub units_per_side: u32,
}

impl Verdict {
    pub fn from_json(text: &str) -> Option<Verdict> {
        let verdict = field(text, "verdict")?;
        let badly = match verdict {
            v if v.contains("badly") => true,
            v if v.contains("well") => false,
            _ => return None,
        };
        Some(Verdict {
            seed: field(text, "seed")?.parse().ok()?,
            badly,
            side: field(text, "side").unwrap_or("").to_string(),
            note: field(text, "note").unwrap_or("").to_string(),
            commander: field(text, "commander").unwrap_or("unknown").to_string(),
            // Nested inside `overrides` in the page's document, but the reader
            // is flat and the key is unique in the object, so it is found either
            // way. A document without it is not rejected: the seed and the
            // judgement are the parts that matter, and a missing muster is
            // reported as unknown rather than guessed at.
            units_per_side: field(text, "units_per_side")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        })
    }
}

/// Everything the trainer was told, read off disk.
#[derive(Default, Debug)]
pub struct Judged {
    pub flagged: Vec<Verdict>,
    pub approved: Vec<Verdict>,
}

impl Judged {
    pub fn is_empty(&self) -> bool {
        self.flagged.is_empty() && self.approved.is_empty()
    }

    /// The seeds the search should spend extra evaluations on.
    ///
    /// Repeated `weight` times, which is what "weighted up" means to
    /// [`crate::train::evaluate`]: it plays every seed it is given, so a seed
    /// listed twice counts twice.
    pub fn training_seeds(&self, weight: usize) -> Vec<u64> {
        let mut seeds = Vec::new();
        for v in &self.flagged {
            for _ in 0..weight.max(1) {
                seeds.push(v.seed);
            }
        }
        seeds
    }

    pub fn approved_seeds(&self) -> Vec<u64> {
        let mut seeds: Vec<u64> = self.approved.iter().map(|v| v.seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        seeds
    }

    pub fn flagged_seeds(&self) -> Vec<u64> {
        let mut seeds: Vec<u64> = self.flagged.iter().map(|v| v.seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        seeds
    }
}

/// Read verdicts from a file of one JSON object per line, or from a directory
/// of one JSON object per file.
///
/// Both, because both are what comes back: the page writes one document per
/// verdict, and pulling them down saves either a directory of documents or a
/// single stream of them depending on how it is asked.
pub fn read(path: &Path) -> Result<Judged, String> {
    let mut documents: Vec<String> = Vec::new();
    if path.is_dir() {
        let entries =
            fs::read_dir(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "json") {
                if let Ok(text) = fs::read_to_string(&p) {
                    documents.push(text.replace('\n', " "));
                }
            }
        }
    } else {
        let text =
            fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        documents.extend(text.lines().filter(|l| l.contains('{')).map(String::from));
    }

    let mut judged = Judged::default();
    let mut unreadable = 0;
    for doc in &documents {
        match Verdict::from_json(doc) {
            Some(v) if v.badly => judged.flagged.push(v),
            Some(v) => judged.approved.push(v),
            // Counted rather than ignored. A verdict file that is silently
            // half-read would make a run look like it acted on judgements it
            // never saw.
            None => unreadable += 1,
        }
    }
    if unreadable > 0 {
        eprintln!(
            "warning: {unreadable} of {} documents in {} carried no readable verdict",
            documents.len(),
            path.display()
        );
    }
    Ok(judged)
}

/// Say out loud what was read, and where it does not match the run.
///
/// The muster check is not pedantry. Conclusions in this simulator do not
/// transfer cleanly between musters — the same seeds give measurably different
/// results at twelve thousand and at twenty thousand — so a verdict passed at
/// one muster and trained on at another is about different ground than the
/// person watching thought they were judging.
pub fn report(judged: &Judged, cfg: &Config, playing: &str) {
    println!(
        "verdicts: {} battles found wanting, {} found well fought",
        judged.flagged.len(),
        judged.approved.len()
    );
    for v in judged.flagged.iter().chain(judged.approved.iter()) {
        let mark = if v.badly { "badly " } else { "well  " };
        let note = if v.note.is_empty() {
            String::new()
        } else {
            format!("  \"{}\"", v.note)
        };
        println!("  {mark} {} on seed {:<10}{note}", v.side, v.seed);
    }

    let mismatched = judged
        .flagged
        .iter()
        .chain(judged.approved.iter())
        .filter(|v| v.units_per_side != 0 && v.units_per_side != cfg.units_per_side)
        .count();
    if mismatched > 0 {
        println!(
            "  warning: {mismatched} of these were watched at a different muster than this run \
             fights at ({} a side).",
        cfg.units_per_side
        );
        println!(
            "  the seed names different ground at a different muster, so those judgements are \
             about a battle this run will not fight."
        );
    }

    // Whose play was being judged. A verdict passed on a commander this build no
    // longer ships is about play nobody can reproduce from here — the seed still
    // names ground worth working on, which is why it is a note rather than a
    // rejection, but the judgement itself was about a different commander.
    let elsewhere = judged
        .flagged
        .iter()
        .chain(judged.approved.iter())
        .filter(|v| v.commander != "unknown" && v.commander != playing)
        .count();
    if elsewhere > 0 {
        println!(
            "  note: {elsewhere} of these judged a commander other than the one this build              ships ({playing}). The ground still counts; the judgement was about different play."
        );
    }

    // The count is the thing that decides whether any of this means anything,
    // so it is said before the run rather than defended after it.
    if judged.flagged.len() < 8 {
        println!(
            "  {} flagged seed{} few enough to overfit to. The final report covers them \
             separately for exactly that reason.\n",
            judged.flagged.len(),
            if judged.flagged.len() == 1 { " is" } else { "s are" }
        );
    } else {
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE_ROW: &str = r#"{"verdict":"badly fought","side":"blue","note":"left flank never closed","commander":"d500043a350fc906","seed":7,"scale":20000,"overrides":{"units_per_side":10000,"max_units":21016,"field_size":450,"grid_dim":128},"tick":2112,"outcome":"both lines holding","red":7529,"blue":7611,"at":"2026-09-02T18:28:59.087Z"}"#;

    /// The shape here is a document the page actually wrote, copied out of a
    /// run of `tools/check-verdict.mjs`. The reader and the writer live in
    /// different languages and cannot share a definition, so what stands
    /// between them is this.
    #[test]
    fn a_document_the_page_writes_is_read_as_the_page_meant_it() {
        let v = Verdict::from_json(PAGE_ROW).expect("the page's own document is unreadable");
        assert!(v.badly);
        assert_eq!(v.seed, 7);
        assert_eq!(v.side, "blue");
        assert_eq!(v.note, "left flank never closed");
        assert_eq!(v.commander, "d500043a350fc906");
        // Reached through the nested `overrides` object, which is the muster the
        // battle was actually watched at rather than the page's scale setting.
        assert_eq!(v.units_per_side, 10000);
    }

    #[test]
    fn well_and_badly_go_to_different_piles() {
        let good = PAGE_ROW.replace("badly fought", "well fought");
        assert!(!Verdict::from_json(&good).unwrap().badly);
        assert!(Verdict::from_json(PAGE_ROW).unwrap().badly);
        // Anything that is neither is not quietly filed as one of them.
        assert!(Verdict::from_json(&PAGE_ROW.replace("badly fought", "meh")).is_none());
    }

    #[test]
    fn a_flagged_seed_is_weighted_up_and_an_approved_one_is_not() {
        let mut judged = Judged::default();
        judged.flagged.push(Verdict::from_json(PAGE_ROW).unwrap());
        judged
            .approved
            .push(Verdict::from_json(&PAGE_ROW.replace("badly fought", "well fought")).unwrap());
        assert_eq!(judged.training_seeds(3), vec![7, 7, 7]);
        assert_eq!(judged.approved_seeds(), vec![7]);
    }

    #[test]
    fn a_file_of_documents_and_a_directory_of_them_read_the_same() {
        let base = std::env::temp_dir().join(format!("borscht-verdicts-{}", std::process::id()));
        let dir = base.join("dir");
        fs::create_dir_all(&dir).unwrap();
        let good = PAGE_ROW.replace("badly fought", "well fought");

        let stream = base.join("verdicts.jsonl");
        fs::write(&stream, format!("{PAGE_ROW}\n{good}\n")).unwrap();
        fs::write(dir.join("a.json"), PAGE_ROW).unwrap();
        // Pretty-printed across lines, which is how a saved document arrives.
        fs::write(dir.join("b.json"), good.replace(',', ",\n")).unwrap();

        let from_file = read(&stream).unwrap();
        let from_dir = read(&dir).unwrap();
        assert_eq!(from_file.flagged.len(), 1);
        assert_eq!(from_file.approved.len(), 1);
        assert_eq!(from_dir.flagged.len(), from_file.flagged.len());
        assert_eq!(from_dir.approved.len(), from_file.approved.len());
        assert_eq!(from_dir.flagged_seeds(), from_file.flagged_seeds());

        let _ = fs::remove_dir_all(&base);
    }
}
