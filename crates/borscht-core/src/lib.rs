pub mod army;
pub mod battle;
pub mod brain;
pub mod color;
pub mod combat;
pub mod commander;
pub mod config;
pub mod fastmath;
pub mod grid;
pub mod morale;
pub mod rng;
pub mod stats;
pub mod terrain;

pub use battle::{Battle, ColorMode, Outcome};
pub use commander::{Order, Posture};
pub use config::Config;
