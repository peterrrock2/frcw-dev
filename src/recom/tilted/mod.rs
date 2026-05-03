//! Tilted-run ReCom optimizer.
//!
//! [`core`] holds the shared engine, generic in an [`AcceptanceRule`].
//! [`fixed`] and [`metropolis`] each provide a concrete rule type.

pub mod core;
pub mod fixed;
pub mod metropolis;

pub use core::{
    multi_tilted_runs, multi_tilted_runs_incremental, multi_tilted_runs_incremental_with_writer,
    multi_tilted_runs_with_writer, AcceptanceRule,
};
pub use fixed::FixedAcceptance;
pub use metropolis::MetropolisAcceptance;
