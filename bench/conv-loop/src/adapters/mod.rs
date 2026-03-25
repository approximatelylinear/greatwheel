pub mod mint;
pub mod tau_bench;

use crate::scenario::Scenario;

/// Trait for converting external benchmark tasks into conv-loop scenarios.
pub trait BenchmarkAdapter {
    /// Load tasks from a file and convert them into scenarios.
    fn load(&self, path: &std::path::Path) -> Result<Vec<Scenario>, String>;

    /// Name of this adapter (for CLI output).
    fn name(&self) -> &str;
}
