use std::path::{Path, PathBuf};

/// Filesystem locations the bench dashboard reads from.
///
/// Constructed from a workspace root, or supplied directly in tests.
#[derive(Debug, Clone)]
pub struct BenchPaths {
    /// Root of `runs/<slug>/run_*.json`.
    pub runs_dir: PathBuf,
    /// Root of `bench/browsecomp/configs/<slug>.toml`.
    pub configs_dir: PathBuf,
    /// `vendor/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl`.
    pub gold_jsonl: PathBuf,
}

impl BenchPaths {
    pub fn new(runs_dir: PathBuf, configs_dir: PathBuf, gold_jsonl: PathBuf) -> Self {
        Self {
            runs_dir,
            configs_dir,
            gold_jsonl,
        }
    }

    /// Standard layout under a Greatwheel checkout root.
    pub fn from_workspace_root(root: &Path) -> Self {
        Self {
            runs_dir: root.join("runs"),
            configs_dir: root.join("bench/browsecomp/configs"),
            gold_jsonl: root.join("vendor/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl"),
        }
    }
}
