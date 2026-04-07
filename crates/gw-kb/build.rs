//! Bake an rpath to libpython into the binary so we don't need
//! LD_LIBRARY_PATH at runtime. This is necessary when the Python
//! interpreter (e.g. uv-managed standalone Python) lives outside
//! the system loader path.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // Query sysconfig once for everything we need
    let script = "import sys, sysconfig; \
                  print(sysconfig.get_config_var('LIBDIR') or ''); \
                  print(sys.base_prefix); \
                  print(sys.prefix)";

    let output = Command::new(&python).args(["-c", script]).output();

    let stdout = match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        _ => {
            println!(
                "cargo:warning=gw-kb: could not query python sysconfig from {} — \
                 you may need LD_LIBRARY_PATH and PYTHONHOME at runtime",
                python
            );
            return;
        }
    };

    let lines: Vec<&str> = stdout.lines().collect();
    let libdir = lines.first().copied().unwrap_or("").trim();
    let base_prefix = lines.get(1).copied().unwrap_or("").trim();
    let prefix = lines.get(2).copied().unwrap_or("").trim();

    // Bake an rpath so the loader finds libpython without LD_LIBRARY_PATH
    if !libdir.is_empty() && libdir != "None" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir);
    }

    // Bake PYTHONHOME so uv-managed standalone Python finds its stdlib.
    // base_prefix points at the underlying CPython install (where stdlib lives);
    // prefix points at the venv. PYTHONHOME wants base_prefix.
    if !base_prefix.is_empty() {
        println!("cargo:rustc-env=GW_KB_PYTHONHOME={}", base_prefix);
    }
    if !prefix.is_empty() {
        println!("cargo:rustc-env=GW_KB_PYTHON_PREFIX={}", prefix);
    }
}
