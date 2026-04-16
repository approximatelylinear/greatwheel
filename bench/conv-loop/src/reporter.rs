use std::collections::BTreeMap;

use crate::assertions::AssertionResult;
use crate::scenario::Scenario;

pub fn print_summary(results: &[(Scenario, Vec<AssertionResult>)]) {
    let mut by_capability: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut total_pass = 0;
    let mut total_fail = 0;

    for (scenario, assertions) in results {
        let entry = by_capability.entry(&scenario.capability).or_insert((0, 0));
        for a in assertions {
            if a.passed {
                entry.0 += 1;
                total_pass += 1;
            } else {
                entry.1 += 1;
                total_fail += 1;
            }
        }
    }

    println!("╭──────────────────────────┬───────┬───────╮");
    println!("│ Capability               │  Pass │  Fail │");
    println!("├──────────────────────────┼───────┼───────┤");
    for (cap, (pass, fail)) in &by_capability {
        println!("│ {cap:<24} │ {pass:>5} │ {fail:>5} │");
    }
    println!("├──────────────────────────┼───────┼───────┤");
    println!(
        "│ {:<24} │ {:>5} │ {:>5} │",
        "TOTAL", total_pass, total_fail
    );
    println!("╰──────────────────────────┴───────┴───────╯");
}
