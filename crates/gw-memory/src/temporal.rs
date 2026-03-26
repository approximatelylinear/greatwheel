//! Temporal query parsing and scoring for Hindsight memory retrieval.
//!
//! Implements the temporal retrieval channel from design-hindsight-memory.md §2.4:
//! - Rule-based temporal parser (§2.4.1, stage 1)
//! - Temporal proximity scoring (§2.4.3)
//! - Recency decay fallback (§2.4.4)

use chrono::{Datelike, DateTime, Duration, NaiveDate, NaiveTime, Utc};

/// A resolved temporal range from a query.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Attempt to parse a temporal expression from a query string.
///
/// Returns `Some(TemporalRange)` if a time expression is detected,
/// `None` otherwise. This is the rule-based stage 1 parser; the LLM
/// fallback (stage 2) is deferred to the hindsight-recall plugin.
pub fn parse_temporal(query: &str, now: DateTime<Utc>) -> Option<TemporalRange> {
    let lower = query.to_lowercase();

    // "yesterday"
    if lower.contains("yesterday") {
        let start = (now - Duration::days(1)).date_naive();
        return Some(day_range(start));
    }

    // "today"
    if lower.contains("today") {
        return Some(day_range(now.date_naive()));
    }

    // "last week" — simple 7-day window ending 7 days ago
    if lower.contains("last week") {
        return Some(TemporalRange {
            start: now - Duration::weeks(2),
            end: now - Duration::weeks(1),
        });
    }

    // "last N days" / "past N days"
    if let Some(n) = parse_last_n_days(&lower) {
        return Some(TemporalRange {
            start: now - Duration::days(n),
            end: now,
        });
    }

    // "last month"
    if lower.contains("last month") {
        let today = now.date_naive();
        let this_month_start = NaiveDate::from_ymd_opt(today.year(), today.month(), 1)?;
        let last_month_start = if today.month() == 1 {
            NaiveDate::from_ymd_opt(today.year() - 1, 12, 1)?
        } else {
            NaiveDate::from_ymd_opt(today.year(), today.month() - 1, 1)?
        };
        return Some(TemporalRange {
            start: last_month_start.and_time(NaiveTime::MIN).and_utc(),
            end: this_month_start.and_time(NaiveTime::MIN).and_utc(),
        });
    }

    // "in <Month> <Year>" or "<Month> <Year>"
    if let Some(range) = parse_month_year(&lower) {
        return Some(range);
    }

    // "before <date>" / "after <date>" — not a range, skip for now

    None
}

/// Score a memory's temporal proximity to a query range.
///
/// From the paper: `s_temp(Q, f) = 1 - |τ_mid^f - τ_mid^Q| / (Δτ / 2)`
///
/// Returns a score in `[0.0, 1.0]` where 1.0 means the fact is centered
/// in the query range.
pub fn temporal_proximity_score(
    fact_start: DateTime<Utc>,
    fact_end: DateTime<Utc>,
    query: &TemporalRange,
) -> f32 {
    let fact_mid_secs = (fact_start.timestamp() + fact_end.timestamp()) / 2;
    let query_mid_secs = (query.start.timestamp() + query.end.timestamp()) / 2;
    let delta_secs = (query.end.timestamp() - query.start.timestamp()).max(1);
    let dist = (fact_mid_secs - query_mid_secs).unsigned_abs() as f64;
    let half_delta = delta_secs as f64 / 2.0;
    (1.0 - dist / half_delta).max(0.0) as f32
}

/// Recency decay score: `exp(-(now - occurred_at) / σ)`.
///
/// `sigma_days` controls the decay rate (default 7 days in config).
pub fn recency_score(occurred_at: DateTime<Utc>, now: DateTime<Utc>, sigma_days: f64) -> f32 {
    let age_secs = (now - occurred_at).num_seconds().max(0) as f64;
    let sigma_secs = sigma_days * 86400.0;
    (-(age_secs / sigma_secs)).exp() as f32
}

/// Check whether two intervals overlap: `[a_start, a_end] ∩ [b_start, b_end] ≠ ∅`
pub fn intervals_overlap(
    a_start: DateTime<Utc>,
    a_end: DateTime<Utc>,
    b_start: DateTime<Utc>,
    b_end: DateTime<Utc>,
) -> bool {
    a_start <= b_end && a_end >= b_start
}

// ---- Internal helpers ---- //

fn day_range(date: NaiveDate) -> TemporalRange {
    TemporalRange {
        start: date.and_time(NaiveTime::MIN).and_utc(),
        end: date.succ_opt().unwrap_or(date).and_time(NaiveTime::MIN).and_utc(),
    }
}

fn parse_last_n_days(lower: &str) -> Option<i64> {
    // Match "last N days" or "past N days"
    for prefix in ["last ", "past "] {
        if let Some(rest) = lower.strip_prefix(prefix).or_else(|| {
            lower.find(prefix).map(|i| &lower[i + prefix.len()..])
        }) {
            let rest = rest.trim_start();
            if let Some(space) = rest.find(' ') {
                let num_str = &rest[..space];
                if rest[space..].trim_start().starts_with("day") {
                    if let Ok(n) = num_str.parse::<i64>() {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

fn parse_month_year(lower: &str) -> Option<TemporalRange> {
    let months = [
        ("january", 1), ("february", 2), ("march", 3), ("april", 4),
        ("may", 5), ("june", 6), ("july", 7), ("august", 8),
        ("september", 9), ("october", 10), ("november", 11), ("december", 12),
        ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4),
        ("jun", 6), ("jul", 7), ("aug", 8), ("sep", 9),
        ("oct", 10), ("nov", 11), ("dec", 12),
    ];

    for (name, month) in months {
        if let Some(idx) = lower.find(name) {
            // Look for a year after the month name
            let after = &lower[idx + name.len()..];
            let after = after.trim_start().trim_start_matches(',').trim_start();
            if let Some(year_str) = after.split_whitespace().next() {
                let year_str = year_str.trim_end_matches(|c: char| !c.is_ascii_digit());
                if let Ok(year) = year_str.parse::<i32>() {
                    if (1500..=2099).contains(&year) {
                        let start = NaiveDate::from_ymd_opt(year, month, 1)?;
                        let end = if month == 12 {
                            NaiveDate::from_ymd_opt(year + 1, 1, 1)?
                        } else {
                            NaiveDate::from_ymd_opt(year, month + 1, 1)?
                        };
                        return Some(TemporalRange {
                            start: start.and_time(NaiveTime::MIN).and_utc(),
                            end: end.and_time(NaiveTime::MIN).and_utc(),
                        });
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn utc(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 12, 0, 0).unwrap()
    }

    #[test]
    fn parse_yesterday() {
        let now = utc(2026, 3, 26);
        let range = parse_temporal("What happened yesterday?", now).unwrap();
        assert_eq!(range.start.date_naive(), NaiveDate::from_ymd_opt(2026, 3, 25).unwrap());
    }

    #[test]
    fn parse_today() {
        let now = utc(2026, 3, 26);
        let range = parse_temporal("What did I learn today?", now).unwrap();
        assert_eq!(range.start.date_naive(), NaiveDate::from_ymd_opt(2026, 3, 26).unwrap());
    }

    #[test]
    fn parse_last_n_days() {
        let now = utc(2026, 3, 26);
        let range = parse_temporal("Events in the last 3 days", now).unwrap();
        let expected_start = now - Duration::days(3);
        assert!((range.start - expected_start).num_seconds().abs() < 2);
    }

    #[test]
    fn parse_month_year() {
        let now = utc(2026, 3, 26);
        let range = parse_temporal("What happened in January 2026?", now).unwrap();
        assert_eq!(range.start, Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap());
        assert_eq!(range.end, Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap());
    }

    #[test]
    fn parse_last_month() {
        let now = utc(2026, 3, 15);
        let range = parse_temporal("What happened last month?", now).unwrap();
        assert_eq!(range.start, Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap());
        assert_eq!(range.end, Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap());
    }

    #[test]
    fn parse_no_temporal() {
        let now = utc(2026, 3, 26);
        assert!(parse_temporal("Who is Marie Curie?", now).is_none());
        assert!(parse_temporal("What is the capital of France?", now).is_none());
    }

    #[test]
    fn proximity_score_centered() {
        let query = TemporalRange {
            start: utc(2026, 3, 1),
            end: utc(2026, 3, 31),
        };
        // Fact centered in query range
        let score = temporal_proximity_score(utc(2026, 3, 15), utc(2026, 3, 16), &query);
        assert!(score > 0.9, "centered fact should score near 1.0, got {score}");
    }

    #[test]
    fn proximity_score_edge() {
        let query = TemporalRange {
            start: utc(2026, 3, 1),
            end: utc(2026, 3, 31),
        };
        // Fact at the very start of range
        let score = temporal_proximity_score(utc(2026, 3, 1), utc(2026, 3, 1), &query);
        assert!(score < 0.1, "edge fact should score near 0.0, got {score}");
    }

    #[test]
    fn recency_score_recent() {
        let now = utc(2026, 3, 26);
        let score = recency_score(utc(2026, 3, 25), now, 7.0);
        assert!(score > 0.8, "1-day-old memory should score high, got {score}");
    }

    #[test]
    fn recency_score_old() {
        let now = utc(2026, 3, 26);
        let score = recency_score(utc(2026, 1, 1), now, 7.0);
        assert!(score < 0.01, "3-month-old memory should score near 0, got {score}");
    }

    #[test]
    fn intervals_overlap_yes() {
        assert!(intervals_overlap(utc(2026, 3, 1), utc(2026, 3, 15), utc(2026, 3, 10), utc(2026, 3, 20)));
    }

    #[test]
    fn intervals_overlap_no() {
        assert!(!intervals_overlap(utc(2026, 3, 1), utc(2026, 3, 5), utc(2026, 3, 10), utc(2026, 3, 20)));
    }
}
