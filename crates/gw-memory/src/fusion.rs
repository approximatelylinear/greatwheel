use std::collections::HashMap;

/// A scored result from a single search backend.
#[derive(Debug, Clone)]
pub struct ScoredKey {
    pub key: String,
    pub score: f32,
}

/// Merge multiple ranked result lists using Reciprocal Rank Fusion.
///
/// Each result list should be ordered by relevance (best first).
/// Returns merged list sorted by RRF score (highest first).
pub fn reciprocal_rank_fusion(result_lists: &[Vec<ScoredKey>], rrf_k: usize) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    for results in result_lists {
        for (rank, item) in results.iter().enumerate() {
            *scores.entry(item.key.clone()).or_insert(0.0) +=
                1.0 / (rrf_k as f32 + rank as f32 + 1.0);
        }
    }

    let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_single_list() {
        let list = vec![
            ScoredKey {
                key: "a".into(),
                score: 1.0,
            },
            ScoredKey {
                key: "b".into(),
                score: 0.5,
            },
        ];
        let result = reciprocal_rank_fusion(&[list], 60);
        assert_eq!(result[0].0, "a");
        assert_eq!(result[1].0, "b");
        assert!(result[0].1 > result[1].1);
    }

    #[test]
    fn test_rrf_two_lists_boost_overlap() {
        let list1 = vec![
            ScoredKey {
                key: "a".into(),
                score: 1.0,
            },
            ScoredKey {
                key: "b".into(),
                score: 0.5,
            },
        ];
        let list2 = vec![
            ScoredKey {
                key: "b".into(),
                score: 1.0,
            },
            ScoredKey {
                key: "c".into(),
                score: 0.5,
            },
        ];
        let result = reciprocal_rank_fusion(&[list1, list2], 60);
        // "b" appears in both lists at different ranks, should be boosted
        assert_eq!(result[0].0, "b");
    }

    #[test]
    fn test_rrf_empty_lists() {
        let result = reciprocal_rank_fusion(&[], 60);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rrf_k_parameter() {
        let list = vec![ScoredKey {
            key: "a".into(),
            score: 1.0,
        }];
        let r1 = reciprocal_rank_fusion(&[list.clone()], 60);
        let r2 = reciprocal_rank_fusion(&[list], 1);
        // Lower k gives higher scores
        assert!(r2[0].1 > r1[0].1);
    }
}
