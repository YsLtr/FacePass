//! Face feature matching module

use crate::error::{Error, Result};
use crate::models::FaceData;

/// Calculate cosine similarity between two feature vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::Recognition(format!(
            "Feature vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let mut dot_product = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot_product += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm = norm_a.sqrt() * norm_b.sqrt();
    if norm < 1e-10 {
        return Ok(0.0);
    }

    Ok(dot_product / norm)
}

/// Calculate L2 (Euclidean) distance between two feature vectors
pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::Recognition(format!(
            "Feature vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = (*x as f64) - (*y as f64);
            diff * diff
        })
        .sum();

    Ok(sum.sqrt())
}

/// Match result containing similarity score and matched face info
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Similarity score (cosine similarity, higher is better)
    pub similarity: f64,
    /// The matched face data
    pub face_data: FaceData,
    /// Index in the original list
    pub index: usize,
    /// Whether the similarity passed the requested threshold
    pub passed_threshold: bool,
}

/// Find the most similar face from a list of candidates.
///
/// Returns `None` only when `candidates` is empty. Otherwise the best candidate is
/// always returned and `passed_threshold` indicates whether it qualifies as a match.
pub fn find_best_match(
    query_feature: &[f32],
    candidates: &[FaceData],
    threshold: f64,
) -> Result<Option<MatchResult>> {
    if candidates.is_empty() {
        return Ok(None);
    }

    let mut best_match: Option<MatchResult> = None;

    for (index, face_data) in candidates.iter().enumerate() {
        let similarity = cosine_similarity(query_feature, &face_data.feature)?;
        let passed_threshold = similarity >= threshold;

        match &best_match {
            Some(current_best) if current_best.similarity >= similarity => {}
            _ => {
                best_match = Some(MatchResult {
                    similarity,
                    face_data: face_data.clone(),
                    index,
                    passed_threshold,
                });
            }
        };
    }

    Ok(best_match)
}

/// Check if a face matches any in the list (returns first match above threshold)
pub fn matches_any(query_feature: &[f32], candidates: &[FaceData], threshold: f64) -> Result<bool> {
    for face_data in candidates {
        let similarity = cosine_similarity(query_feature, &face_data.feature)?;
        if similarity >= threshold {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Get all matches above threshold, sorted by similarity (descending)
pub fn find_all_matches(
    query_feature: &[f32],
    candidates: &[FaceData],
    threshold: f64,
) -> Result<Vec<MatchResult>> {
    let mut matches = Vec::new();

    for (index, face_data) in candidates.iter().enumerate() {
        let similarity = cosine_similarity(query_feature, &face_data.feature)?;

        if similarity >= threshold {
            matches.push(MatchResult {
                similarity,
                face_data: face_data.clone(),
                index,
                passed_threshold: true,
            });
        }
    }

    // Sort by similarity (descending)
    matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let similarity = cosine_similarity(&a, &a).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![-1.0f32, 0.0, 0.0, 0.0];
        let similarity = cosine_similarity(&a, &b).unwrap();
        assert!((similarity + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![3.0f32, 4.0, 0.0];
        let distance = l2_distance(&a, &b).unwrap();
        assert!((distance - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_best_match() {
        let query: Vec<f32> = vec![1.0; 128];

        let candidates = vec![
            FaceData::new("face1", vec![0.5; 128]),
            FaceData::new("face2", vec![0.9; 128]), // More similar
            FaceData::new("face3", vec![0.3; 128]),
        ];

        let result = find_best_match(&query, &candidates, 0.5).unwrap();
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.face_data.label, "face2");
        assert_eq!(matched.index, 1);
        assert!(matched.passed_threshold);
    }

    #[test]
    fn test_find_best_match_no_match() {
        let query: Vec<f32> = vec![1.0; 128];
        let candidates = vec![FaceData::new("face1", vec![-1.0; 128])];

        let result = find_best_match(&query, &candidates, 0.9).unwrap();
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.face_data.label, "face1");
        assert!(!matched.passed_threshold);
    }

    #[test]
    fn test_find_best_match_empty_candidates() {
        let query: Vec<f32> = vec![1.0; 128];

        let result = find_best_match(&query, &[], 0.9).unwrap();
        assert!(result.is_none());
    }
}
