//! Face feature matching module

use crate::error::{Error, Result};
use crate::models::{FaceData, FaceEmbedding};

fn ensure_query_embedding_valid(query: &FaceEmbedding) -> Result<()> {
    if !query.is_valid() {
        return Err(Error::Recognition(
            "Query embedding is missing model metadata or has an invalid length".to_string(),
        ));
    }
    Ok(())
}

fn ensure_comparable(query: &FaceEmbedding, candidate: &FaceData) -> Result<()> {
    ensure_query_embedding_valid(query)?;
    if !candidate.is_valid() {
        return Err(Error::Recognition(format!(
            "Stored face '{}' has invalid embedding metadata",
            candidate.label
        )));
    }

    if query.model_id != candidate.model_id {
        return Err(Error::Recognition(format!(
            "Cannot compare embeddings from different models: '{}' vs '{}'",
            query.model_id, candidate.model_id
        )));
    }

    if query.embedding_dim != candidate.embedding_dim {
        return Err(Error::Recognition(format!(
            "Embedding dimension mismatch for model '{}': {} vs {}",
            query.model_id, query.embedding_dim, candidate.embedding_dim
        )));
    }

    if query.feature.len() != query.embedding_dim {
        return Err(Error::Recognition(format!(
            "Query embedding length mismatch: expected {}, got {}",
            query.embedding_dim,
            query.feature.len()
        )));
    }

    Ok(())
}

/// Calculate cosine similarity between a query embedding and a stored face.
pub fn cosine_similarity(query: &FaceEmbedding, candidate: &FaceData) -> Result<f64> {
    ensure_comparable(query, candidate)?;

    let mut dot_product = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in query.feature.iter().zip(candidate.feature.iter()) {
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

/// Calculate L2 (Euclidean) distance between a query embedding and a stored face.
pub fn l2_distance(query: &FaceEmbedding, candidate: &FaceData) -> Result<f64> {
    ensure_comparable(query, candidate)?;

    let sum: f64 = query
        .feature
        .iter()
        .zip(candidate.feature.iter())
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
pub fn find_best_match(
    query: &FaceEmbedding,
    candidates: &[FaceData],
    threshold: f64,
) -> Result<Option<MatchResult>> {
    if candidates.is_empty() {
        return Ok(None);
    }

    let mut best_match: Option<MatchResult> = None;

    for (index, face_data) in candidates.iter().enumerate() {
        let similarity = cosine_similarity(query, face_data)?;
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
pub fn matches_any(query: &FaceEmbedding, candidates: &[FaceData], threshold: f64) -> Result<bool> {
    for face_data in candidates {
        let similarity = cosine_similarity(query, face_data)?;
        if similarity >= threshold {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Get all matches above threshold, sorted by similarity (descending)
pub fn find_all_matches(
    query: &FaceEmbedding,
    candidates: &[FaceData],
    threshold: f64,
) -> Result<Vec<MatchResult>> {
    let mut matches = Vec::new();

    for (index, face_data) in candidates.iter().enumerate() {
        let similarity = cosine_similarity(query, face_data)?;

        if similarity >= threshold {
            matches.push(MatchResult {
                similarity,
                face_data: face_data.clone(),
                index,
                passed_threshold: true,
            });
        }
    }

    matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(feature: Vec<f32>) -> FaceEmbedding {
        FaceEmbedding::new("ghostfacenet-512", feature)
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let feature: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let candidate = FaceData::new("face1", "ghostfacenet-512", feature.clone());
        let similarity = cosine_similarity(&query(feature), &candidate).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_rejects_cross_model_compare() {
        let candidate = FaceData::new("face1", "sface-128", vec![1.0; 128]);
        let error = cosine_similarity(&query(vec![1.0; 512]), &candidate)
            .unwrap_err()
            .to_string();
        assert!(error.contains("different models"));
    }

    #[test]
    fn test_l2_distance() {
        let candidate = FaceData::new("face1", "ghostfacenet-512", vec![3.0; 512]);
        let distance = l2_distance(&query(vec![0.0; 512]), &candidate).unwrap();
        assert!(distance > 0.0);
    }

    #[test]
    fn test_find_best_match() {
        let candidates = vec![
            FaceData::new("face1", "ghostfacenet-512", vec![0.5; 512]),
            FaceData::new("face2", "ghostfacenet-512", vec![0.9; 512]),
            FaceData::new("face3", "ghostfacenet-512", vec![0.3; 512]),
        ];

        let result = find_best_match(&query(vec![1.0; 512]), &candidates, 0.5).unwrap();
        assert!(result.is_some());
        let matched = result.unwrap();
        assert_eq!(matched.face_data.label, "face2");
        assert_eq!(matched.index, 1);
        assert!(matched.passed_threshold);
    }

    #[test]
    fn test_find_best_match_empty_candidates() {
        let result = find_best_match(&query(vec![1.0; 512]), &[], 0.9).unwrap();
        assert!(result.is_none());
    }
}
