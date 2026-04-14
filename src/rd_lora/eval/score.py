from __future__ import annotations


def subject_score(
    dino_ref_max_mean: float,
    clip_i_ref_max_mean: float,
    clip_t_mean: float,
) -> float:
    """
    Composite score for subject_personalization task.
    Weights: DINO 0.60, CLIP-I 0.10, CLIP-T 0.30
    All inputs must be in [0, 1] (cosine normalized).
    """
    return round(0.60 * dino_ref_max_mean + 0.10 * clip_i_ref_max_mean + 0.30 * clip_t_mean, 6)


def style_score(
    ocr_exact_match: float,
    ocr_cer_mean: float,
    clip_t_mean: float,
) -> float:
    """
    Composite score for style_domain task.
    ocr_cer_score = max(0, 1 - min(1, ocr_cer_mean))
    ocr_text_score = 0.70 * ocr_cer_score + 0.30 * ocr_exact_match
    score = 0.70 * ocr_text_score + 0.30 * clip_t_mean
    All inputs must be in [0, 1].
    """
    ocr_cer_score = max(0.0, 1.0 - min(1.0, ocr_cer_mean))
    ocr_text_score = 0.70 * ocr_cer_score + 0.30 * ocr_exact_match
    return round(0.70 * ocr_text_score + 0.30 * clip_t_mean, 6)


def backend_score(task_scores: dict[str, float]) -> float:
    """
    Backend-level aggregate: mean of subject and style task scores.
    Requires both "subject_personalization" and "style_domain" keys.
    """
    required = ["subject_personalization", "style_domain"]
    for key in required:
        if key not in task_scores:
            raise ValueError(f"Missing required task score: {key}")
    return round(0.5 * task_scores["subject_personalization"] + 0.5 * task_scores["style_domain"], 6)
