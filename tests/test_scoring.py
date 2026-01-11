from scenarioops.graph.tools.scoring import hash_scoring_result, score_with_rubric


def test_scoring_is_deterministic() -> None:
    sample_scores = {
        "relevance": 0.9,
        "credibility": 0.8,
        "recency": 0.4,
        "specificity": 0.7,
    }

    first = score_with_rubric(sample_scores)
    second = score_with_rubric(sample_scores)

    assert first.action == second.action
    assert first.normalized_total == second.normalized_total
    assert hash_scoring_result(first) == hash_scoring_result(second)
