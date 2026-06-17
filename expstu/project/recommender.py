from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from math import log1p, sqrt
from typing import Any, Dict, List, Mapping, Sequence


def _cosine_similarity(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    common_keys = set(left) & set(right)
    if not common_keys:
        return 0.0

    numerator = sum(left[key] * right[key] for key in common_keys)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    value_text = str(value).replace("T", " ")
    try:
        return datetime.fromisoformat(value_text)
    except ValueError:
        return None


def _build_user_vectors(ratings: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, float]]:
    user_vectors: Dict[int, Dict[int, float]] = defaultdict(dict)
    for rating in ratings:
        user_vectors[int(rating["user_id"])][int(rating["product_id"])] = float(rating["rating"])
    return user_vectors


def _build_item_vectors(ratings: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, float]]:
    item_vectors: Dict[int, Dict[int, float]] = defaultdict(dict)
    for rating in ratings:
        item_vectors[int(rating["product_id"])][int(rating["user_id"])] = float(rating["rating"])
    return item_vectors


def _build_category_preferences(
    ratings: Sequence[Mapping[str, Any]], products_by_id: Mapping[int, Mapping[str, Any]]
) -> Dict[int, Dict[str, float]]:
    preferences: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rating in ratings:
        product = products_by_id.get(int(rating["product_id"]))
        if not product:
            continue
        category = str(product["category"])
        preferences[int(rating["user_id"])][category].append(float(rating["rating"]))

    averaged: Dict[int, Dict[str, float]] = {}
    for user_id, category_values in preferences.items():
        averaged[user_id] = {
            category: round(sum(values) / len(values), 2) for category, values in category_values.items() if values
        }
    return averaged


def _build_item_similarities(item_vectors: Mapping[int, Mapping[int, float]]) -> Dict[int, Dict[int, float]]:
    similarities: Dict[int, Dict[int, float]] = defaultdict(dict)
    product_ids = list(item_vectors)
    for index, left_product_id in enumerate(product_ids):
        for right_product_id in product_ids[index + 1 :]:
            similarity = _cosine_similarity(item_vectors[left_product_id], item_vectors[right_product_id])
            if similarity <= 0:
                continue
            similarities[left_product_id][right_product_id] = similarity
            similarities[right_product_id][left_product_id] = similarity
    return similarities


def _build_user_similarities(user_vectors: Mapping[int, Mapping[int, float]]) -> Dict[int, Dict[int, float]]:
    similarities: Dict[int, Dict[int, float]] = defaultdict(dict)
    user_ids = list(user_vectors)
    for index, left_user_id in enumerate(user_ids):
        for right_user_id in user_ids[index + 1 :]:
            similarity = _cosine_similarity(user_vectors[left_user_id], user_vectors[right_user_id])
            if similarity <= 0:
                continue
            similarities[left_user_id][right_user_id] = similarity
            similarities[right_user_id][left_user_id] = similarity
    return similarities


def _build_popularity(
    ratings: Sequence[Mapping[str, Any]], products_by_id: Mapping[int, Mapping[str, Any]]
) -> Dict[int, Dict[str, float]]:
    product_ratings: Dict[int, List[float]] = defaultdict(list)
    latest_rating_at: Dict[int, datetime | None] = defaultdict(lambda: None)
    now = datetime.now()

    for rating in ratings:
        product_id = int(rating["product_id"])
        product_ratings[product_id].append(float(rating["rating"]))
        parsed_timestamp = _parse_timestamp(rating.get("created_at"))
        current_latest = latest_rating_at.get(product_id)
        if parsed_timestamp and (current_latest is None or parsed_timestamp > current_latest):
            latest_rating_at[product_id] = parsed_timestamp

    max_count = max((len(values) for values in product_ratings.values()), default=1)
    popularity: Dict[int, Dict[str, float]] = {}
    for product_id, product in products_by_id.items():
        product_values = product_ratings.get(product_id, [])
        count = len(product_values)
        average = sum(product_values) / count if count else 0.0
        count_score = log1p(count) / log1p(max_count) if max_count > 1 else 0.0
        latest_timestamp = latest_rating_at.get(product_id)
        freshness_score = 0.0
        if latest_timestamp:
            age_days = max(0, (now - latest_timestamp).days)
            freshness_score = 1 / (1 + age_days / 7)

        popularity[product_id] = {
            "average_rating": round(average, 2),
            "count": float(count),
            "count_score": round(count_score, 4),
            "freshness_score": round(freshness_score, 4),
            "score": (average / 5.0) * 0.65 + count_score * 0.2 + freshness_score * 0.15,
        }
    return popularity


def _score_item_based(
    target_vector: Mapping[int, float],
    product_id: int,
    item_similarities: Mapping[int, Mapping[int, float]],
) -> float:
    similarity_sum = 0.0
    weighted_score = 0.0
    for seen_product_id, seen_rating in target_vector.items():
        similarity = item_similarities.get(product_id, {}).get(seen_product_id, 0.0)
        if similarity <= 0:
            continue
        similarity_sum += abs(similarity)
        weighted_score += similarity * seen_rating
    if similarity_sum == 0:
        return 0.0
    return weighted_score / similarity_sum


def _score_user_based(
    target_user_id: int,
    product_id: int,
    user_vectors: Mapping[int, Mapping[int, float]],
    user_similarities: Mapping[int, Mapping[int, float]],
) -> float:
    similarity_sum = 0.0
    weighted_score = 0.0
    for other_user_id, other_vector in user_vectors.items():
        if other_user_id == target_user_id:
            continue
        if product_id not in other_vector:
            continue
        similarity = user_similarities.get(target_user_id, {}).get(other_user_id, 0.0)
        if similarity <= 0:
            continue
        similarity_sum += abs(similarity)
        weighted_score += similarity * other_vector[product_id]
    if similarity_sum == 0:
        return 0.0
    return weighted_score / similarity_sum


def generate_recommendations(
    target_user_id: int,
    users: Sequence[Mapping[str, Any]],
    products: Sequence[Mapping[str, Any]],
    ratings: Sequence[Mapping[str, Any]],
    limit: int = 6,
) -> List[Dict[str, Any]]:
    products_by_id = {int(product["id"]): product for product in products}
    user_vectors = _build_user_vectors(ratings)
    item_vectors = _build_item_vectors(ratings)
    user_similarities = _build_user_similarities(user_vectors)
    item_similarities = _build_item_similarities(item_vectors)
    category_preferences = _build_category_preferences(ratings, products_by_id)
    popularity = _build_popularity(ratings, products_by_id)
    target_vector = user_vectors.get(target_user_id, {})

    rated_product_ids = set(target_vector)
    scored_items: List[Dict[str, Any]] = []

    for product in products:
        product_id = int(product["id"])
        if product_id in rated_product_ids:
            continue

        item_score = _score_item_based(target_vector, product_id, item_similarities)
        user_score = _score_user_based(target_user_id, product_id, user_vectors, user_similarities)

        category = str(product["category"])
        category_score = category_preferences.get(target_user_id, {}).get(category, 0.0)
        popularity_score = popularity.get(product_id, {}).get("score", 0.0) * 5.0
        freshness_score = popularity.get(product_id, {}).get("freshness_score", 0.0) * 5.0

        final_score = (
            item_score * 0.34
            + user_score * 0.28
            + category_score * 0.18
            + popularity_score * 0.12
            + freshness_score * 0.08
        )
        final_score = max(0.0, min(final_score, 5.0))

        reasons: List[str] = []
        if item_score:
            reasons.append("与您高评分商品相似")
        if user_score:
            reasons.append("相似用户喜欢这类商品")
        if category_score:
            reasons.append(f"您对 {category} 类目偏好较高")
        if freshness_score:
            reasons.append("近期热度较高")
        if not reasons:
            reasons.append("基于全局热度兜底推荐")

        scored_items.append(
            {
                "id": product_id,
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "description": product["description"],
                "score": round(final_score, 2),
                "score_detail": {
                    "item_cf_score": round(item_score, 2),
                    "user_cf_score": round(user_score, 2),
                    "content_score": round(category_score, 2),
                    "popularity_score": round(popularity_score, 2),
                    "freshness_score": round(freshness_score, 2),
                },
                "reasons": reasons,
                "avg_rating": round(popularity.get(product_id, {}).get("average_rating", 0.0), 2),
                "rating_count": int(popularity.get(product_id, {}).get("count", 0)),
            }
        )

    scored_items.sort(key=lambda item: (item["score"], item["rating_count"]), reverse=True)
    return scored_items[:limit]


def summarize_catalog(products: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    category_counter = Counter(str(product["category"]) for product in products)
    return [
        {"name": category, "value": count}
        for category, count in sorted(category_counter.items(), key=lambda item: item[1], reverse=True)
    ]
