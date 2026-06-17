from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine, delete, func, insert, select, update
from sqlalchemy.engine import Engine
from werkzeug.security import check_password_hash, generate_password_hash

from recommender import generate_recommendations, summarize_catalog

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "recom.db"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


def _seed_timestamp(days_ago: int) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def _database_url() -> str:
    return os.getenv("DATABASE_URL") or f"sqlite:///{DB_PATH}"


def _build_engine() -> Engine:
    database_url = _database_url()
    connect_args: Dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, future=True, pool_pre_ping=True, connect_args=connect_args)


engine = _build_engine()
metadata = MetaData()

users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("age", Integer, nullable=False),
    Column("city", String(255), nullable=False),
    Column("loyalty_level", String(50), nullable=False),
)

products_table = Table(
    "products",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("category", String(255), nullable=False),
    Column("price", Float, nullable=False),
    Column("description", String(500), nullable=False),
)

ratings_table = Table(
    "ratings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, nullable=False),
    Column("product_id", Integer, nullable=False),
    Column("rating", Float, nullable=False),
    Column("created_at", String(32), nullable=False),
)

admin_users_table = Table(
    "admin_users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(255), nullable=False, unique=True),
    Column("password_hash", String(255), nullable=False),
)

operation_logs_table = Table(
    "operation_logs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("admin_username", String(255), nullable=False),
    Column("action", String(50), nullable=False),
    Column("entity_type", String(50), nullable=False),
    Column("entity_id", Integer, nullable=True),
    Column("detail", String(2000), nullable=False),
    Column("created_at", String(32), nullable=False),
)

USERS = [
    (1, "王晨", 24, "上海", "gold"),
    (2, "李静", 31, "杭州", "silver"),
    (3, "张伟", 28, "深圳", "gold"),
    (4, "刘洋", 35, "北京", "bronze"),
    (5, "陈婷", 22, "成都", "silver"),
    (6, "赵磊", 29, "武汉", "gold"),
]

PRODUCTS = [
    (1, "Spark 实战指南", "大数据", 89.0, "面向工程实践的离线与流式计算入门。"),
    (2, "Kafka 流处理基础", "大数据", 99.0, "围绕消息流与实时推荐管道展开。"),
    (3, "Python 数据分析", "工具", 79.0, "适合做特征工程和业务分析。"),
    (4, "Vue 可视化面板", "前端", 109.0, "适合快速搭建数据大屏。"),
    (5, "机器学习导论", "算法", 119.0, "涵盖监督学习与特征选择。"),
    (6, "推荐系统工程化", "算法", 129.0, "从召回、排序到在线服务。"),
    (7, "Flink 实时计算", "大数据", 139.0, "低延迟流式计算与状态管理。"),
    (8, "FastAPI 服务开发", "后端", 96.0, "快速搭建推荐 API 与管理接口。"),
    (9, "SQLite 数据建模", "数据库", 68.0, "轻量级本地数据存储与分析。"),
    (10, "Docker 部署实践", "运维", 88.0, "容器化与环境交付。"),
    (11, "推荐系统案例合集", "算法", 149.0, "适合复盘召回、排序与特征。"),
    (12, "大屏交互设计", "前端", 99.0, "强调可视化呈现与交互体验。"),
]

RATINGS = [
    (1, 1, 4.8, _seed_timestamp(12)),
    (1, 2, 4.6, _seed_timestamp(11)),
    (1, 5, 4.2, _seed_timestamp(10)),
    (1, 6, 4.9, _seed_timestamp(9)),
    (2, 1, 4.4, _seed_timestamp(10)),
    (2, 3, 4.1, _seed_timestamp(9)),
    (2, 4, 4.7, _seed_timestamp(8)),
    (2, 8, 4.6, _seed_timestamp(7)),
    (3, 2, 4.9, _seed_timestamp(8)),
    (3, 5, 4.7, _seed_timestamp(7)),
    (3, 6, 4.8, _seed_timestamp(6)),
    (3, 11, 4.6, _seed_timestamp(5)),
    (4, 3, 4.2, _seed_timestamp(7)),
    (4, 7, 4.8, _seed_timestamp(6)),
    (4, 9, 4.0, _seed_timestamp(5)),
    (4, 10, 3.8, _seed_timestamp(4)),
    (5, 4, 4.5, _seed_timestamp(6)),
    (5, 8, 4.7, _seed_timestamp(5)),
    (5, 12, 4.6, _seed_timestamp(4)),
    (5, 3, 4.1, _seed_timestamp(3)),
    (6, 1, 4.7, _seed_timestamp(3)),
    (6, 6, 4.9, _seed_timestamp(2)),
    (6, 7, 4.6, _seed_timestamp(2)),
    (6, 11, 4.8, _seed_timestamp(1)),
]
def initialize_database() -> None:
    metadata.create_all(engine)
    with engine.begin() as connection:
        user_count = connection.execute(select(func.count()).select_from(users_table)).scalar_one()
        if not user_count:
            connection.execute(
                insert(users_table),
                [
                    {
                        "id": user_id,
                        "name": name,
                        "age": age,
                        "city": city,
                        "loyalty_level": loyalty_level,
                    }
                    for user_id, name, age, city, loyalty_level in USERS
                ],
            )

        product_count = connection.execute(select(func.count()).select_from(products_table)).scalar_one()
        if not product_count:
            connection.execute(
                insert(products_table),
                [
                    {
                        "id": product_id,
                        "name": name,
                        "category": category,
                        "price": price,
                        "description": description,
                    }
                    for product_id, name, category, price, description in PRODUCTS
                ],
            )

        rating_count = connection.execute(select(func.count()).select_from(ratings_table)).scalar_one()
        if not rating_count:
            connection.execute(
                insert(ratings_table),
                [
                    {
                        "user_id": user_id,
                        "product_id": product_id,
                        "rating": rating,
                        "created_at": created_at,
                    }
                    for user_id, product_id, rating, created_at in RATINGS
                ],
            )

        admin_count = connection.execute(select(func.count()).select_from(admin_users_table)).scalar_one()
        if not admin_count:
            connection.execute(
                insert(admin_users_table).values(
                    username=DEFAULT_ADMIN_USERNAME,
                    password_hash=generate_password_hash(DEFAULT_ADMIN_PASSWORD),
                )
            )


def fetch_rows(query: Any, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    with engine.connect() as connection:
        result = connection.execute(query, params or {})
        return [dict(row._mapping) for row in result.fetchall()]


def fetch_single_row(query: Any, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(query, params or {}).fetchone()
    return dict(row._mapping) if row else None


def _next_id(table: Table) -> int:
    with engine.connect() as connection:
        next_id = connection.execute(select(func.coalesce(func.max(table.c.id), 0) + 1)).scalar_one()
    return int(next_id)


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是整数") from exc


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数字") from exc


def _request_data() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload
    return request.form.to_dict(flat=True)


def _admin_username() -> str:
    return str(session.get("admin_username") or DEFAULT_ADMIN_USERNAME)


def _log_operation(action: str, entity_type: str, entity_id: int | None, detail: Dict[str, Any]) -> None:
    with engine.begin() as connection:
        connection.execute(
            insert(operation_logs_table).values(
                admin_username=_admin_username(),
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                detail=json.dumps(detail, ensure_ascii=False),
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )


def _product_stats(products: List[Dict[str, Any]], ratings: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {}
    for product in products:
        product_id = int(product["id"])
        product_ratings = [rating for rating in ratings if int(rating["product_id"]) == product_id]
        count = len(product_ratings)
        average = round(sum(float(rating["rating"]) for rating in product_ratings) / max(1, count), 2)
        latest_created_at = max((rating["created_at"] for rating in product_ratings), default="")
        stats[product_id] = {
            "count": count,
            "avg": average,
            "latest_created_at": latest_created_at,
            "total_score": round(sum(float(rating["rating"]) for rating in product_ratings), 2),
        }
    return stats


def fetch_dashboard_data(user_id: int) -> Dict[str, Any]:
    users = fetch_rows(select(users_table).order_by(users_table.c.id))
    products = fetch_rows(select(products_table).order_by(products_table.c.id))
    ratings = fetch_rows(select(ratings_table).order_by(ratings_table.c.created_at.desc(), ratings_table.c.id.desc()))

    if not users:
        return {
            "selected_user": {},
            "users": [],
            "products": [],
            "ratings": [],
            "summary": {"user_count": 0, "product_count": 0, "rating_count": 0, "avg_rating": 0.0},
            "category_breakdown": [],
            "rating_distribution": [],
            "top_products": [],
            "recommendations": [],
        }

    selected_user = next((user for user in users if int(user["id"]) == user_id), users[0])
    recommendations = generate_recommendations(user_id, users, products, ratings, limit=6)
    stats = _product_stats(products, ratings)

    summary = {
        "user_count": len(users),
        "product_count": len(products),
        "rating_count": len(ratings),
        "avg_rating": round(sum(float(rating["rating"]) for rating in ratings) / max(1, len(ratings)), 2),
    }

    rating_distribution = [
        {"name": "1-2 星", "value": sum(1 for row in ratings if float(row["rating"]) < 3.0)},
        {"name": "3-4 星", "value": sum(1 for row in ratings if 3.0 <= float(row["rating"]) < 4.5)},
        {"name": "4.5-5 星", "value": sum(1 for row in ratings if float(row["rating"]) >= 4.5)},
    ]

    top_products = sorted(
        products,
        key=lambda product: (
            stats[int(product["id"])]["avg"],
            stats[int(product["id"])]["count"],
            stats[int(product["id"])]["total_score"],
        ),
        reverse=True,
    )[:5]

    return {
        "selected_user": dict(selected_user),
        "users": users,
        "products": products,
        "ratings": ratings,
        "summary": summary,
        "category_breakdown": summarize_catalog(products),
        "rating_distribution": rating_distribution,
        "top_products": [
            {
                "id": int(product["id"]),
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "count": stats[int(product["id"])] ["count"],
                "avg_rating": stats[int(product["id"])] ["avg"],
            }
            for product in top_products
        ],
        "recommendations": recommendations,
    }


def _require_admin(view_function):
    @wraps(view_function)
    def wrapper(*args, **kwargs):
        if not session.get("admin_username"):
            return redirect(url_for("login", next=request.path))
        return view_function(*args, **kwargs)

    return wrapper


def _validate_user_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = str(payload.get("name", "")).strip()
    city = str(payload.get("city", "")).strip()
    loyalty_level = str(payload.get("loyalty_level", "")).strip()
    age = _coerce_int(payload.get("age"), "年龄")
    if not name or not city or not loyalty_level:
        raise ValueError("用户信息不完整")
    return {"name": name, "age": age, "city": city, "loyalty_level": loyalty_level}


def _validate_product_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = str(payload.get("name", "")).strip()
    category = str(payload.get("category", "")).strip()
    description = str(payload.get("description", "")).strip()
    price = _coerce_float(payload.get("price"), "价格")
    if not name or not category or not description:
        raise ValueError("商品信息不完整")
    return {"name": name, "category": category, "price": price, "description": description}


def _validate_rating_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_id = _coerce_int(payload.get("user_id"), "用户 ID")
    product_id = _coerce_int(payload.get("product_id"), "商品 ID")
    rating = _coerce_float(payload.get("rating"), "评分")
    if not (1.0 <= rating <= 5.0):
        raise ValueError("评分必须在 1 到 5 之间")
    return {"user_id": user_id, "product_id": product_id, "rating": rating}


def _delete_user(user_id: int) -> None:
    with engine.begin() as connection:
        connection.execute(delete(ratings_table).where(ratings_table.c.user_id == user_id))
        connection.execute(delete(users_table).where(users_table.c.id == user_id))


def _delete_product(product_id: int) -> None:
    with engine.begin() as connection:
        connection.execute(delete(ratings_table).where(ratings_table.c.product_id == product_id))
        connection.execute(delete(products_table).where(products_table.c.id == product_id))


def _admin_state(user_id: int) -> Dict[str, Any]:
    dashboard = fetch_dashboard_data(user_id)
    logs = fetch_rows(select(operation_logs_table).order_by(operation_logs_table.c.id.desc()).limit(20))
    return {
        "admin_username": session.get("admin_username"),
        "dashboard": dashboard,
        "users": dashboard["users"],
        "products": dashboard["products"],
        "ratings": dashboard["ratings"],
        "logs": logs,
    }


def create_app() -> Flask:
    initialize_database()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "recom-demo-secret-key"

    @app.get("/")
    def index() -> str:
        users = fetch_rows(select(users_table).order_by(users_table.c.id))
        first_user_id = int(users[0]["id"]) if users else 1
        data = fetch_dashboard_data(first_user_id)
        return render_template(
            "index.html",
            initial_data=data,
            users=users,
            is_admin_authenticated=bool(session.get("admin_username")),
            admin_username=session.get("admin_username"),
        )

    @app.route("/login", methods=["GET", "POST"])
    def login() -> str:
        if session.get("admin_username"):
            return redirect(url_for("admin"))

        error_message = ""
        if request.method == "POST":
            username = str(request.form.get("username", "")).strip()
            password = str(request.form.get("password", ""))
            admin_row = fetch_single_row(select(admin_users_table).where(admin_users_table.c.username == username))
            if admin_row and check_password_hash(admin_row["password_hash"], password):
                session["admin_username"] = admin_row["username"]
                session["admin_user_id"] = int(admin_row["id"])
                _log_operation("login", "admin", int(admin_row["id"]), {"username": admin_row["username"]})
                next_path = request.args.get("next") or url_for("admin")
                return redirect(next_path)
            error_message = "用户名或密码错误"

        return render_template("login.html", error_message=error_message)

    @app.post("/logout")
    def logout() -> Any:
        session.clear()
        return redirect(url_for("index"))

    @app.get("/admin")
    @_require_admin
    def admin() -> str:
        users = fetch_rows(select(users_table).order_by(users_table.c.id))
        first_user_id = int(users[0]["id"]) if users else 1
        state = _admin_state(first_user_id)
        return render_template(
            "admin.html",
            initial_data=state,
            admin_username=session.get("admin_username"),
            is_admin_authenticated=True,
        )

    @app.get("/api/users")
    def api_users() -> Any:
        return jsonify(fetch_rows(select(users_table).order_by(users_table.c.id)))

    @app.get("/api/products")
    def api_products() -> Any:
        return jsonify(fetch_rows(select(products_table).order_by(products_table.c.id)))

    @app.get("/api/dashboard")
    def api_dashboard() -> Any:
        user_id = int(request.args.get("user_id", 1))
        return jsonify(fetch_dashboard_data(user_id))

    @app.get("/api/recommendations/<int:user_id>")
    def api_recommendations(user_id: int) -> Any:
        payload = fetch_dashboard_data(user_id)
        return jsonify(
            {
                "selected_user": payload["selected_user"],
                "recommendations": payload["recommendations"],
            }
        )

    @app.get("/api/admin/state")
    @_require_admin
    def api_admin_state() -> Any:
        user_id = int(request.args.get("user_id", 1))
        return jsonify(_admin_state(user_id))

    @app.get("/api/admin/logs")
    @_require_admin
    def api_admin_logs() -> Any:
        return jsonify(fetch_rows(select(operation_logs_table).order_by(operation_logs_table.c.id.desc()).limit(50)))

    @app.post("/api/admin/users")
    @_require_admin
    def api_create_user() -> Any:
        try:
            payload = _validate_user_payload(_request_data())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        user_id = _next_id(users_table)
        with engine.begin() as connection:
            connection.execute(insert(users_table).values(id=user_id, **payload))

        _log_operation("create", "user", user_id, payload)
        return jsonify({"ok": True, "user": {"id": user_id, **payload}})

    @app.put("/api/admin/users/<int:user_id>")
    @_require_admin
    def api_update_user(user_id: int) -> Any:
        existing_user = fetch_single_row(select(users_table).where(users_table.c.id == user_id))
        if not existing_user:
            return jsonify({"ok": False, "error": "用户不存在"}), 404
        try:
            payload = _validate_user_payload(_request_data())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        with engine.begin() as connection:
            connection.execute(update(users_table).where(users_table.c.id == user_id).values(**payload))

        _log_operation("update", "user", user_id, {"before": existing_user, "after": payload})
        return jsonify({"ok": True, "user": {"id": user_id, **payload}})

    @app.delete("/api/admin/users/<int:user_id>")
    @_require_admin
    def api_delete_user(user_id: int) -> Any:
        existing_user = fetch_single_row(select(users_table).where(users_table.c.id == user_id))
        _delete_user(user_id)
        if existing_user:
            _log_operation("delete", "user", user_id, existing_user)
        return jsonify({"ok": True})

    @app.post("/api/admin/products")
    @_require_admin
    def api_create_product() -> Any:
        try:
            payload = _validate_product_payload(_request_data())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        product_id = _next_id(products_table)
        with engine.begin() as connection:
            connection.execute(insert(products_table).values(id=product_id, **payload))

        _log_operation("create", "product", product_id, payload)
        return jsonify({"ok": True, "product": {"id": product_id, **payload}})

    @app.put("/api/admin/products/<int:product_id>")
    @_require_admin
    def api_update_product(product_id: int) -> Any:
        existing_product = fetch_single_row(select(products_table).where(products_table.c.id == product_id))
        if not existing_product:
            return jsonify({"ok": False, "error": "商品不存在"}), 404
        try:
            payload = _validate_product_payload(_request_data())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        with engine.begin() as connection:
            connection.execute(update(products_table).where(products_table.c.id == product_id).values(**payload))

        _log_operation("update", "product", product_id, {"before": existing_product, "after": payload})
        return jsonify({"ok": True, "product": {"id": product_id, **payload}})

    @app.delete("/api/admin/products/<int:product_id>")
    @_require_admin
    def api_delete_product(product_id: int) -> Any:
        existing_product = fetch_single_row(select(products_table).where(products_table.c.id == product_id))
        _delete_product(product_id)
        if existing_product:
            _log_operation("delete", "product", product_id, existing_product)
        return jsonify({"ok": True})

    @app.post("/api/admin/ratings")
    @_require_admin
    def api_create_rating() -> Any:
        try:
            payload = _validate_rating_payload(_request_data())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        if not fetch_single_row(select(users_table.c.id).where(users_table.c.id == payload["user_id"])):
            return jsonify({"ok": False, "error": "用户不存在"}), 404
        if not fetch_single_row(select(products_table.c.id).where(products_table.c.id == payload["product_id"])):
            return jsonify({"ok": False, "error": "商品不存在"}), 404

        with engine.begin() as connection:
            result = connection.execute(
                insert(ratings_table).values(
                    user_id=payload["user_id"],
                    product_id=payload["product_id"],
                    rating=payload["rating"],
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
            rating_id = int(result.inserted_primary_key[0])

        _log_operation("create", "rating", rating_id, payload)
        return jsonify({"ok": True, "rating": {"id": rating_id, **payload}})

    @app.put("/api/admin/ratings/<int:rating_id>")
    @_require_admin
    def api_update_rating(rating_id: int) -> Any:
        existing_rating = fetch_single_row(select(ratings_table).where(ratings_table.c.id == rating_id))
        if not existing_rating:
            return jsonify({"ok": False, "error": "评分不存在"}), 404
        try:
            payload = _validate_rating_payload(_request_data())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        if not fetch_single_row(select(users_table.c.id).where(users_table.c.id == payload["user_id"])):
            return jsonify({"ok": False, "error": "用户不存在"}), 404
        if not fetch_single_row(select(products_table.c.id).where(products_table.c.id == payload["product_id"])):
            return jsonify({"ok": False, "error": "商品不存在"}), 404

        with engine.begin() as connection:
            connection.execute(
                update(ratings_table)
                .where(ratings_table.c.id == rating_id)
                .values(
                    user_id=payload["user_id"],
                    product_id=payload["product_id"],
                    rating=payload["rating"],
                )
            )

        _log_operation("update", "rating", rating_id, {"before": existing_rating, "after": payload})
        return jsonify({"ok": True, "rating": {"id": rating_id, **payload}})

    @app.delete("/api/admin/ratings/<int:rating_id>")
    @_require_admin
    def api_delete_rating(rating_id: int) -> Any:
        existing_rating = fetch_single_row(select(ratings_table).where(ratings_table.c.id == rating_id))
        with engine.begin() as connection:
            connection.execute(delete(ratings_table).where(ratings_table.c.id == rating_id))
        if existing_rating:
            _log_operation("delete", "rating", rating_id, existing_rating)
        return jsonify({"ok": True})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
