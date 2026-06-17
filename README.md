# recom

一个可直接运行的大数据推荐系统演示项目，包含后端、数据库、推荐算法和前端可视化仪表盘。

## 功能

- 后端基于 Flask，提供用户、商品、推荐结果和仪表盘数据接口。
- 数据层基于 SQLAlchemy，默认使用 SQLite，本地零配置可运行；也可以通过 `DATABASE_URL` 切换到 PostgreSQL 或 MySQL。
- 推荐算法采用混合策略：item-based CF、user-based CF、类目偏好、热度和新鲜度信号。
- 前端包含公共 Web 仪表盘和管理后台，支持用户、商品、评分的增删改查，并展示操作日志。

## 运行方式

进入目录后直接启动：

```bash
cd expstu/project
pip install -r requirements.txt
python3 1.py
```

然后在浏览器打开 `http://127.0.0.1:5000`。

如果要切到 PostgreSQL 或 MySQL，先设置 `DATABASE_URL`，例如：

```bash
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/recom"
```

或：

```bash
export DATABASE_URL="mysql+pymysql://user:password@localhost:3306/recom"
```

管理后台入口：`http://127.0.0.1:5000/login`。

默认账号：`admin / admin123`

## 接口

- `GET /api/users`：获取用户列表
- `GET /api/products`：获取商品列表
- `GET /api/dashboard?user_id=1`：获取仪表盘数据
- `GET /api/recommendations/<user_id>`：获取指定用户的推荐列表
- `GET /api/admin/state?user_id=1`：获取管理后台状态
- `GET /api/admin/logs`：获取最近操作日志
- `POST /api/admin/users`：新增用户
- `POST /api/admin/products`：新增商品
- `POST /api/admin/ratings`：新增评分
- `PUT /api/admin/users/<id>`：编辑用户
- `PUT /api/admin/products/<id>`：编辑商品
- `PUT /api/admin/ratings/<id>`：编辑评分
- `DELETE /api/admin/users/<id>`：删除用户
- `DELETE /api/admin/products/<id>`：删除商品
- `DELETE /api/admin/ratings/<id>`：删除评分

## 项目文件

- `expstu/project/app.py`：Flask 应用、SQLAlchemy 数据层和路由
- `expstu/project/recommender.py`：推荐算法实现
- `expstu/project/templates/index.html`：前端页面模板
- `expstu/project/templates/login.html`：登录页
- `expstu/project/templates/admin.html`：管理后台页
- `expstu/project/static/app.js`：前端交互逻辑
- `expstu/project/static/admin.js`：后台交互逻辑
- `expstu/project/static/style.css`：前端样式