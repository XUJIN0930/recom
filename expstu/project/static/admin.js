const initialAdminData = window.__ADMIN_DATA__;

const adminUserSelect = document.getElementById("adminUserSelect");
const adminMetricCards = document.getElementById("adminMetricCards");
const adminRecommendationList = document.getElementById("adminRecommendationList");
const adminCategoryChart = document.getElementById("adminCategoryChart");
const adminRatingChart = document.getElementById("adminRatingChart");
const usersTable = document.getElementById("usersTable");
const productsTable = document.getElementById("productsTable");
const ratingsTable = document.getElementById("ratingsTable");
const userForm = document.getElementById("userForm");
const productForm = document.getElementById("productForm");
const ratingForm = document.getElementById("ratingForm");
const ratingUserSelect = document.getElementById("ratingUserSelect");
const ratingProductSelect = document.getElementById("ratingProductSelect");
const logsTable = document.getElementById("logsTable");
const tabButtons = document.querySelectorAll(".tab");
const tabPanels = document.querySelectorAll(".table-panel");

let currentState = initialAdminData;

function formatCurrency(value) {
  return `¥${Number(value).toFixed(0)}`;
}

function renderMetricCards(summary) {
  adminMetricCards.innerHTML = "";
  const cards = [
    { label: "用户数", value: summary.user_count, note: "后台可维护用户池" },
    { label: "商品数", value: summary.product_count, note: "新增商品会影响召回" },
    { label: "评分数", value: summary.rating_count, note: "用于训练与验证推荐" },
    { label: "平均评分", value: summary.avg_rating, note: "衡量整体偏好强度" },
  ];

  cards.forEach((card) => {
    const element = document.createElement("div");
    element.className = "metric";
    element.innerHTML = `
      <div class="metric-label">${card.label}</div>
      <div class="metric-value">${card.value}</div>
      <div class="metric-note">${card.note}</div>
    `;
    adminMetricCards.appendChild(element);
  });
}

function renderBarChart(container, items, palette) {
  const maxValue = Math.max(...items.map((item) => item.value), 1);
  container.innerHTML = "";

  items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "chart-row";
    const width = Math.max(8, (item.value / maxValue) * 100);
    const color = palette[index % palette.length];

    row.innerHTML = `
      <div class="chart-row-head">
        <span>${item.name}</span>
        <strong>${item.value}</strong>
      </div>
      <div class="chart-track">
        <div class="chart-fill" style="width: ${width}%; background: ${color};"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function renderPreviewRecommendations(recommendations) {
  adminRecommendationList.innerHTML = "";
  recommendations.forEach((item) => {
    const element = document.createElement("article");
    element.className = "recommendation-item";
    element.innerHTML = `
      <header>
        <div>
          <span class="badge">${item.category}</span>
          <h3>${item.name}</h3>
          <p>${item.description}</p>
        </div>
        <div class="score">${item.score.toFixed(2)}</div>
      </header>
      <div class="meta">平均分 ${item.avg_rating.toFixed(2)} · 评分数 ${item.rating_count} · ${formatCurrency(item.price)}</div>
      <div class="reason-list">${item.reasons.map((reason) => `<span class="reason">${reason}</span>`).join("")}</div>
      <div class="score-detail">
        <div>Item CF：${item.score_detail.item_cf_score.toFixed(2)}</div>
        <div>User CF：${item.score_detail.user_cf_score.toFixed(2)}</div>
        <div>类目/热度：${item.score_detail.content_score.toFixed(2)} / ${item.score_detail.popularity_score.toFixed(2)}</div>
      </div>
    `;
    adminRecommendationList.appendChild(element);
  });
}

function renderUsersTable(users) {
  usersTable.innerHTML = "";
  ratingUserSelect.innerHTML = "";
  users.forEach((user) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${user.id}</td>
      <td>${user.name}</td>
      <td>${user.city}</td>
      <td>${user.loyalty_level}</td>
      <td><button class="button" data-edit-user="${user.id}">编辑</button></td>
      <td><button class="button danger-button" data-delete-user="${user.id}">删除</button></td>
    `;
    usersTable.appendChild(row);

    const option = document.createElement("option");
    option.value = user.id;
    option.textContent = `${user.name} · ${user.city}`;
    ratingUserSelect.appendChild(option);
  });
}

function renderProductsTable(products) {
  productsTable.innerHTML = "";
  ratingProductSelect.innerHTML = "";
  products.forEach((product) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${product.id}</td>
      <td>${product.name}</td>
      <td>${product.category}</td>
      <td>${formatCurrency(product.price)}</td>
      <td><button class="button" data-edit-product="${product.id}">编辑</button></td>
      <td><button class="button danger-button" data-delete-product="${product.id}">删除</button></td>
    `;
    productsTable.appendChild(row);

    const option = document.createElement("option");
    option.value = product.id;
    option.textContent = `${product.name} · ${product.category}`;
    ratingProductSelect.appendChild(option);
  });
}

function renderRatingsTable(ratings) {
  ratingsTable.innerHTML = "";
  ratings.forEach((rating) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${rating.id}</td>
      <td>${rating.user_id}</td>
      <td>${rating.product_id}</td>
      <td>${Number(rating.rating).toFixed(1)}</td>
      <td>${rating.created_at}</td>
      <td><button class="button" data-edit-rating="${rating.id}">编辑</button></td>
      <td><button class="button danger-button" data-delete-rating="${rating.id}">删除</button></td>
    `;
    ratingsTable.appendChild(row);
  });
}

function renderLogsTable(logs) {
  logsTable.innerHTML = "";
  logs.forEach((log) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${log.id}</td>
      <td>${log.admin_username}</td>
      <td>${log.action}</td>
      <td>${log.entity_type}</td>
      <td>${log.entity_id ?? "-"}</td>
      <td>${log.detail}</td>
      <td>${log.created_at}</td>
    `;
    logsTable.appendChild(row);
  });
}

function renderAdminUserSelect(users, selectedUserId) {
  adminUserSelect.innerHTML = "";
  users.forEach((user) => {
    const option = document.createElement("option");
    option.value = user.id;
    option.textContent = `${user.name} · ${user.city}`;
    if (Number(user.id) === Number(selectedUserId)) {
      option.selected = true;
    }
    adminUserSelect.appendChild(option);
  });
}

function renderAdminCharts(dashboard) {
  renderBarChart(adminCategoryChart, dashboard.category_breakdown, ["#7ce7c1", "#66a6ff", "#ffb86b", "#c08bff"]);
  renderBarChart(adminRatingChart, dashboard.rating_distribution, ["#66a6ff", "#7ce7c1", "#ffb86b"]);
}

function renderState(state) {
  currentState = state;
  const dashboard = state.dashboard;
  renderMetricCards(dashboard.summary);
  renderAdminUserSelect(state.users, dashboard.selected_user.id);
  renderPreviewRecommendations(dashboard.recommendations);
  renderAdminCharts(dashboard);
  renderUsersTable(state.users);
  renderProductsTable(state.products);
  renderRatingsTable(state.ratings);
  renderLogsTable(state.logs || []);
}

async function refreshState(userId = adminUserSelect.value || currentState.dashboard.selected_user.id) {
  const response = await fetch(`/api/admin/state?user_id=${userId}`);
  const state = await response.json();
  renderState(state);
}

async function submitJson(url, payload, method) {
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.error || "请求失败");
  }

  return response.json();
}

function promptEditUser(user) {
  const name = window.prompt("姓名", user.name);
  if (name === null) return null;
  const age = window.prompt("年龄", String(user.age));
  if (age === null) return null;
  const city = window.prompt("城市", user.city);
  if (city === null) return null;
  const loyaltyLevel = window.prompt("会员等级", user.loyalty_level);
  if (loyaltyLevel === null) return null;
  return { name, age, city, loyalty_level: loyaltyLevel };
}

function promptEditProduct(product) {
  const name = window.prompt("商品名称", product.name);
  if (name === null) return null;
  const category = window.prompt("类目", product.category);
  if (category === null) return null;
  const price = window.prompt("价格", String(product.price));
  if (price === null) return null;
  const description = window.prompt("描述", product.description);
  if (description === null) return null;
  return { name, category, price, description };
}

function promptEditRating(rating) {
  const userId = window.prompt("用户 ID", String(rating.user_id));
  if (userId === null) return null;
  const productId = window.prompt("商品 ID", String(rating.product_id));
  if (productId === null) return null;
  const score = window.prompt("评分", String(rating.rating));
  if (score === null) return null;
  return { user_id: userId, product_id: productId, rating: score };
}

userForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(userForm);
  const payload = Object.fromEntries(formData.entries());
  await submitJson("/api/admin/users", payload, "POST");
  userForm.reset();
  await refreshState();
});

productForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(productForm);
  const payload = Object.fromEntries(formData.entries());
  await submitJson("/api/admin/products", payload, "POST");
  productForm.reset();
  await refreshState();
});

ratingForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(ratingForm);
  const payload = Object.fromEntries(formData.entries());
  await submitJson("/api/admin/ratings", payload, "POST");
  ratingForm.reset();
  await refreshState();
});

adminUserSelect.addEventListener("change", async (event) => {
  await refreshState(event.target.value);
});

document.addEventListener("click", async (event) => {
  const userId = event.target.getAttribute("data-delete-user");
  const productId = event.target.getAttribute("data-delete-product");
  const ratingId = event.target.getAttribute("data-delete-rating");
  const editUserId = event.target.getAttribute("data-edit-user");
  const editProductId = event.target.getAttribute("data-edit-product");
  const editRatingId = event.target.getAttribute("data-edit-rating");

  if (userId) {
    await fetch(`/api/admin/users/${userId}`, { method: "DELETE" });
    await refreshState();
  }

  if (productId) {
    await fetch(`/api/admin/products/${productId}`, { method: "DELETE" });
    await refreshState();
  }

  if (ratingId) {
    await fetch(`/api/admin/ratings/${ratingId}`, { method: "DELETE" });
    await refreshState();
  }

  if (editUserId) {
    const user = currentState.users.find((item) => String(item.id) === String(editUserId));
    const payload = user ? promptEditUser(user) : null;
    if (!payload) return;
    await submitJson(`/api/admin/users/${editUserId}`, payload, "PUT");
    await refreshState(adminUserSelect.value);
  }

  if (editProductId) {
    const product = currentState.products.find((item) => String(item.id) === String(editProductId));
    const payload = product ? promptEditProduct(product) : null;
    if (!payload) return;
    await submitJson(`/api/admin/products/${editProductId}`, payload, "PUT");
    await refreshState(adminUserSelect.value);
  }

  if (editRatingId) {
    const rating = currentState.ratings.find((item) => String(item.id) === String(editRatingId));
    const payload = rating ? promptEditRating(rating) : null;
    if (!payload) return;
    await submitJson(`/api/admin/ratings/${editRatingId}`, payload, "PUT");
    await refreshState(adminUserSelect.value);
  }
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    tabButtons.forEach((item) => item.classList.remove("active"));
    tabPanels.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`.table-panel[data-panel="${button.dataset.tab}"]`).classList.add("active");
  });
});

renderState(initialAdminData);