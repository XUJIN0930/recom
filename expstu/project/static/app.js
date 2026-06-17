const initialData = window.__INITIAL_DATA__;

const userSelect = document.getElementById("userSelect");
const recommendationList = document.getElementById("recommendationList");
const metricCards = document.getElementById("metricCards");
const topProductTable = document.getElementById("topProductTable");

function formatCurrency(value) {
  return `¥${Number(value).toFixed(0)}`;
}

function renderMetricCards(summary) {
  metricCards.innerHTML = "";
  const cards = [
    { label: "用户数", value: summary.user_count, note: "覆盖真实与模拟用户" },
    { label: "商品数", value: summary.product_count, note: "包含大数据与平台组件" },
    { label: "评分数", value: summary.rating_count, note: "用于冷启动与协同过滤" },
    { label: "平均评分", value: summary.avg_rating, note: "反映全局偏好强度" },
  ];

  cards.forEach((card) => {
    const element = document.createElement("div");
    element.className = "metric";
    element.innerHTML = `
      <div class="metric-label">${card.label}</div>
      <div class="metric-value">${card.value}</div>
      <div class="metric-note">${card.note}</div>
    `;
    metricCards.appendChild(element);
  });
}

function renderUsers(users, selectedUserId) {
  userSelect.innerHTML = "";
  users.forEach((user) => {
    const option = document.createElement("option");
    option.value = user.id;
    option.textContent = `${user.name} · ${user.city} · ${user.loyalty_level}`;
    if (Number(user.id) === Number(selectedUserId)) {
      option.selected = true;
    }
    userSelect.appendChild(option);
  });
}

function renderRecommendations(recommendations) {
  recommendationList.innerHTML = "";
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
      <div class="reason-list">
        ${item.reasons.map((reason) => `<span class="reason">${reason}</span>`).join("")}
      </div>
      <div class="score-detail">
        <div>协同过滤：${item.score_detail.collaborative_score.toFixed(2)}</div>
        <div>类目偏好：${item.score_detail.content_score.toFixed(2)}</div>
        <div>热度分：${item.score_detail.popularity_score.toFixed(2)}</div>
      </div>
    `;
    recommendationList.appendChild(element);
  });
}

function renderTopProducts(products) {
  topProductTable.innerHTML = "";
  products.forEach((product) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${product.name}</td>
      <td>${product.category}</td>
      <td>${product.avg_rating.toFixed(2)}</td>
      <td>${product.count}</td>
      <td>${formatCurrency(product.price)}</td>
    `;
    topProductTable.appendChild(row);
  });
}

function renderBarChart(container, items, palette) {
  const maxValue = Math.max(...items.map((item) => item.value), 1);
  container.innerHTML = "";

  items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "chart-row";

    const barWidth = Math.max(8, (item.value / maxValue) * 100);
    const color = palette[index % palette.length];

    row.innerHTML = `
      <div class="chart-row-head">
        <span>${item.name}</span>
        <strong>${item.value}</strong>
      </div>
      <div class="chart-track">
        <div class="chart-fill" style="width: ${barWidth}%; background: ${color};"></div>
      </div>
    `;

    container.appendChild(row);
  });
}

function drawCharts(data) {
  renderBarChart(
    document.getElementById("categoryChart"),
    data.category_breakdown,
    ["#7ce7c1", "#66a6ff", "#ffb86b", "#c08bff", "#ff7a90", "#5dd5f0"],
  );

  renderBarChart(
    document.getElementById("ratingChart"),
    data.rating_distribution,
    ["#66a6ff", "#7ce7c1", "#ffb86b"],
  );
}

async function loadDashboard(userId) {
  const response = await fetch(`/api/dashboard?user_id=${userId}`);
  const data = await response.json();
  renderMetricCards(data.summary);
  renderUsers(data.users, data.selected_user.id);
  renderRecommendations(data.recommendations);
  renderTopProducts(data.top_products);
  drawCharts(data);
}

userSelect.addEventListener("change", (event) => {
  loadDashboard(event.target.value);
});

renderMetricCards(initialData.summary);
renderUsers(initialData.users, initialData.selected_user.id);
renderRecommendations(initialData.recommendations);
renderTopProducts(initialData.top_products);
drawCharts(initialData);
