let accuracyChart, lossChart, reliabilityChart;
let lastClientReliability = null; // 缓存上一次的 reliability 数据

window.onload = function() {
    createCharts();
    setInterval(fetchData, 1000);  // 每秒获取一次数据
};

async function fetchData() {
    try {
        const resp = await fetch("/metrics");
        if (!resp.ok) {
            console.error("Failed to fetch /metrics", resp);
            return;
        }
        const data = await resp.json();
        updateCharts(data);

        if (data.server_uptime) {
            document.getElementById("server-time").textContent =
                `Server Uptime: ${data.server_uptime}`;
        }
    } catch(err) {
        console.error(err);
    }
}

// **创建所有图表**
function createCharts() {
    const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
    accuracyChart = new Chart(accuracyCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Global Accuracy', data: [], borderColor: 'blue', borderWidth: 2 }] },
        options: { responsive: true, scales: { x: { title: { display: true, text: 'Round' } }, y: { min: 0, max: 1 } } }
    });

    const lossCtx = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Global Loss', data: [], borderColor: 'orange', borderWidth: 2 }] },
        options: { responsive: true, scales: { x: { title: { display: true, text: 'Round' } }, y: { min: 0 } } }
    });

    const reliabilityCtx = document.getElementById('reliabilityChart').getContext('2d');
    reliabilityChart = new Chart(reliabilityCtx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Round' } },
                y: { title: { display: true, text: 'Reliability Score' }, min: 0, max: 1 }
            }
        }
    });
}

// **更新所有图表**
function updateCharts(metrics) {
    if (metrics.client_status) {
        updateNodeStatus(metrics.client_status);
    }

    if (accuracyChart) {
        accuracyChart.data.labels = metrics.accuracy.map((_, i) => i + 1);
        accuracyChart.data.datasets[0].data = metrics.accuracy;

        if (metrics.accuracy.length > 0) {
            let minAccuracy = Math.min(...metrics.accuracy);  // 找到最小值
            accuracyChart.options.scales.y.min = minAccuracy; // 设置 y 轴最小值
        }

        accuracyChart.update();
    }

    if (lossChart) {
        lossChart.data.labels = metrics.loss.map((_, i) => i + 1);
        lossChart.data.datasets[0].data = metrics.loss;
        lossChart.update();
    }

    if (metrics.client_reliability) {
        updateReliabilityChart(metrics.client_reliability);
    }
}

// **更新 Reliability Chart，确保数据结构与 Python 端一致**
function updateReliabilityChart(clientReliability) {
    if (lastClientReliability && deepEqual(clientReliability, lastClientReliability)) {
        return; // Data unchanged, skip update
    }

    let maxRounds = 0;
    let datasets = [];

    // Process each client's reliability data
    Object.keys(clientReliability).forEach((clientId, index) => {
        let roundToEntry = {};

        // Step 1: Build a round-to-entry map, prioritizing "rejoin" status
        clientReliability[clientId].forEach(entry => {
            const round = entry.round;
            const status = entry.status || "unknown";

            // Only override if no entry exists or this is a "rejoin" entry
            if (!(round in roundToEntry) || status === "rejoin") {
                roundToEntry[round] = {
                    reliability: entry.reliability,
                    status: status,
                    keep_weights: entry.keep_weights || false // Handle rejoin cases
                };
            }
            maxRounds = Math.max(maxRounds, round);
        });

        // Step 2: Sort rounds and prepare data points
        const sortedRounds = Object.keys(roundToEntry).map(Number).sort((a, b) => a - b);
        let dataPoints = [];

        // Step 3: Fill gaps for continuity (mimicking Python's _fill_missing_rounds)
        for (let round = 1; round <= maxRounds; round++) {
            if (round in roundToEntry) {
                const entry = roundToEntry[round];
                dataPoints.push({
                    x: round,
                    y: entry.reliability,
                    status: entry.status,
                    keep_weights: entry.keep_weights
                });
            } else if (dataPoints.length > 0) {
                // Fill missing round with the previous reliability, degraded slightly
                const prev = dataPoints[dataPoints.length - 1];
                const newReliability = Math.max(0.1, prev.y - 0.05); // Mimic Python decay
                dataPoints.push({
                    x: round,
                    y: newReliability,
                    status: prev.status === "failure" ? "failure" : "missed",
                    keep_weights: false
                });
            } else {
                // No prior data, use default
                dataPoints.push({ x: round, y: 0.5, status: "missed", keep_weights: false });
            }
        }

        // Step 4: Create dataset with points and custom styling
        datasets.push({
            label: `Client ${clientId.substring(0, 8)}`, // Shorten ID like Python
            data: dataPoints,
            borderColor: `hsl(${index * 60}, 70%, 50%)`,
            borderWidth: 1.5,
            fill: false,
            pointRadius: 5,
            pointStyle: point => {
                switch (point.raw.status) {
                    case "success": return "circle";
                    case "failure": return "cross";
                    case "rejoin": return point.raw.keep_weights ? "triangle" : "triangleDown";
                    default: return "circle"; // Fallback for "missed" or "pending"
                }
            },
            pointBackgroundColor: `hsl(${index * 60}, 70%, 50%)`,
            pointBorderColor: point => {
                switch (point.raw.status) {
                    case "failure": return "red";
                    case "rejoin": return point.raw.keep_weights ? "green" : "blue";
                    default: return "gray";
                }
            },
            showLine: true
        });
    });

    // Update chart
    reliabilityChart.data.labels = Array.from({ length: maxRounds }, (_, i) => i + 1);
    reliabilityChart.data.datasets = datasets;
    reliabilityChart.options.scales.y.min = 0;
    reliabilityChart.options.scales.y.max = 1.1; // Match Python's ylim
    reliabilityChart.update();

    // Cache the current data
    lastClientReliability = JSON.parse(JSON.stringify(clientReliability));
}

// **深度比较对象是否相同，避免不必要刷新**
function deepEqual(obj1, obj2) {
    if (obj1 === obj2) return true;
    if (typeof obj1 !== 'object' || typeof obj2 !== 'object' || obj1 === null || obj2 === null) return false;

    let keys1 = Object.keys(obj1);
    let keys2 = Object.keys(obj2);
    if (keys1.length !== keys2.length) return false;

    for (let key of keys1) {
        if (!keys2.includes(key) || !deepEqual(obj1[key], obj2[key])) {
            return false;
        }
    }
    return true;
}

// **更新客户端状态**
function updateNodeStatus(clientStatus) {
    let html = `
      <table border="1" cellpadding="6" style="border-collapse: collapse;">
        <tr>
          <th>Client ID</th>
          <th>Status</th>
          <th>Last Active Round</th>
          <th>Missed Rounds</th>
        </tr>
    `;
    for (const clientId in clientStatus) {
        const statusObj = clientStatus[clientId];
        const isActive = statusObj.active;
        const lastRound = statusObj.last_active_round;
        const missed = statusObj.missed_rounds;

        const color = isActive ? "green" : "gray";
        const statusText = isActive ? "Online" : "Offline";

        html += `
          <tr>
            <td>${clientId}</td>
            <td style="color: ${color}; font-weight:bold;">${statusText}</td>
            <td>${lastRound}</td>
            <td>${missed}</td>
          </tr>
        `;
    }

    html += `</table>`;

    document.getElementById("node-status").innerHTML = html;
}
