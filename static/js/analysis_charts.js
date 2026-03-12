// analysis_charts.js — reads JSON data from #analysis-data and renders Chart.js charts
(function () {
  var raw = document.getElementById('analysis-data');
  if (!raw) return;
  var d = JSON.parse(raw.textContent);

  Chart.defaults.color = '#6b7280';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';

  // Macronutrient Patterns (stacked bar)
  new Chart(document.getElementById('macroPatternChart'), {
    type: 'bar',
    data: {
      labels: d.labels,
      datasets: [
        { label: 'Protein', data: d.protein, backgroundColor: 'rgba(163,230,53,0.75)', borderRadius: 4, borderSkipped: false },
        { label: 'Carbs', data: d.carbs, backgroundColor: 'rgba(34,211,238,0.65)', borderRadius: 4, borderSkipped: false },
        { label: 'Fat', data: d.fat, backgroundColor: 'rgba(251,191,36,0.65)', borderRadius: 4, borderSkipped: false }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { display: false } },
        y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true }
      },
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, padding: 16 } } }
    }
  });

  // Sugar Intake (line)
  new Chart(document.getElementById('sugarChart'), {
    type: 'line',
    data: {
      labels: d.labels,
      datasets: [{
        label: 'Sugar (g)',
        data: d.sugar,
        borderColor: '#fbbf24',
        backgroundColor: 'rgba(251,191,36,0.08)',
        tension: 0.4,
        fill: true,
        pointRadius: 4,
        pointBackgroundColor: '#fbbf24'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true }
      }
    }
  });

  // Weekly Calories vs Target (bar + line)
  new Chart(document.getElementById('weeklyCalChart'), {
    type: 'bar',
    data: {
      labels: d.labels,
      datasets: [
        { label: 'Calories', data: d.calories, backgroundColor: 'rgba(163,230,53,0.7)', borderRadius: 4, borderSkipped: false },
        { label: 'Target', data: Array(7).fill(d.dailyGoal), type: 'line', borderColor: 'rgba(34,211,238,0.6)', borderDash: [4, 4], pointRadius: 0, fill: false }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, padding: 14 } } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true }
      }
    }
  });

  // Nutrient Balance (doughnut)
  new Chart(document.getElementById('balanceChart'), {
    type: 'doughnut',
    data: {
      labels: ['Protein', 'Carbs', 'Fat'],
      datasets: [{
        data: [d.protPct, d.carbPct, d.fatPct],
        backgroundColor: ['rgba(163,230,53,0.8)', 'rgba(34,211,238,0.8)', 'rgba(251,191,36,0.8)'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, padding: 14 } } },
      cutout: '62%'
    }
  });
})();
