document.addEventListener("DOMContentLoaded", function () {
  // --- DOM Elements ---
  const tabButtons = document.querySelectorAll(".tab-btn");
  const startWebcamBtn = document.getElementById("start-webcam");
  const webcamFeed = document.getElementById("webcam-feed");
  const webcamPlaceholder = document.getElementById("webcam-placeholder");
  const imageUpload = document.getElementById("image-upload");
  const imagePreview = document.getElementById("image-preview");
  const imagePlaceholder = document.getElementById("image-placeholder");
  const analyzeImageBtn = document.getElementById("analyze-image");
  const videoUpload = document.getElementById("video-upload");
  const videoPreview = document.getElementById("video-preview");
  const videoStreamPreview = document.getElementById("video-stream-preview");
  const videoPlaceholder = document.getElementById("video-placeholder");
  const analyzeVideoBtn = document.getElementById("analyze-video");
  const resetDataBtn = document.getElementById("reset-data");
  const sessionTimeEl = document.getElementById("session-time");
  const totalDetectionsEl = document.getElementById("total-detections");
  const analyticsContent = document.getElementById("analytics-content");
  const analyticsEmptyState = document.querySelector(
    "#analytics-tab .empty-state"
  );
  const legendContainer = document.getElementById("emotion-legend");

  // --- State Variables ---
  let webcamActive = false;
  let emotionChart = null;
  let sessionTimer = null;
  let secondsElapsed = 0;
  let emotionCounts = {};
  let isSessionActive = false;
  let emotionPollInterval = null;
  let videoPollInterval = null;
  let sessionStartTime = null;
  let sessionType = null; // 'webcam', 'image', 'video'
  let historyData = [];
  let currentImageFile = null;
  let currentVideoFile = null;

  const EMOTION_COLORS = {
    Angry: "#ef4444",
    Disgust: "#8b5cf6",
    Fear: "#f59e0b",
    Happy: "#10b981",
    Neutral: "#6b7280",
    Sad: "#3b82f6",
    Surprise: "#ec4899",
  };

  // --- Initialization ---
  Chart.register(ChartDataLabels);
  resetAnalytics();
  loadHistory();
  renderHistory();

  // --- Event Listeners ---
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const tabName = button.getAttribute("data-tab");
      const parentPanel = button.closest(".left-panel, .right-panel");
      parentPanel
        .querySelectorAll(".tab-btn")
        .forEach((btn) => btn.classList.remove("active"));
      parentPanel
        .querySelectorAll(".tab-content")
        .forEach((content) => content.classList.remove("active"));
      button.classList.add("active");
      parentPanel.querySelector(`#${tabName}-tab`).classList.add("active");
    });
  });

  document
    .getElementById("select-image")
    .addEventListener("click", () => imageUpload.click());
  document
    .getElementById("select-video")
    .addEventListener("click", () => videoUpload.click());
  imageUpload.addEventListener("change", handleFileSelect);
  videoUpload.addEventListener("change", handleFileSelect);
  startWebcamBtn.addEventListener("click", handleWebcamToggle);
  analyzeImageBtn.addEventListener("click", handleImageAnalysis);
  analyzeVideoBtn.addEventListener("click", handleVideoAnalysis);
  resetDataBtn.addEventListener("click", () => resetAnalytics(true));

  // --- Core Functions ---

  function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    const isVideo = file.type.startsWith("video/");
    const previewEl = isVideo ? videoPreview : imagePreview;
    const placeholderEl = isVideo ? videoPlaceholder : imagePlaceholder;
    const analyzeBtn = isVideo ? analyzeVideoBtn : analyzeImageBtn;

    if (isVideo) {
      currentVideoFile = file;
      placeholderEl.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i><p>Đang tải video...</p>';
      const videoURL = URL.createObjectURL(file);
      previewEl.src = videoURL;
      previewEl.onloadedmetadata = () => {
        placeholderEl.style.display = "none";
        previewEl.style.display = "block";
        analyzeBtn.style.display = "inline-flex";
      };
    } else {
      currentImageFile = file;
      const reader = new FileReader();
      reader.onload = (event) => {
        previewEl.src = event.target.result;
        placeholderEl.style.display = "none";
        previewEl.style.display = "block";
        analyzeBtn.style.display = "inline-flex";
      };
      reader.readAsDataURL(file);
    }
  }

  function handleWebcamToggle() {
    webcamActive = !webcamActive;
    if (webcamActive) {
      startWebcamBtn.innerHTML =
        '<span class="loading"></span> Đang khởi động...';
      startWebcamBtn.disabled = true;
      resetAnalytics(false);
      sessionType = "webcam";
      sessionStartTime = new Date().toISOString();

      fetch("/start_webcam", { method: "POST" })
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            webcamFeed.src = "/video_feed";
            webcamPlaceholder.style.display = "none";
            webcamFeed.style.display = "block";
            startWebcamBtn.innerHTML = '<i class="fas fa-stop"></i> Dừng lại';
            startWebcamBtn.style.background = "#ef4444";
            startWebcamBtn.disabled = false;
            startEmotionPolling();
          } else {
            alert("Lỗi khởi động camera: " + data.error);
            webcamActive = false;
            startWebcamBtn.innerHTML = '<i class="fas fa-play"></i> Bắt đầu';
            startWebcamBtn.disabled = false;
          }
        });
    } else {
      addToHistory();
      stopSession();
      stopEmotionPolling();
      webcamFeed.src = "";
      webcamPlaceholder.style.display = "block";
      webcamFeed.style.display = "none";
      fetch("/stop_webcam", { method: "POST" });
      startWebcamBtn.innerHTML = '<i class="fas fa-play"></i> Bắt đầu';
      startWebcamBtn.style.background = "#5b9bd5";
    }
  }

  function handleImageAnalysis() {
    const file = currentImageFile || imageUpload.files[0];
    if (!file) return alert("Vui lòng chọn ảnh trước!");

    sessionType = "image";
    sessionStartTime = new Date().toISOString();
    const formData = new FormData();
    formData.append("image", file);

    analyzeImageBtn.innerHTML = '<span class="loading"></span> Đang phân tích...';
    analyzeImageBtn.disabled = true;
    startSession();

    fetch("/analyze_image", { method: "POST", body: formData })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          imagePreview.src = `/static/uploads/${data.filename}?t=${new Date().getTime()}`;
          updateAnalytics(data.emotions);
        } else {
          alert("Lỗi: " + data.error);
        }
      })
      .catch((error) => alert("Lỗi kết nối: " + error.message))
      .finally(() => {
        analyzeImageBtn.innerHTML = '<i class="fas fa-search"></i> Phân tích';
        analyzeImageBtn.disabled = false;
        addToHistory();
        stopSession();
      });
  }

  function handleVideoAnalysis() {
    const file = currentVideoFile || videoUpload.files[0];
    if (!file) return alert("Vui lòng chọn video trước!");

    analyzeVideoBtn.innerHTML = '<span class="loading"></span> Đang xử lý...';
    analyzeVideoBtn.disabled = true;

    const formData = new FormData();
    formData.append("video", file);

    fetch("/analyze_video", { method: "POST", body: formData })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          resetAnalytics(false);
          sessionType = "video";
          sessionStartTime = new Date().toISOString();

          videoPreview.style.display = "none"; // Hide original video
          videoStreamPreview.style.display = "block";
          videoStreamPreview.src = `/video_analysis_feed/${data.filename}`;
          
          // Start analysis only when the video actually starts playing
          videoPreview.onplay = () => {
            startSession();
            startVideoEmotionPolling();

            // Sync timer with video's currentTime using requestAnimationFrame
            function updateVideoTime() {
              if (!videoPreview.paused && !videoPreview.ended) {
                const currentTime = Math.floor(videoPreview.currentTime);
                sessionTimeEl.textContent = `${currentTime}s`;
                secondsElapsed = currentTime; // Keep state in sync for history
                requestAnimationFrame(updateVideoTime);
              }
            }
            updateVideoTime();
          };

          videoPreview.play(); // Play hidden video for audio

          videoPreview.onended = () => {
            stopVideoEmotionPolling();
            stopSession();
            addToHistory();
            videoStreamPreview.src = ""; // Clear stream
            videoStreamPreview.style.display = "none";
            videoPreview.style.display = "block"; // Show original video again
            analyzeVideoBtn.innerHTML = '<i class="fas fa-search"></i> Phân tích';
            analyzeVideoBtn.disabled = false;
          };
        } else {
          alert("Lỗi: " + data.error);
          analyzeVideoBtn.innerHTML = '<i class="fas fa-search"></i> Phân tích';
          analyzeVideoBtn.disabled = false;
        }
      })
      .catch((error) => {
        alert("Lỗi kết nối: " + error.message);
        analyzeVideoBtn.innerHTML = '<i class="fas fa-search"></i> Phân tích';
        analyzeVideoBtn.disabled = false;
      });
  }

  // --- Analytics & UI Functions ---

  function startSession() {
    if (isSessionActive) return;
    isSessionActive = true;
    analyticsEmptyState.style.display = "none";
    analyticsContent.style.display = "flex";
    if (sessionType !== 'video') {
      sessionTimer = setInterval(() => {
        secondsElapsed++;
        sessionTimeEl.textContent = `${secondsElapsed}s`;
      }, 1000);
    }
  }

  function stopSession() {
    clearInterval(sessionTimer);
    isSessionActive = false;
  }

  function startEmotionPolling() {
    emotionPollInterval = setInterval(() => {
      fetch("/get_emotions")
        .then((response) => response.json())
        .then((data) => {
          if (data.success && data.emotions) {
            const hasEmotions = Object.values(data.emotions).some((c) => c > 0);
            if (hasEmotions && !isSessionActive) startSession();
            if (hasEmotions) {
              emotionCounts = data.emotions;
              updateAnalytics(null);
            }
          }
        });
    }, 500);
  }

  function stopEmotionPolling() {
    clearInterval(emotionPollInterval);
    emotionPollInterval = null;
  }

  function startVideoEmotionPolling() {
    videoPollInterval = setInterval(() => {
      fetch("/get_video_emotions")
        .then((response) => response.json())
        .then((data) => {
          if (data.success && data.emotions) {
            emotionCounts = data.emotions;
            updateAnalytics(null);
          }
        });
    }, 500);
  }

  function stopVideoEmotionPolling() {
    clearInterval(videoPollInterval);
    videoPollInterval = null;
  }

  function resetAnalytics(hidePanel = true) {
    stopSession();
    secondsElapsed = 0;
    emotionCounts = {
      Angry: 0, Disgust: 0, Fear: 0, Happy: 0,
      Neutral: 0, Sad: 0, Surprise: 0,
    };
    sessionTimeEl.textContent = "0s";
    totalDetectionsEl.textContent = "0";

    if (hidePanel) {
      analyticsContent.style.display = "none";
      analyticsEmptyState.style.display = "block";
    }
    renderChart();
    renderLegend();
  }

  function updateAnalytics(newEmotions) {
    if (newEmotions) { // For single-batch analysis like images
      emotionCounts = newEmotions;
    }
    const totalDetections = Object.values(emotionCounts).reduce((a, b) => a + b, 0);
    totalDetectionsEl.textContent = totalDetections;
    renderChart();
    renderLegend();
  }

  function renderChart() {
    const ctx = document.getElementById("emotion-chart").getContext("2d");
    const labels = Object.keys(emotionCounts);
    const data = Object.values(emotionCounts);
    const backgroundColors = labels.map((label) => EMOTION_COLORS[label]);

    if (emotionChart) {
      emotionChart.data.datasets[0].data = data;
      emotionChart.update();
    } else {
      emotionChart = new Chart(ctx, {
        type: "doughnut",
        data: {
          labels,
          datasets: [{
            data,
            backgroundColor: backgroundColors,
            borderColor: "#2a344f",
            borderWidth: 3,
          }],
        },
        options: {
          responsive: true,
          cutout: "60%",
          plugins: {
            legend: { display: false },
            datalabels: {
              formatter: (value) => (value > 0 ? value : ""),
              color: "#fff",
              font: { weight: "bold", size: 16 },
            },
          },
        },
      });
    }
  }

  function renderLegend() {
    legendContainer.innerHTML = "";
    Object.entries(EMOTION_COLORS).forEach(([emotion, color]) => {
      legendContainer.innerHTML += `<div class="legend-item"><div class="legend-color" style="background-color: ${color}"></div><span>${emotion}</span></div>`;
    });
  }

  // --- History Functions ---
  function loadHistory() {
    try {
      const saved = localStorage.getItem("emotionHistory");
      if (saved) historyData = JSON.parse(saved);
    } catch (e) { historyData = []; }
  }

  function saveHistory() {
    try {
      localStorage.setItem("emotionHistory", JSON.stringify(historyData));
    } catch (e) { console.error("Error saving history:", e); }
  }

  function addToHistory() {
    const saveHistoryCheckbox = document.getElementById("save-history");
    if (!saveHistoryCheckbox || !saveHistoryCheckbox.checked) return;

    const totalDetections = Object.values(emotionCounts).reduce((a, b) => a + b, 0);
    if (totalDetections === 0) return;

    const session = {
      id: Date.now(),
      type: sessionType || "webcam",
      timestamp: sessionStartTime || new Date().toISOString(),
      duration: secondsElapsed,
      totalDetections: totalDetections,
      emotions: { ...emotionCounts },
      dominantEmotion: getDominantEmotion(),
    };

    historyData.unshift(session);
    if (historyData.length > 50) historyData = historyData.slice(0, 50);
    saveHistory();
    renderHistory();
  }

  function getDominantEmotion() {
    let maxEmotion = "Neutral", maxCount = 0;
    for (const [emotion, count] of Object.entries(emotionCounts)) {
      if (count > maxCount) {
        maxCount = count;
        maxEmotion = emotion;
      }
    }
    return maxEmotion;
  }

  function renderHistory() {
    const historyContent = document.getElementById("history-content");
    const historyEmptyState = document.querySelector("#history-tab .empty-state");

    if (historyData.length === 0) {
      historyEmptyState.style.display = "block";
      historyContent.style.display = "none";
      return;
    }

    historyEmptyState.style.display = "none";
    historyContent.style.display = "block";

    let html = '<div class="history-list">';
    historyData.forEach((session) => {
      const date = new Date(session.timestamp);
      const dateStr = date.toLocaleDateString("vi-VN");
      const timeStr = date.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
      const emotionColor = EMOTION_COLORS[session.dominantEmotion] || "#6b7280";

      html += `
        <div class="history-item">
          <div class="history-header">
            <div class="history-info">
              <span class="history-type">
                <i class="fas fa-${getTypeIcon(session.type)}"></i> 
                ${getTypeName(session.type)}
              </span>
              <span class="history-date">${dateStr} ${timeStr}</span>
            </div>
            <button class="btn-delete" onclick="deleteHistoryItem(${session.id})">
              <i class="fas fa-trash"></i>
            </button>
          </div>
          <div class="history-stats">
            <div class="history-stat">
              <span class="stat-label">Thời gian:</span>
              <span class="stat-value">${session.duration}s</span>
            </div>
            <div class="history-stat">
              <span class="stat-label">Nhận diện:</span>
              <span class="stat-value">${session.totalDetections}</span>
            </div>
            <div class="history-stat">
              <span class="stat-label">Cảm xúc chủ đạo:</span>
              <span class="stat-value" style="color: ${emotionColor}">
                ${session.dominantEmotion}
              </span>
            </div>
          </div>
          <div class="history-emotions">
            ${Object.entries(session.emotions)
              .filter(([_, count]) => count > 0)
              .map(([emotion, count]) => `
              <div class="emotion-badge" style="background-color: ${EMOTION_COLORS[emotion]}20; color: ${EMOTION_COLORS[emotion]}">
                ${emotion}: ${count}
              </div>`)
              .join("")}
          </div>
        </div>`;
    });
    html += '</div>';
    html += `<div class="history-actions"><button class="btn-danger" id="clear-history"><i class="fas fa-trash-alt"></i> Xóa toàn bộ lịch sử</button></div>`;
    historyContent.innerHTML = html;
    document.getElementById("clear-history").addEventListener("click", clearHistory);
  }

  function getTypeIcon(type) {
    return { webcam: "video", image: "image", video: "film" }[type] || "question";
  }

  function getTypeName(type) {
    return { webcam: "Webcam", image: "Ảnh", video: "Video" }[type] || "Không xác định";
  }

  window.deleteHistoryItem = function (id) {
    if (confirm("Bạn có chắc muốn xóa phiên này?")) {
      historyData = historyData.filter((item) => item.id !== id);
      saveHistory();
      renderHistory();
    }
  };

  function clearHistory() {
    if (confirm("Bạn có chắc muốn xóa toàn bộ lịch sử?")) {
      historyData = [];
      saveHistory();
      renderHistory();
    }
  }
});