/**
 * MSST Web Port – Frontend logic
 * All heavy work (model inference) happens on the server.
 */

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const dropzone = $("#dropzone");
  const audioFileInput = $("#audioFile");
  const fileNameEl = $("#fileName");
  const runBtn = $("#runBtn");
  const clearBtn = $("#clearBtn");
  const statusCard = $("#statusCard");
  const statusMessage = $("#statusMessage");
  const progressFill = $("#progressFill");
  const resultsCard = $("#resultsCard");
  const resultsMeta = $("#resultsMeta");
  const stemsList = $("#stemsList");
  const spinner = $("#spinner");

  let selectedFile = null;

  // ---------------------------------------------------------------------------
  // Dropzone
  // ---------------------------------------------------------------------------
  dropzone.addEventListener("click", () => audioFileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });

  audioFileInput.addEventListener("change", () => {
    if (audioFileInput.files.length) handleFile(audioFileInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    fileNameEl.textContent = `${file.name} (${formatBytes(file.size)})`;
    runBtn.disabled = false;
    resultsCard.classList.add("hidden");
    statusCard.classList.add("hidden");
  }

  // ---------------------------------------------------------------------------
  // Run separation
  // ---------------------------------------------------------------------------
  runBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    const modelType = $("#modelType").value;
    const configPath = $("#configPath").value;
    const checkpointPath = $("#checkpointPath").value;
    const targetInstrument = $("#targetInstrument").value.trim();
    const useTta = $("#useTta").checked;
    const extractInstrumental = $("#extractInstrumental").checked;

    if (!configPath) {
      alert("Please select a config YAML file.");
      return;
    }

    // UI state
    runBtn.disabled = true;
    spinner.classList.remove("hidden");
    statusCard.classList.remove("hidden");
    resultsCard.classList.add("hidden");
    statusMessage.textContent = "Uploading and running separation on the server… this may take a while.";
    statusMessage.classList.remove("error-text");
    progressFill.style.width = "15%";

    // Fake progress while waiting (real progress would need SSE/websocket)
    let progress = 15;
    const progressInterval = setInterval(() => {
      progress = Math.min(progress + Math.random() * 8, 90);
      progressFill.style.width = progress + "%";
    }, 1200);

    const formData = new FormData();
    formData.append("audio", selectedFile);
    formData.append("model_type", modelType);
    formData.append("config_path", configPath);
    if (checkpointPath) formData.append("checkpoint_path", checkpointPath);
    if (targetInstrument) formData.append("target_instrument", targetInstrument);
    formData.append("use_tta", useTta);
    formData.append("extract_instrumental", extractInstrumental);

    try {
      const resp = await fetch("/api/separate", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);
      progressFill.style.width = "100%";

      const data = await resp.json();

      if (!resp.ok || data.error) {
        throw new Error(data.error || "Unknown server error");
      }

      statusMessage.textContent = `Done in ${data.elapsed_seconds}s on ${data.device}`;
      showResults(data);
    } catch (err) {
      clearInterval(progressInterval);
      statusMessage.textContent = "Error: " + err.message;
      statusMessage.classList.add("error-text");
      progressFill.style.width = "0%";
      console.error(err);
    } finally {
      runBtn.disabled = false;
      spinner.classList.add("hidden");
    }
  });

  function showResults(data) {
    resultsCard.classList.remove("hidden");

    resultsMeta.innerHTML = `
      <span>Duration: ${data.duration}s</span>
      <span>Sample rate: ${data.sample_rate} Hz</span>
      <span>Stems: ${data.instruments.join(", ")}</span>
      <span>Job: ${data.job_id}</span>
    `;

    stemsList.innerHTML = "";
    for (const [instr, info] of Object.entries(data.stems)) {
      const item = document.createElement("div");
      item.className = "stem-item";
      item.innerHTML = `
        <span class="stem-name">${instr}</span>
        <a href="${info.url}" download="${info.filename}">Download WAV</a>
      `;
      stemsList.appendChild(item);
    }
  }

  // ---------------------------------------------------------------------------
  // Clear cache
  // ---------------------------------------------------------------------------
  clearBtn.addEventListener("click", async () => {
    try {
      const resp = await fetch("/api/clear_cache", { method: "POST" });
      const data = await resp.json();
      alert(data.status || "Cache cleared");
    } catch (e) {
      alert("Failed to clear cache");
    }
  });

  // ---------------------------------------------------------------------------
  // Utils
  // ---------------------------------------------------------------------------
  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  // Health check on load
  fetch("/api/health")
    .then((r) => r.json())
    .then((d) => {
      const badge = $("#deviceBadge");
      if (badge) {
        badge.textContent = `Device: ${d.device} · Torch ${d.torch_version}`;
      }
    })
    .catch(() => {});
})();
