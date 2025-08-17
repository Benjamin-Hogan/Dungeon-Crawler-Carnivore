// Elements
const cameraSel = document.getElementById("cameraSel");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const captureBtn = document.getElementById("captureBtn");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const upcManual = document.getElementById("upcManual");
const upcBtn = document.getElementById("upcBtn");
const voiceChk = document.getElementById("voiceChk");
const metaEl = document.getElementById("meta");
const roastEl = document.getElementById("roast");
const nutriEl = document.getElementById("nutri");

let stream = null;
let running = false;
let bestLabelBox = null;
let bestBarcodeBox = null;
let currentUPC = null;

// Populate cameras
(async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === "videoinput");
  cameraSel.innerHTML = cams
    .map((c) => `<option value="${c.deviceId}">${c.label || c.deviceId}</option>`)
    .join("");
  if (!cams.length) statusEl.textContent = "No cameras found.";
})();

// Start
startBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: cameraSel.value ? { exact: cameraSel.value } : undefined,
        facingMode: "environment",
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    captureBtn.disabled = false;
    statusEl.textContent = "Scanning…";
    detectLoop();
    requestAnimationFrame(drawLoop);
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Failed to start camera.";
  }
});

// Stop
stopBtn.addEventListener("click", () => {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  bestLabelBox = bestBarcodeBox = null;
  currentUPC = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  captureBtn.disabled = true;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  statusEl.textContent = "Stopped.";
});

// Manual UPC
upcBtn.addEventListener("click", async () => {
  const upc = (upcManual.value || "").replace(/\D/g, "");
  if (!upc) return;
  const j = await postJSON("/api/roast", { upc });
  renderFood(j);
});

// Capture current ROI (priority: barcode; else label)
captureBtn.addEventListener("click", async () => {
  if (!running) return;
  const box = bestBarcodeBox || bestLabelBox;
  if (!box) {
    statusEl.textContent = "No ROI yet. Hold steady.";
    return;
  }
  const crop = cropBoxFromVideo(box);
  const fd = new FormData();
  fd.append("file", crop, "roi.jpg");
  fd.append("type", bestBarcodeBox ? "barcode" : "label");
  if (currentUPC) fd.append("upc", currentUPC);
  fd.append("bbox", JSON.stringify([box.x, box.y, box.w, box.h]));
  statusEl.textContent = "Analyzing…";
  const res = await fetch("/api/analyze", { method: "POST", body: fd });
  renderFood(await res.json());
  statusEl.textContent = "Ready.";
});

// ------------- Real-time processing -------------

async function detectLoop() {
  if (!running) return;
  try {
    const fd = new FormData();
    fd.append("file", frameToBlob(), "frame.jpg");
    const res = await fetch("/api/detect", { method: "POST", body: fd });
    const j = await res.json();
    bestLabelBox = j.label
      ? { x: j.label.bbox[0], y: j.label.bbox[1], w: j.label.bbox[2], h: j.label.bbox[3] }
      : null;
    if (j.barcode) {
      bestBarcodeBox = {
        x: j.barcode.bbox[0],
        y: j.barcode.bbox[1],
        w: j.barcode.bbox[2],
        h: j.barcode.bbox[3],
        conf: 0.9,
      };
      currentUPC = j.barcode.upc || null;
    } else {
      bestBarcodeBox = null;
    }
  } catch (_e) {
    /* ignore */
  }
  // run detection frequently for smoother boxes
  setTimeout(detectLoop, 300);
}

function drawLoop() {
  if (!running) return;
  drawOverlay();
  requestAnimationFrame(drawLoop);
}

function drawOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (bestBarcodeBox) {
    ctx.strokeStyle = "rgba(0,230,118,0.95)";
    ctx.lineWidth = 3;
    rect(bestBarcodeBox);
    text("BARCODE", bestBarcodeBox.x, bestBarcodeBox.y - 6, "#00e676");
  }
  if (bestLabelBox) {
    ctx.strokeStyle = "rgba(255,119,233,0.95)";
    ctx.lineWidth = 3;
    rect(bestLabelBox);
    text("NUTRITION LABEL", bestLabelBox.x, bestLabelBox.y - 6, "#ff77e9");
  }
}
function rect(b) {
  ctx.strokeRect(b.x, b.y, b.w, b.h);
}
function text(t, x, y, color) {
  ctx.fillStyle = color;
  ctx.font = "bold 14px system-ui";
  ctx.fillText(t, x + 2, Math.max(14, y));
}

// Capture helpers
function frameToBlob() {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  const cctx = c.getContext("2d");
  cctx.drawImage(video, 0, 0, c.width, c.height);
  return dataURLtoBlob(c.toDataURL("image/jpeg", 0.85));
}

function cropBoxFromVideo(box) {
  const c = document.createElement("canvas");
  c.width = Math.max(1, box.w);
  c.height = Math.max(1, box.h);
  const cctx = c.getContext("2d");
  cctx.drawImage(video, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
  return dataURLtoBlob(c.toDataURL("image/jpeg", 0.9));
}

function dataURLtoBlob(dataUrl) {
  const bstr = atob(dataUrl.split(",")[1]);
  let n = bstr.length;
  const u8 = new Uint8Array(n);
  while (n--) u8[n] = bstr.charCodeAt(n);
  return new Blob([u8], { type: "image/jpeg" });
}

// Backend helpers
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}
function speak(t) {
  if (!voiceChk.checked || !("speechSynthesis" in window) || !t) return;
  const u = new SpeechSynthesisUtterance(t);
  u.rate = 1.0;
  u.pitch = 0.95;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}
function renderFood(j) {
  metaEl.textContent = j.product
    ? `${j.product.name || "Unknown"} ${
        j.product.brand ? "• " + j.product.brand : ""
      } ${j.product.upc ? "• UPC " + j.product.upc : ""}`
    : "";
  roastEl.textContent = j.roast || j.error || "";
  nutriEl.textContent = JSON.stringify(j.nutrition || {}, null, 2);
  speak(j.roast);
}

