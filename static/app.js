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
let lastFrameAt = 0;
let bestLabelBox = null;
let bestBarcodeBox = null;
let currentUPC = null;
let zxingReader = null;
let zxingActive = false;

// Populate cameras
(async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === "videoinput");
  cameraSel.innerHTML = cams
    .map(
      (c) => `<option value="${c.deviceId}">${c.label || c.deviceId}</option>`
    )
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
    startZxing(); // barcode
    requestAnimationFrame(loop); // label
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Failed to start camera.";
  }
});

// Stop
stopBtn.addEventListener("click", () => {
  running = false;
  if (zxingActive) stopZxing();
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

// ZXing for barcode (green box)
function startZxing() {
  if (zxingActive) return;
  zxingReader = new ZXing.BrowserMultiFormatReader();
  zxingActive = true;
  const tick = async () => {
    if (!zxingActive) return;
    try {
      const result = await zxingReader.decodeOnceFromVideoElement(video);
      if (result && result.text) {
        currentUPC = result.text.replace(/\D/g, "");
        // Points are often 2 for 1D; build a bbox
        const pts = (result.resultPoints || []).map((p) => ({
          x: p.x,
          y: p.y,
        }));
        if (pts.length >= 2) {
          const minx = Math.min(...pts.map((p) => p.x)),
            maxx = Math.max(...pts.map((p) => p.x));
          const miny = Math.min(...pts.map((p) => p.y)),
            maxy = Math.max(...pts.map((p) => p.y));
          bestBarcodeBox = {
            x: minx - 10,
            y: miny - 20,
            w: maxx - minx + 20,
            h: maxy - miny + 40,
            conf: 0.9,
          };
        }
      }
    } catch (_e) {
      /* try again */
    }
    if (zxingActive) setTimeout(tick, 100); // throttled
  };
  tick();
}
function stopZxing() {
  try {
    zxingReader && zxingReader.reset();
  } catch (_e) {}
  zxingActive = false;
  zxingReader = null;
  currentUPC = null;
  bestBarcodeBox = null;
}

// OpenCV.js loop for label (magenta box)
function loop(ts) {
  if (!running) return;
  if (ts - lastFrameAt > 66) {
    // ~15fps
    lastFrameAt = ts;
    detectLabelBox();
    drawOverlay();
  }
  requestAnimationFrame(loop);
}

function detectLabelBox() {
  if (!window.cv || video.readyState < 2) return;
  const mat = new cv.Mat(video.videoHeight, video.videoWidth, cv.CV_8UC4);
  const cap = new cv.VideoCapture(video);
  cap.read(mat);

  let gray = new cv.Mat();
  cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
  let grad = new cv.Mat();
  let kernel3 = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.morphologyEx(gray, grad, cv.MORPH_GRADIENT, kernel3);
  let thr = new cv.Mat();
  cv.threshold(grad, thr, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
  let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(15, 9));
  let closed = new cv.Mat();
  cv.morphologyEx(thr, closed, cv.MORPH_CLOSE, kernel);

  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(
    closed,
    contours,
    hierarchy,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
  );

  let best = null;
  let bestScore = 0;
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const r = cv.boundingRect(c);
    const area = r.width * r.height;
    if (area < 0.06 * mat.cols * mat.rows) continue; // skip small
    const ar = r.width / (r.height + 1e-6);
    const score =
      (area / (mat.cols * mat.rows)) * (ar > 0.35 && ar < 1.8 ? 1.0 : 0.6);
    if (score > bestScore) {
      bestScore = score;
      best = r;
    }
  }
  bestLabelBox = best
    ? { x: best.x, y: best.y, w: best.width, h: best.height, conf: 0.7 }
    : null;

  [mat, gray, grad, thr, closed, contours, hierarchy, kernel3, kernel].forEach(
    (m) => {
      try {
        m.delete();
      } catch (_e) {}
    }
  );
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

// Crop current box to Blob
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
