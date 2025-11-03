/* =======================
   BGone – app.js (FINAL)
   ======================= */

/* ---- UI елементи ---- */
const file = document.getElementById("file");
const cameraBtn = document.getElementById("camera");
const picker = document.getElementById("picker");
const uploadBgBtn = document.getElementById("uploadBgBtn");
const bgFile = document.getElementById("bgFile");
const savePng = document.getElementById("savePng");
const autoRemoveBtn = document.getElementById("autoRemove"); // "Автовидалення фону (AI)"
const featherInput = document.getElementById("feather");      // 0..3
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

/* ---- Стани ---- */
let img = new Image();          // вихідне зображення
let bgImg = null;               // фон-картинка
let session = null;             // ONNX Runtime сесія
let lastMat = null;             // { alpha: Float32Array, width, height }

/* ---- Робоча канва для прозорого форграунду ---- */
const fgCanvas = document.createElement("canvas");
const fgCtx = fgCanvas.getContext("2d", { willReadFrequently: true });

/* ---- Налаштування ---- */
const MODEL_PATH = "/bgone/models/u2netp.onnx";
const INVERT_ALPHA = false; // якщо побачиш, що "вирізається навпаки" — постав true

/* =======================
   1) Базове відмальовування
   ======================= */
function draw() {
  if (!img.src) return;
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;

  if (bgImg) ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
  else {
    ctx.fillStyle = picker.value || "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

/* =======================
   2) Підготовка вхідного тензора 1×3×320×320 (NCHW)
   ======================= */
function imageToTensor(imageBitmap, target = 320) {
  const tmp = document.createElement("canvas");
  tmp.width = target;
  tmp.height = target;
  const tctx = tmp.getContext("2d", { willReadFrequently: true });

  const scale = Math.min(target / imageBitmap.width, target / imageBitmap.height);
  const w = Math.round(imageBitmap.width * scale);
  const h = Math.round(imageBitmap.height * scale);
  const ox = Math.floor((target - w) / 2);
  const oy = Math.floor((target - h) / 2);

  tctx.fillStyle = "black";
  tctx.fillRect(0, 0, target, target);
  tctx.drawImage(imageBitmap, ox, oy, w, h);

  const { data } = tctx.getImageData(0, 0, target, target);
  const floatData = new Float32Array(1 * 3 * target * target);
  let idx = 0;
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < target; y++) {
      for (let x = 0; x < target; x++) {
        const i = (y * target + x) * 4;
        floatData[idx++] = data[i + c] / 255;
      }
    }
  }
  return { floatData, ox, oy, w, h, target };
}

/* =======================
   3) Завантаження моделі (1 раз)
   ======================= */
async function loadModel() {
  if (session) return;
  if (!globalThis.ort) {
    throw new Error("ONNX Runtime не підключено. Додай ort.min.js у index.html");
  }
  session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["wasm"]
  });
  console.log("U2Netp loaded inputs:", session.inputNames, "outputs:", session.outputNames);
}

/* =======================
   4) Інференс: усереднюємо ВСІ виходи -> маска 0..1
   ======================= */
async function runMatting(originalImage) {
  await loadModel();

  const ib = await createImageBitmap(originalImage);
  const { floatData, ox, oy, w, h, target } = imageToTensor(ib, 320);

  const inputName = session.inputNames[0];
  const outputNames = session.outputNames.slice(); // очікуємо ~7 штук

  const inputTensor = new ort.Tensor("float32", floatData, [1, 3, target, target]);
  const results = await session.run({ [inputName]: inputTensor });

  // 1) Витягуємо всі виходи, sigmoid -> 0..1, усереднюємо
  let merged = null;
  let H = 320, W = 320;

  outputNames.forEach((name) => {
    const out = results[name];
    if (!out) return;
    const data = out.data;
    const dims = out.dims || out.dims_;

    if (Array.isArray(dims) && dims.length === 4) {
      if (dims[2] && dims[3]) { H = dims[2]; W = dims[3]; }
      else if (dims[1] && dims[2]) { H = dims[1]; W = dims[2]; }
    }

    const probs = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      probs[i] = 1 / (1 + Math.exp(-v)); // sigmoid
    }

    if (!merged) merged = probs;
    else {
      for (let i = 0; i < merged.length; i++) merged[i] += probs[i];
    }
  });

  if (!merged) throw new Error("Модель не повернула виходів.");

  for (let i = 0; i < merged.length; i++) merged[i] /= outputNames.length;

  // Невеликий лог для діагностики
  let min = 1, max = 0;
  for (let i = 0; i < merged.length; i++) { if (merged[i] < min) min = merged[i]; if (merged[i] > max) max = merged[i]; }
  console.log(`Mat range: min=${min.toFixed(3)} max=${max.toFixed(3)} (H=${H}, W=${W})`);

  // 2) Малюємо маску H×W
  const tmp = document.createElement("canvas");
  tmp.width = W; tmp.height = H;
  const tctx = tmp.getContext("2d");
  const imgData = tctx.createImageData(W, H);
  for (let i = 0; i < merged.length; i++) {
    const g = Math.max(0, Math.min(255, Math.round(merged[i] * 255)));
    const k = i * 4;
    imgData.data[k] = g;
    imgData.data[k + 1] = g;
    imgData.data[k + 2] = g;
    imgData.data[k + 3] = 255;
  }
  tctx.putImageData(imgData, 0, 0);

  // 3) Вирізаємо центральну частину (ox,oy,w,h) і масштабуємо під оригінал
  const sub = document.createElement("canvas");
  sub.width = w; sub.height = h;
  sub.getContext("2d").drawImage(tmp, ox, oy, w, h, 0, 0, w, h);

  const OW = originalImage.naturalWidth || originalImage.width;
  const OH = originalImage.naturalHeight || originalImage.height;
  const full = document.createElement("canvas");
  full.width = OW; full.height = OH;
  const fctx = full.getContext("2d");
  fctx.imageSmoothingEnabled = true;
  fctx.imageSmoothingQuality = "high";
  fctx.drawImage(sub, 0, 0, OW, OH);

  const fullData = fctx.getImageData(0, 0, OW, OH).data;
  const alpha = new Float32Array(OW * OH);
  for (let i = 0, j = 0; i < fullData.length; i += 4, j++) {
    let a = fullData[i] / 255; // беремо будь-який канал
    if (INVERT_ALPHA) a = 1 - a;
    alpha[j] = a;
  }

  return { alpha, width: OW, height: OH };
}

/* =======================
   5) Композиція: фон + прозорий форграунд
   ======================= */
function compositeWithAlpha(originalImage, alphaArr, w, h) {
  canvas.width = w;
  canvas.height = h;

  // фон
  if (bgImg) ctx.drawImage(bgImg, 0, 0, w, h);
  else {
    ctx.fillStyle = picker.value || "#ffffff";
    ctx.fillRect(0, 0, w, h);
  }

  // форграунд з альфою
  fgCanvas.width = w;
  fgCanvas.height = h;
  fgCtx.clearRect(0, 0, w, h);
  fgCtx.drawImage(originalImage, 0, 0, w, h);

  const fg = fgCtx.getImageData(0, 0, w, h);
  const d = fg.data;
  const feather = Number(featherInput ? featherInput.value : 1.5); // 0..3
  const gamma = 1 / (1 + feather / 2);

  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    let a = alphaArr[p];
    a = Math.pow(a, gamma); // м’якість краю
    d[i + 3] = Math.max(0, Math.min(255, Math.round(a * 255)));
  }
  fgCtx.putImageData(fg, 0, 0);

  ctx.drawImage(fgCanvas, 0, 0);
}

/* =======================
   6) Обробники подій
   ======================= */
// Upload основного зображення
file.addEventListener("change", e => {
  const f = e.target.files?.[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  img = new Image();
  img.onload = () => { URL.revokeObjectURL(url); lastMat = null; draw(); };
  img.src = url;
});

// Upload фону
uploadBgBtn.addEventListener("click", () => bgFile.click());
bgFile.addEventListener("change", e => {
  const f = e.target.files?.[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  bgImg = new Image();
  bgImg.onload = () => {
    URL.revokeObjectURL(url);
    if (lastMat) compositeWithAlpha(img, lastMat.alpha, lastMat.width, lastMat.height);
    else draw();
  };
  bgImg.src = url;
});

// Зміна кольору фону
picker.addEventListener("input", () => {
  if (lastMat) compositeWithAlpha(img, lastMat.alpha, lastMat.width, lastMat.height);
  else draw();
});

// Камера → знімок у img
cameraBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    const video = document.createElement("video");
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const dataURL = canvas.toDataURL("image/png");
    stream.getTracks().forEach(t => t.stop());

    img = new Image();
    img.onload = () => { lastMat = null; draw(); };
    img.src = dataURL;
  } catch {
    alert("Не вдалося відкрити камеру. Переконайся у HTTPS та дозволах.");
  }
});

// Автовидалення фону (AI)
autoRemoveBtn.addEventListener("click", async () => {
  if (!img.src) { alert("Спочатку завантаж зображення або зроби фото."); return; }
  try {
    lastMat = await runMatting(img);
    compositeWithAlpha(img, lastMat.alpha, lastMat.width, lastMat.height);
  } catch (err) {
    console.error("BGone matting error:", err);
    alert("Не вдалося обробити фон. Перевір консоль і наявність /bgone/models/u2netp.onnx");
  }
});

// Збереження PNG
savePng.addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = "bgone.png";
  a.click();
});

/* =======================
   7) Початковий стан
   ======================= */
window.addEventListener("load", () => {
  if (img && img.complete && img.naturalWidth) draw();
});
