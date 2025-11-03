/* =======================
   BGone – app.js (FULL)
   ======================= */

/* ---- UI елементи ---- */
const file = document.getElementById("file");
const cameraBtn = document.getElementById("camera");
const picker = document.getElementById("picker");
const uploadBgBtn = document.getElementById("uploadBgBtn");
const bgFile = document.getElementById("bgFile");
const savePng = document.getElementById("savePng");
const autoRemoveBtn = document.getElementById("autoRemove"); // кнопка "Автовидалення фону (AI)"
const featherInput = document.getElementById("feather");      // повзунок м’якості краю (0..3)
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

/* ---- Стани ---- */
let img = new Image();          // вихідне зображення
let bgImg = null;               // фон-картинка (якщо завантажено)
let session = null;             // ONNX Runtime сесія
let lastMat = null;             // { alpha: Float32Array, width, height } – остання маска

/* ---- Робоча канва для прозорого форграунду ---- */
const fgCanvas = document.createElement("canvas");
const fgCtx = fgCanvas.getContext("2d", { willReadFrequently: true });

/* ---- Шлях до моделі ---- */
const MODEL_PATH = "/bgone/models/u2netp.onnx";

/* =======================
   1) Базове відмальовування
   ======================= */
function draw() {
  if (!img.src) return;
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;

  // Фон – або зображення, або колір
  if (bgImg) {
    ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = picker.value || "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // Поки без альфи: просто поверх
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

  // Вписуємо зображення в квадрат 320×320 з letterbox (чорні поля)
  const scale = Math.min(target / imageBitmap.width, target / imageBitmap.height);
  const w = Math.round(imageBitmap.width * scale);
  const h = Math.round(imageBitmap.height * scale);
  const ox = Math.floor((target - w) / 2);
  const oy = Math.floor((target - h) / 2);

  tctx.fillStyle = "black";
  tctx.fillRect(0, 0, target, target);
  tctx.drawImage(imageBitmap, ox, oy, w, h);

  const { data } = tctx.getImageData(0, 0, target, target);

  // NCHW: 1×3×H×W, значення у [0..1]
  const floatData = new Float32Array(1 * 3 * target * target);
  let idx = 0;
  // Канали в порядку R,G,B
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < target; y++) {
      for (let x = 0; x < target; x++) {
        const i = (y * target + x) * 4;
        const v = data[i + c] / 255;
        floatData[idx++] = v;
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
  console.log("U2Netp loaded", "inputs:", session.inputNames, "outputs:", session.outputNames);
}

/* =======================
   4) Інференс: отримуємо маску (alpha 0..1) під оригінальний розмір
   ======================= */
async function runMatting(originalImage) {
  await loadModel();

  const ib = await createImageBitmap(originalImage);
  const { floatData, ox, oy, w, h, target } = imageToTensor(ib, 320);

  const inputName = session.inputNames[0];    // динамічні назви важливі!
  const outputName = session.outputNames[0];

  const inputTensor = new ort.Tensor("float32", floatData, [1, 3, target, target]);
  const results = await session.run({ [inputName]: inputTensor });
  const out = results[outputName];

  const outData = out.data;       // Float32Array
  const dims = out.dims || out.dims_;
  // Очікувано: [1,1,320,320] або [1,320,320,1]
  let H = 320, W = 320;
  if (Array.isArray(dims) && dims.length === 4) {
    if (dims[2] && dims[3]) { H = dims[2]; W = dims[3]; }
    else if (dims[1] && dims[2]) { H = dims[1]; W = dims[2]; }
  }

  // Логіти → ймовірності (сигмоїда) і малюємо у канву H×W
  const tmp = document.createElement("canvas");
  tmp.width = W; tmp.height = H;
  const tctx = tmp.getContext("2d");
  const imgData = tctx.createImageData(W, H);

  for (let i = 0; i < outData.length; i++) {
    const p = 1 / (1 + Math.exp(-outData[i])); // 0..1
    const g = Math.max(0, Math.min(255, Math.round(p * 255)));
    const k = i * 4;
    imgData.data[k] = g;
    imgData.data[k + 1] = g;
    imgData.data[k + 2] = g;
    imgData.data[k + 3] = 255;
  }
  tctx.putImageData(imgData, 0, 0);

  // Вирізаємо центральну частину (де був оригінал) і масштабуємо під оригінал
  const sub = document.createElement("canvas");
  sub.width = w; sub.height = h;
  sub.getContext("2d").drawImage(tmp, ox, oy, w, h, 0, 0, w, h);

  const full = document.createElement("canvas");
  const OW = originalImage.naturalWidth || originalImage.width;
  const OH = originalImage.naturalHeight || originalImage.height;
  full.width = OW; full.height = OH;
  const fctx = full.getContext("2d");
  fctx.imageSmoothingEnabled = true;
  fctx.imageSmoothingQuality = "high";
  fctx.drawImage(sub, 0, 0, OW, OH);

  // Витягуємо альфу 0..1
  const fullData = fctx.getImageData(0, 0, OW, OH).data;
  const alpha = new Float32Array(OW * OH);
  for (let i = 0, j = 0; i < fullData.length; i += 4, j++) {
    alpha[j] = fullData[i] / 255; // будь-який канал (однакові)
  }
  return { alpha, width: OW, height: OH };
}

/* =======================
   5) Композиція: фон + прозорий форграунд
   ======================= */
function compositeWithAlpha(originalImage, alphaArr, w, h) {
  canvas.width = w;
  canvas.height = h;

  // Підкладка
  if (bgImg) {
    ctx.drawImage(bgImg, 0, 0, w, h);
  } else {
    ctx.fillStyle = picker.value || "#ffffff";
    ctx.fillRect(0, 0, w, h);
  }

  // Готуємо форграунд з альфою
  fgCanvas.width = w;
  fgCanvas.height = h;
  fgCtx.clearRect(0, 0, w, h);
  fgCtx.drawImage(originalImage, 0, 0, w, h);

  const fg = fgCtx.getImageData(0, 0, w, h);
  const d = fg.data;
  const feather = Number(featherInput ? featherInput.value : 1.5); // 0..3

  // Невелика "м’якість" по альфі через гамма-корекцію
  const gamma = 1 / (1 + feather / 2);
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    let a = alphaArr[p];
    a = Math.pow(a, gamma);
    d[i + 3] = Math.max(0, Math.min(255, Math.round(a * 255)));
  }
  fgCtx.putImageData(fg, 0, 0);

  // Зверху кладемо прозорий форграунд
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

// Колір фону
if (picker) {
  picker.addEventListener("input", () => {
    if (lastMat) compositeWithAlpha(img, lastMat.alpha, lastMat.width, lastMat.height);
    else draw();
  });
}

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
  } catch (e) {
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
    alert("Не вдалося обробити фон. Перевір консоль (F12 → Console) і присутність /bgone/models/u2netp.onnx");
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
  // Пробуємо намалювати, якщо img уже був призначений деінде
  if (img && img.complete && img.naturalWidth) draw();
});
