// === 0) DOM-елементи з твого коду ===
const file = document.getElementById("file");
const cameraBtn = document.getElementById("camera");
const picker = document.getElementById("picker");
const savePng = document.getElementById("savePng");
const savePdf = document.getElementById("savePdf");
const uploadBgBtn = document.getElementById("uploadBgBtn");
const bgFile = document.getElementById("bgFile");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

// Нові елементи:
const autoRemoveBtn = document.getElementById("autoRemove");
const featherInput = document.getElementById("feather");

// Робочі канви
const fgCanvas = document.createElement("canvas");
const fgCtx = fgCanvas.getContext("2d", { willReadFrequently: true });

// Стан
let img = new Image();
let bgImg = null;
let session = null;  // ONNX сесія
let lastMat = null;  // Float32Array( h * w ) — мат (0..1) після моделі

// === 1) Завантаження моделі один раз ===
async function loadModel() {
  if (session) return;
  // u2netp очікує 320x320 RGB, нормалізований тензор
  session = await ort.InferenceSession.create("/bgone/models/u2netp.onnx", {
    executionProviders: ["wasm"]
  });
  console.log("U2Netp loaded");
}

// === 2) Утіліта: ресайз у 320x320 і підготовка тензора ===
function imageToTensor(imageBitmap, target = 320) {
  const tmp = document.createElement("canvas");
  tmp.width = target;
  tmp.height = target;
  const tctx = tmp.getContext("2d", { willReadFrequently: true });
  // Вписуємо картинку з збереженням пропорцій і заповненням чорним (ок)
  const scale = Math.min(target / imageBitmap.width, target / imageBitmap.height);
  const w = Math.round(imageBitmap.width * scale);
  const h = Math.round(imageBitmap.height * scale);
  const ox = Math.floor((target - w) / 2);
  const oy = Math.floor((target - h) / 2);
  tctx.fillStyle = "black";
  tctx.fillRect(0, 0, target, target);
  tctx.drawImage(imageBitmap, ox, oy, w, h);

  const { data } = tctx.getImageData(0, 0, target, target);
  // NCHW: 1x3x320x320, нормалізація до [0,1]
  const floatData = new Float32Array(1 * 3 * target * target);
  let idx = 0;
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < target; y++) {
      for (let x = 0; x < target; x++) {
        const i = (y * target + x) * 4;
        const v = data[i + c] / 255; // просте масштабування
        floatData[idx++] = v;
      }
    }
  }
  return { floatData, ox, oy, w, h, target };
}

// === 3) Інференс U²-Netp → маска (0..1) розміром оригіналу ===
async function runMatting(originalImage) {
  await loadModel();

  // a) Підготовка
  // Створимо ImageBitmap для швидкого ресайзу
  const ib = await createImageBitmap(originalImage);
  const { floatData, ox, oy, w, h, target } = imageToTensor(ib, 320);

  const input = new ort.Tensor("float32", floatData, [1, 3, target, target]);
  const feeds = { "input": input }; // ім’я інпуту у деяких збірках: "input" / "image"
  let results = await session.run(feeds);
  // пошукай ключ виходу — часто "output" або "u2netp_output"
  const out = results[Object.keys(results)[0]]; // беремо перший тензор
  const outData = out.data; // Float32Array(1*1*320*320) або подібне
  // b) Забираємо центр (там де реальна картинка), повертаємо у розмір оригіналу

  // 1) Масштабуємо мат в канву 320x320
  const tmp = document.createElement("canvas");
  tmp.width = target; tmp.height = target;
  const tctx = tmp.getContext("2d");
  // З outData будуємо ImageData (сіра карта)
  const imgData = tctx.createImageData(target, target);
  for (let i = 0; i < outData.length; i++) {
    // outData може бути логітами; застосуємо сигмоїду
    const v = 1 / (1 + Math.exp(-outData[i]));
    const k = i * 4;
    const g = Math.max(0, Math.min(255, Math.round(v * 255)));
    imgData.data[k] = g; imgData.data[k+1] = g; imgData.data[k+2] = g; imgData.data[k+3] = 255;
  }
  tctx.putImageData(imgData, 0, 0);

  // 2) Вирізаємо центральний прямокутник (де намальоване зображення)
  const sub = document.createElement("canvas");
  sub.width = w; sub.height = h;
  const sctx = sub.getContext("2d");
  sctx.drawImage(tmp, ox, oy, w, h, 0, 0, w, h);

  // 3) Масштабуємо під оригінальний розмір
  const full = document.createElement("canvas");
  full.width = originalImage.naturalWidth || originalImage.width;
  full.height = originalImage.naturalHeight || originalImage.height;
  const fctx = full.getContext("2d");
  fctx.imageSmoothingEnabled = true;
  fctx.imageSmoothingQuality = "high";
  fctx.drawImage(sub, 0, 0, full.width, full.height);

  // 4) Отримаємо Float32 масив 0..1
  const fullData = fctx.getImageData(0, 0, full.width, full.height).data;
  const alpha = new Float32Array(full.width * full.height);
  for (let i = 0, j = 0; i < fullData.length; i += 4, j++) {
    alpha[j] = fullData[i] / 255; // беремо червоний канал (усі рівні)
  }
  return { alpha, width: full.width, height: full.height };
}

// === 4) Композиція: фон (колір/картинка) + зображення з прозорим альфа-каналом ===
function compositeWithAlpha(originalImage, alphaArr, w, h) {
  // готуємо цільову канву під оригінал
  canvas.width = w; canvas.height = h;

  // фон
  if (bgImg) {
    ctx.drawImage(bgImg, 0, 0, w, h);
  } else {
    ctx.fillStyle = picker.value || "#ffffff";
    ctx.fillRect(0, 0, w, h);
  }

  // кладемо оригінал і задаємо альфу
  fgCanvas.width = w; fgCanvas.height = h;
  fgCtx.clearRect(0, 0, w, h);
  fgCtx.drawImage(originalImage, 0, 0, w, h);

  const fg = fgCtx.getImageData(0, 0, w, h);
  const d = fg.data;
  const feather = Number(featherInput.value); // 0..3
  // м’якість краю — легка гамма-корекція і трохи розмиття порога
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    let a = alphaArr[p];
    // підкрутимо м’якість: а^gamma, де gamma ~ 1/(1+feather/2)
    const gamma = 1 / (1 + feather / 2);
    a = Math.pow(a, gamma);
    d[i+3] = Math.max(0, Math.min(255, Math.round(a * 255)));
  }
  fgCtx.putImageData(fg, 0, 0);

  // зверху — наш прозорий форграунд
  ctx.drawImage(fgCanvas, 0, 0);
}

// === 5) Хендлери (інтегруємо в потік) ===

// Завантаження оригіналу
file.addEventListener("change", e => {
  const f = e.target.files?.[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  img = new Image();
  img.onload = () => { URL.revokeObjectURL(url); draw(); };
  img.src = url;
});

// Завантаження фону
uploadBgBtn.addEventListener("click", () => bgFile.click());
bgFile.addEventListener("change", e => {
  const f = e.target.files?.[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  bgImg = new Image();
  bgImg.onload = () => { URL.revokeObjectURL(url); if (lastMat) compositeWithAlpha(img, lastMat.alpha, lastMat.width, lastMat.height); else draw(); };
  bgImg.src = url;
});

// Зміна кольору фону
picker.addEventListener("input", () => {
  if (lastMat) compositeWithAlpha(img, lastMat.alpha, lastMat.width, lastMat.height);
  else draw();
});

// Камера
cameraBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    const video = document.createElement("video");
    video.srcObject = stream;
    await video.play();
    // знімок
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL("image/png");
    stream.getTracks().forEach(t => t.stop());
    img = new Image();
    img.onload = () => draw();
    img.src = dataURL;
  } catch (e) {
    alert("Не вдалося відкрити камеру. Перевір HTTPS/дозволи.");
  }
});

// Рендер без альфи (початковий перегляд)
function draw() {
  if (!img.src) return;
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  if (bgImg) ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
  else { ctx.fillStyle = picker.value || "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height); }
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

// Автовидалення фону (AI)
autoRemoveBtn.addEventListener("click", async () => {
  if (!img.src) { alert("Спочатку завантаж зображення або зроби фото."); return; }
  try {
    lastMat = await runMatting(img);
    compositeWithAlpha(img, lastMat.alpha, lastMat.width, lastMat.height);
  } catch (err) {
    console.error(err);
    alert("Не вдалося обробити фон. Спробуй інше фото або перезавантаж сторінку.");
  }
});

// Збереження PNG
savePng.addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = "bgone.png";
  a.click();
});

// Збереження PDF (через jsPDF)
savePdf.addEventListener("click", () => {
  const { jsPDF } = window.jspdf;
  const pdf = new jsPDF({ orientation: canvas.width >= canvas.height ? "l" : "p", unit: "px", format: [canvas.width, canvas.height] });
  const imgData = canvas.toDataURL("image/png");
  pdf.addImage(imgData, "PNG", 0, 0, canvas.width, canvas.height);
  pdf.save("bgone.pdf");
});
