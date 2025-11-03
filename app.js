const file = document.getElementById("file");
const cameraBtn = document.getElementById("camera");
const picker = document.getElementById("picker");
const savePng = document.getElementById("savePng");
const savePdf = document.getElementById("savePdf");
const uploadBgBtn = document.getElementById("uploadBgBtn");
const bgFile = document.getElementById("bgFile");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

let img = new Image();
let bgImg = null;

function draw() {
  if (!img.src) return;

  // 1) Зберігаємо оригінальну роздільну здатність
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  // 2) Малюємо фон (колір або завантажене зображення)
  if (bgImg) {
    ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = picker.value || "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // 3) ТУТ ВСТАВ СВІЙ АЛГОРИТМ ВИДАЛЕННЯ ФОНУ
  //    Наприклад, якщо у тебе вже є маска/alpha — застосуй її перед малюванням.
  //    Поки — просто малюємо оригінал поверх як заглушку:
  ctx.drawImage(img, 0, 0);
}

// --- Події ---
file.addEventListener("change", e => {
  const f = e.target.files?.[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  img = new Image();
  img.onload = () => { URL.revokeObjectURL(url); draw(); };
  img.src = url;
});

uploadBgBtn.addEventListener("click", () => bgFile.click());
bgFile.addEventListener("change", e => {
  const f = e.target.files?.[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  bgImg = new Image();
  bgImg.onload = () => { URL.revokeObjectURL(url); draw(); };
  bgImg.src = url;
});

picker.addEventListener("input", draw);

cameraBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    const video = document.createElement("video");
    video.srcObject = stream;
    await video.play();

    // Беремо кадр з відео
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (bgImg) ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
    else { ctx.fillStyle = picker.value || "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height); }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    stream.getTracks().forEach(t => t.stop());
  } catch (err) {
    alert("Камеру не вдалося відкрити. Перевір HTTPS і дозвіл у браузері.");
  }
});

savePng.addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = "bgone.png";
  a.click();
});

savePdf.addEventListener("click", async () => {
  // Тимчасово: системний друк → Зберегти як PDF.
  // За потреби підключимо pdf-lib/jsPDF і зробимо справжній PDF 1:1.
  window.print();
});
