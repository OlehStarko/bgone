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
  // Підганяємо канву під оригінал (зберігаємо вихідну роздільну здатність)
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;

  // Малюємо фон (колір або зображення)
  if (bgImg) {
    // впишемо BG по центру (простий варіант: розтягнути на весь розмір)
    ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = picker.value || "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // Тут мав би бути ваш алгоритм видалення фону.
  // Поки що — малюємо поверх вихідну картинку як заглушку.
  ctx.drawImage(img, 0, 0);
}

file.addEventListener("change", async e => {
  const f = e.target.files?.[0];
  if (!f) return;
  img = new Image();
  img.onload = draw;
  img.src = URL.createObjectURL(f);
});

uploadBgBtn.addEventListener("click", () => bgFile.click());
bgFile.addEventListener("change", e => {
  const f = e.target.files?.[0];
  if (!f) return;
  bgImg = new Image();
  bgImg.onload = draw;
  bgImg.src = URL.createObjectURL(f);
});

picker.addEventListener("input", draw);

cameraBtn.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    const video = document.createElement("video");
    video.srcObject = stream;
    await video.play();

    // Кадр із відео
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Заповнюємо фоном
    if (bgImg) ctx.drawImage(bgImg, 0, 0, canvas.width, canvas.height);
    else {
      ctx.fillStyle = picker.value || "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    // Знімаємо кадр
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // зупиняємо камеру
    stream.getTracks().forEach(t => t.stop());
  } catch (e) {
    alert("Не вдалося відкрити камеру. Переконайся, що сайт відкрито по HTTPS і є дозвіл.");
  }
});

savePng.addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = "bgone.png";
  a.click();
});

savePdf.addEventListener("click", async () => {
  // Легка PDF-заглушка: генеруємо PNG і підказуємо конвертацію у PDF через системний Share/Print
  // Або пізніше додамо PDFLib/Canvas-to-PDF.
  window.print(); // тимчасовий варіант; або додамо бібліотеку пізніше
});
