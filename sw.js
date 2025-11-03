const CACHE = "bgone-v1";
const ASSETS = [
  "/bgone/",
  "/bgone/index.html",
  "/bgone/styles.css",
  "/bgone/app.js",
  "/bgone/manifest.webmanifest",
  "/bgone/icons/icon-192.png",
  "/bgone/icons/icon-512.png"
];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.map(k => (k === CACHE ? null : caches.delete(k))))
    )
  );
});

// ВАЖЛИВО: fetch-обробник — щоб Android робив WebAPK (не просто ярлик)
self.addEventListener("fetch", e => {
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
