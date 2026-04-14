import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const ui = {
  centerLat: document.getElementById("centerLat"),
  centerLon: document.getElementById("centerLon"),
  radiusM: document.getElementById("radiusM"),
  outputShape: document.getElementById("outputShape"),
  widthMm: document.getElementById("widthMm"),
  heightMm: document.getElementById("heightMm"),
  meshSpacing: document.getElementById("meshSpacing"),
  verticalExaggeration: document.getElementById("verticalExaggeration"),
  baseHeight: document.getElementById("baseHeight"),
  adaptiveTriangulation: document.getElementById("adaptiveTriangulation"),
  adaptiveRelief: document.getElementById("adaptiveRelief"),
  adaptiveIterations: document.getElementById("adaptiveIterations"),
  adaptiveMaxPoints: document.getElementById("adaptiveMaxPoints"),
  adaptiveMinAngle: document.getElementById("adaptiveMinAngle"),
  anisotropicOn: document.getElementById("anisotropicOn"),
  anisotropicStrength: document.getElementById("anisotropicStrength"),
  earthEngineProject: document.getElementById("earthEngineProject"),
  demDatasetId: document.getElementById("demDatasetId"),
  viewerBedSize: document.getElementById("viewerBedSize"),
  generateBtn: document.getElementById("generateBtn"),
  clearCacheBtn: document.getElementById("clearCacheBtn"),
  cacheStats: document.getElementById("cacheStats"),
  spinner: document.getElementById("spinner"),
  statusText: document.getElementById("statusText"),
  statusMeta: document.getElementById("statusMeta"),
  downloadLink: document.getElementById("downloadLink"),
  meshInfo: document.getElementById("meshInfo"),
  meshColor: document.getElementById("meshColor"),
  meshColorPreview: document.getElementById("meshColorPreview"),
  sunAzimuth: document.getElementById("sunAzimuth"),
  sunAzimuthValue: document.getElementById("sunAzimuthValue"),
  sunElevation: document.getElementById("sunElevation"),
  sunElevationValue: document.getElementById("sunElevationValue"),
  historyTableBody: document.getElementById("historyTableBody"),
  historyEmptyRow: document.getElementById("historyEmptyRow"),
};

const PARAM_BY_INPUT_ID = {
  centerLat: "center_lat",
  centerLon: "center_lon",
  radiusM: "radius_m",
  outputShape: "output_shape",
  widthMm: "output_width_mm",
  heightMm: "output_height_mm",
  verticalExaggeration: "vertical_exaggeration",
  meshSpacing: "mesh_spacing_mm",
  baseHeight: "base_height_mm",
  adaptiveTriangulation: "adaptive_triangulation",
  adaptiveRelief: "adaptive_relief_threshold_mm",
  adaptiveIterations: "adaptive_iterations",
  adaptiveMaxPoints: "adaptive_max_new_points",
  adaptiveMinAngle: "adaptive_min_angle_deg",
  anisotropicOn: "adaptive_anisotropic_refinement",
  anisotropicStrength: "adaptive_anisotropic_strength",
  earthEngineProject: "earth_engine_project",
  demDatasetId: "dem_dataset_id",
};

const map = L.map("map").setView([Number(ui.centerLat.value), Number(ui.centerLon.value)], 10);
map.doubleClickZoom.disable();

const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
});

const googleSat = L.tileLayer("https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", {
  maxZoom: 20,
  subdomains: ["mt0", "mt1", "mt2", "mt3"],
  attribution: "Imagery: Google",
});

osm.addTo(map);
L.control.layers({ "OpenStreetMap": osm, "Google Aerial": googleSat }).addTo(map);

let activeBasemap = "osm";
map.on("baselayerchange", (event) => {
  activeBasemap = event.layer === googleSat ? "aerial" : "osm";
  updateOverlay();
});

const selectionPane = map.createPane("selectionPane");
selectionPane.style.zIndex = "450";
selectionPane.style.pointerEvents = "none";

const marker = L.marker([Number(ui.centerLat.value), Number(ui.centerLon.value)], { draggable: true }).addTo(map);
marker.setZIndexOffset(1000);
marker.dragging?.enable();
let overlay = null;
const historyJobsById = new Map();

function valNumber(el) {
  return Number(el.value);
}

function metersToLat(m) {
  return m / 111320;
}

function metersToLon(m, lat) {
  return m / (111320 * Math.max(0.01, Math.cos((lat * Math.PI) / 180)));
}

function squareCoords(lat, lon, radiusM) {
  const latD = metersToLat(radiusM);
  const lonD = metersToLon(radiusM, lat);
  return [
    [lat + latD, lon + lonD],
    [lat + latD, lon - lonD],
    [lat - latD, lon - lonD],
    [lat - latD, lon + lonD],
  ];
}

function hexCoords(lat, lon, radiusM) {
  const points = [];
  for (let i = 0; i < 6; i += 1) {
    const angle = (Math.PI / 3) * i;
    const dyM = Math.sin(angle) * radiusM;
    const dxM = Math.cos(angle) * radiusM;
    points.push([lat + metersToLat(dyM), lon + metersToLon(dxM, lat)]);
  }
  return points;
}

function updateOverlay() {
  const lat = valNumber(ui.centerLat);
  const lon = valNumber(ui.centerLon);
  const radiusM = Math.max(1, valNumber(ui.radiusM));
  const shape = ui.outputShape.value;

  if (overlay) {
    map.removeLayer(overlay);
  }

  const outlineColor = activeBasemap === "aerial" ? "#6b7280" : "#000000";

  if (shape === "circular") {
    overlay = L.circle([lat, lon], {
      pane: "selectionPane",
      radius: radiusM,
      color: outlineColor,
      weight: 2,
      fillColor: "#38bdf8",
      fillOpacity: 0.28,
    }).addTo(map);
    overlay.bringToFront();
    return;
  }

  const points = shape === "square" ? squareCoords(lat, lon, radiusM) : hexCoords(lat, lon, radiusM);
  overlay = L.polygon(points, {
    pane: "selectionPane",
    color: outlineColor,
    weight: 2,
    fillColor: "#38bdf8",
    fillOpacity: 0.28,
  }).addTo(map);
  overlay.bringToFront();
}

marker.on("dragend", () => {
  const { lat, lng } = marker.getLatLng();
  ui.centerLat.value = lat.toFixed(6);
  ui.centerLon.value = lng.toFixed(6);
  updateOverlay();
  if (!map.dragging.enabled()) {
    map.dragging.enable();
  }
});

marker.on("dragstart", () => {
  if (map.dragging.enabled()) {
    map.dragging.disable();
  }
});

marker.on("mousedown", (event) => {
  if (map.dragging.enabled()) {
    map.dragging.disable();
  }
  L.DomEvent.stopPropagation(event);
});

marker.on("mouseup", () => {
  if (!map.dragging.enabled()) {
    map.dragging.enable();
  }
});

marker.on("touchstart", (event) => {
  if (map.dragging.enabled()) {
    map.dragging.disable();
  }
  L.DomEvent.stopPropagation(event);
});

marker.on("touchend", () => {
  if (!map.dragging.enabled()) {
    map.dragging.enable();
  }
});

map.on("dblclick", (event) => {
  const { lat, lng } = event.latlng;
  marker.setLatLng([lat, lng]);
  ui.centerLat.value = lat.toFixed(6);
  ui.centerLon.value = lng.toFixed(6);
  updateOverlay();
});

[ui.centerLat, ui.centerLon].forEach((el) => {
  el.addEventListener("input", () => {
    marker.setLatLng([valNumber(ui.centerLat), valNumber(ui.centerLon)]);
    updateOverlay();
  });
});

[ui.radiusM, ui.outputShape].forEach((el) => {
  el.addEventListener("input", updateOverlay);
  el.addEventListener("change", updateOverlay);
});

const viewerEl = document.getElementById("viewer");
const scene = new THREE.Scene();
scene.background = new THREE.Color("#f6f8fb");
scene.fog = new THREE.Fog(0xf1f5f9, 180, 520);

const camera = new THREE.PerspectiveCamera(50, viewerEl.clientWidth / viewerEl.clientHeight, 0.1, 5000);
camera.position.set(140, 120, 140);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
viewerEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);

const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(220, 65, 120);
scene.add(sun);

const fill = new THREE.DirectionalLight(0xe2e8f0, 0.45);
fill.position.set(-120, 140, -110);
scene.add(fill);

ambient.intensity = 0.38;

const basePlane = new THREE.Mesh(
  new THREE.PlaneGeometry(160, 160),
  new THREE.MeshStandardMaterial({
    color: 0xd1d5db,
    roughness: 0.95,
    metalness: 0.0,
  }),
);
basePlane.rotation.x = -Math.PI / 2;
basePlane.position.y = 0;
scene.add(basePlane);

let grid = null;

function updatePrintBedSize() {
  const size = Math.max(40, Number(ui.viewerBedSize?.value || 160));
  basePlane.geometry.dispose();
  basePlane.geometry = new THREE.PlaneGeometry(size, size);

  if (grid) {
    scene.remove(grid);
    grid.geometry.dispose();
    if (Array.isArray(grid.material)) {
      grid.material.forEach((mat) => mat.dispose());
    } else {
      grid.material.dispose();
    }
  }

  const divisions = Math.max(8, Math.round(size / 8));
  grid = new THREE.GridHelper(size, divisions, 0x94a3b8, 0xb6c2d1);
  grid.position.y = 0.02;
  scene.add(grid);
}

let mesh = null;

function updateSunDirection() {
  const azimuthDeg = Number(ui.sunAzimuth?.value ?? 28);
  const elevationDeg = Number(ui.sunElevation?.value ?? 14);
  if (ui.sunAzimuthValue) {
    ui.sunAzimuthValue.textContent = `${Math.round(azimuthDeg)} deg`;
  }
  if (ui.sunElevationValue) {
    ui.sunElevationValue.textContent = `${Math.round(elevationDeg)} deg`;
  }
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  const radius = 260;
  const x = Math.cos(el) * Math.cos(az) * radius;
  const y = Math.sin(el) * radius;
  const z = Math.cos(el) * Math.sin(az) * radius;
  sun.position.set(x, y, z);
}

function updateMeshColor() {
  if (ui.meshColorPreview && ui.meshColor?.value) {
    ui.meshColorPreview.style.background = ui.meshColor.value;
  }
  if (!mesh || !mesh.material || !(mesh.material instanceof THREE.MeshStandardMaterial)) {
    return;
  }
  mesh.material.color.set(ui.meshColor?.value ?? "#9ca3af");
}

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) {
    return "-";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)} s`;
  }
  const mins = Math.floor(seconds / 60);
  const rem = seconds % 60;
  return `${mins}m ${rem.toFixed(1)}s`;
}

function formatFileSize(bytes) {
  if (bytes == null || Number.isNaN(bytes)) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIdx = 0;
  while (size >= 1024 && unitIdx < units.length - 1) {
    size /= 1024;
    unitIdx += 1;
  }
  return `${size.toFixed(size >= 100 || unitIdx === 0 ? 0 : 1)} ${units[unitIdx]}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function loadParameterHelp() {
  const res = await fetch("/api/parameter-help");
  if (!res.ok) {
    return;
  }
  const help = await res.json();
  for (const [inputId, key] of Object.entries(PARAM_BY_INPUT_ID)) {
    const input = document.getElementById(inputId);
    if (!input) {
      continue;
    }
    const label = input.closest("label");
    if (!label) {
      continue;
    }
    const message = help[key];
    if (!message) {
      continue;
    }
    if (label.querySelector(".help-dot")) {
      continue;
    }
    const dot = document.createElement("span");
    dot.className = "help-dot";
    dot.textContent = "i";
    dot.tabIndex = 0;
    dot.dataset.help = message;
    dot.title = message;
    dot.setAttribute("aria-label", message);
    label.insertBefore(dot, input);
  }
}

function statusBadge(status) {
  if (status === "done") {
    return '<span class="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700">done</span>';
  }
  if (status === "error") {
    return '<span class="rounded-full border border-rose-200 bg-rose-50 px-2 py-0.5 text-xs font-semibold text-rose-700">error</span>';
  }
  if (status === "running") {
    return '<span class="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700">running</span>';
  }
  return '<span class="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-semibold text-slate-600">queued</span>';
}

function cacheBadge(cacheHit) {
  if (cacheHit == null) {
    return '<span class="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-xs font-semibold text-slate-500">cache -</span>';
  }
  if (cacheHit) {
    return '<span class="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-xs font-semibold text-emerald-700">cache hit</span>';
  }
  return '<span class="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-700">cache miss</span>';
}

function trianglesBadge(triangles) {
  const text = triangles != null ? `${triangles.toLocaleString()} tris` : "- tris";
  return `<span class="rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-xs font-semibold text-sky-700">${text}</span>`;
}

function durationBadge(durationSeconds) {
  return `<span class="rounded-full border border-violet-200 bg-violet-50 px-2 py-0.5 text-xs font-semibold text-violet-700">${formatDuration(durationSeconds)}</span>`;
}

function clearStatusMeta() {
  if (!ui.statusMeta) {
    return;
  }
  ui.statusMeta.innerHTML = "";
}

function setStatusMeta(job) {
  if (!ui.statusMeta) {
    return;
  }
  ui.statusMeta.innerHTML = `${trianglesBadge(job.triangles)} ${cacheBadge(job.cache_hit)} ${durationBadge(job.duration_seconds)}`;
}

function rowActions(job) {
  if (job.status !== "done" || !job.stl_url) {
    return '<span class="text-xs text-slate-400">-</span>';
  }
  return `
    <div class="flex gap-2">
      <button type="button" data-action="display" data-job-id="${escapeHtml(job.job_id)}" data-url="${escapeHtml(job.stl_url)}" class="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-50">Display</button>
      <a href="${escapeHtml(job.stl_url)}" download class="rounded-md bg-emerald-600 px-2 py-1 text-xs font-semibold text-white hover:bg-emerald-500">Download</a>
      <button type="button" data-action="delete" data-job-id="${escapeHtml(job.job_id)}" class="rounded-md border border-rose-300 bg-white px-2 py-1 text-xs font-semibold text-rose-700 hover:bg-rose-50">Delete Model</button>
    </div>
  `;
}

async function hydrateStlSizes(jobs) {
  const pending = jobs.filter((job) => job.status === "done" && job.stl_url && (job.stl_size_bytes == null));
  if (pending.length === 0) {
    return;
  }

  await Promise.all(
    pending.map(async (job) => {
      try {
        const res = await fetch(job.stl_url, { method: "HEAD" });
        if (!res.ok) {
          return;
        }
        const len = res.headers.get("content-length");
        if (!len) {
          return;
        }
        const parsed = Number(len);
        if (!Number.isNaN(parsed) && parsed >= 0) {
          job.stl_size_bytes = parsed;
        }
      } catch {
        // Keep rendering with fallback '-' if HEAD fails.
      }
    }),
  );
}

function renderHistory(jobs) {
  if (!Array.isArray(jobs) || jobs.length === 0) {
    ui.historyTableBody.innerHTML = '<tr id="historyEmptyRow"><td colspan="10" class="px-3 py-4 text-slate-500">No generated models yet.</td></tr>';
    return;
  }

  historyJobsById.clear();
  for (const job of jobs) {
    historyJobsById.set(job.job_id, job);
  }

  const rows = jobs.map((job) => {
    const created = new Date(job.created_at).toLocaleString();
    const triangles = job.triangles != null ? job.triangles.toLocaleString() : "-";
    const request = job.request || {};
    const centerLat = request.center_lat != null ? Number(request.center_lat).toFixed(4) : "-";
    const centerLon = request.center_lon != null ? Number(request.center_lon).toFixed(4) : "-";
    const verticalExaggeration = request.vertical_exaggeration != null ? Number(request.vertical_exaggeration).toFixed(2) : "-";
    const spacing = request.mesh_spacing_mm != null ? Number(request.mesh_spacing_mm).toFixed(2) : "-";
    const stlSize = formatFileSize(job.stl_size_bytes);
    const duration = formatDuration(job.duration_seconds);
    return `
      <tr>
        <td class="px-3 py-2 text-slate-700">${escapeHtml(created)}</td>
        <td class="px-3 py-2 text-slate-700">${centerLat}, ${centerLon}</td>
        <td class="px-3 py-2">${statusBadge(job.status)}</td>
        <td class="px-3 py-2 text-slate-700">${triangles}</td>
        <td class="px-3 py-2">${cacheBadge(job.cache_hit)}</td>
        <td class="px-3 py-2 text-slate-700">${verticalExaggeration}</td>
        <td class="px-3 py-2 text-slate-700">${spacing}</td>
        <td class="px-3 py-2 text-slate-700">${stlSize}</td>
        <td class="px-3 py-2 text-slate-700">${duration}</td>
        <td class="px-3 py-2">${rowActions(job)}</td>
      </tr>
    `;
  });
  ui.historyTableBody.innerHTML = rows.join("");
}

async function refreshHistory() {
  const res = await fetch("/api/jobs");
  if (!res.ok) {
    return;
  }
  const jobs = await res.json();
  await hydrateStlSizes(jobs);
  renderHistory(jobs);
}

async function refreshCacheStats() {
  if (!ui.cacheStats) {
    return;
  }
  try {
    const res = await fetch("/api/cache/stats");
    if (!res.ok) {
      ui.cacheStats.textContent = "Cache: unavailable";
      return;
    }
    const data = await res.json();
    const size = formatFileSize(data.size_bytes ?? null);
    const files = Number(data.file_count ?? 0).toLocaleString();
    ui.cacheStats.textContent = `Cache: ${size} (${files} files)`;
  } catch {
    ui.cacheStats.textContent = "Cache: unavailable";
  }
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  const w = viewerEl.clientWidth;
  const h = viewerEl.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  map.invalidateSize();
});

function setWorking(isWorking, text) {
  ui.spinner.classList.toggle("hidden", !isWorking);
  ui.generateBtn.disabled = isWorking;
  ui.generateBtn.classList.toggle("opacity-50", isWorking);
  setStatus(isWorking ? "working" : "idle", text);
  if (isWorking) {
    clearStatusMeta();
  }
}

function setStatus(kind, text) {
  const base = "rounded-full px-3 py-1 text-sm font-semibold border transition-colors";
  const byKind = {
    idle: "bg-slate-100 text-slate-700 border-slate-200",
    working: "bg-amber-50 text-amber-700 border-amber-200",
    done: "bg-emerald-50 text-emerald-700 border-emerald-200",
    error: "bg-rose-50 text-rose-700 border-rose-200",
    info: "bg-sky-50 text-sky-700 border-sky-200",
  };
  ui.statusText.className = `${base} ${byKind[kind] || byKind.info}`;
  ui.statusText.textContent = text;
}

function formPayload() {
  return {
    center_lat: valNumber(ui.centerLat),
    center_lon: valNumber(ui.centerLon),
    radius_m: valNumber(ui.radiusM),
    output_shape: ui.outputShape.value,
    output_width_mm: valNumber(ui.widthMm),
    output_height_mm: valNumber(ui.heightMm),
    vertical_exaggeration: valNumber(ui.verticalExaggeration),
    mesh_spacing_mm: valNumber(ui.meshSpacing),
    base_height_mm: valNumber(ui.baseHeight),
    adaptive_triangulation: ui.adaptiveTriangulation.value === "true",
    adaptive_relief_threshold_mm: valNumber(ui.adaptiveRelief),
    adaptive_max_new_points: Math.round(valNumber(ui.adaptiveMaxPoints)),
    adaptive_iterations: Math.round(valNumber(ui.adaptiveIterations)),
    adaptive_min_angle_deg: valNumber(ui.adaptiveMinAngle),
    adaptive_anisotropic_refinement: ui.anisotropicOn.value === "true",
    adaptive_anisotropic_strength: valNumber(ui.anisotropicStrength),
    earth_engine_project: ui.earthEngineProject.value,
    dem_dataset_id: ui.demDatasetId.value,
  };
}

function applyRequestToInputs(request) {
  if (!request) {
    return;
  }
  if (request.center_lat != null) ui.centerLat.value = Number(request.center_lat).toFixed(6);
  if (request.center_lon != null) ui.centerLon.value = Number(request.center_lon).toFixed(6);
  if (request.radius_m != null) ui.radiusM.value = String(request.radius_m);
  if (request.output_shape != null) ui.outputShape.value = request.output_shape;
  if (request.output_width_mm != null) ui.widthMm.value = String(request.output_width_mm);
  if (request.output_height_mm != null) ui.heightMm.value = String(request.output_height_mm);
  if (request.vertical_exaggeration != null) ui.verticalExaggeration.value = String(request.vertical_exaggeration);
  if (request.mesh_spacing_mm != null) ui.meshSpacing.value = String(request.mesh_spacing_mm);
  if (request.base_height_mm != null) ui.baseHeight.value = String(request.base_height_mm);
  if (request.adaptive_triangulation != null) ui.adaptiveTriangulation.value = request.adaptive_triangulation ? "true" : "false";
  if (request.adaptive_relief_threshold_mm != null) ui.adaptiveRelief.value = String(request.adaptive_relief_threshold_mm);
  if (request.adaptive_iterations != null) ui.adaptiveIterations.value = String(request.adaptive_iterations);
  if (request.adaptive_max_new_points != null) ui.adaptiveMaxPoints.value = String(request.adaptive_max_new_points);
  if (request.adaptive_min_angle_deg != null) ui.adaptiveMinAngle.value = String(request.adaptive_min_angle_deg);
  if (request.adaptive_anisotropic_refinement != null) ui.anisotropicOn.value = request.adaptive_anisotropic_refinement ? "true" : "false";
  if (request.adaptive_anisotropic_strength != null) ui.anisotropicStrength.value = String(request.adaptive_anisotropic_strength);
  if (request.earth_engine_project != null) ui.earthEngineProject.value = String(request.earth_engine_project);
  if (request.dem_dataset_id != null) ui.demDatasetId.value = String(request.dem_dataset_id);

  marker.setLatLng([valNumber(ui.centerLat), valNumber(ui.centerLon)]);
  updateOverlay();
  if (overlay && typeof overlay.getBounds === "function") {
    map.fitBounds(overlay.getBounds(), { padding: [18, 18] });
  } else {
    map.setView([valNumber(ui.centerLat), valNumber(ui.centerLon)], 10);
  }
}

function loadStl(url) {
  const loader = new STLLoader();
  loader.load(
    url,
    (geometry) => {
      geometry.computeVertexNormals();
      geometry.center();
      geometry.computeBoundingBox();
      const bbox = geometry.boundingBox;
      const maxDim = Math.max(
        bbox.max.x - bbox.min.x,
        bbox.max.y - bbox.min.y,
        bbox.max.z - bbox.min.z,
      );
      const scale = maxDim > 0 ? 120 / maxDim : 1;
      geometry.scale(scale, scale, scale);

      if (mesh) {
        scene.remove(mesh);
        mesh.geometry.dispose();
      }

      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(ui.meshColor?.value ?? "#9ca3af"),
        roughness: 0.92,
        metalness: 0.03,
        side: THREE.DoubleSide,
      });

      mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = -Math.PI / 2;

      const liftedBox = new THREE.Box3().setFromObject(mesh);
      const lift = Math.max(0, -liftedBox.min.y) + 0.35;
      mesh.position.y += lift;

      controls.target.set(0, lift * 0.5, 0);
      scene.add(mesh);
      ui.meshInfo.textContent = "Loaded STL. Use mouse/touch to orbit and zoom.";
    },
    undefined,
    (error) => {
      ui.meshInfo.textContent = `Failed to load STL: ${error.message || error}`;
    },
  );
}

async function pollJob(jobId) {
  while (true) {
    const statusRes = await fetch(`/api/jobs/${jobId}`);
    if (!statusRes.ok) {
      setWorking(false, "Failed to query job status.");
      return;
    }
    const status = await statusRes.json();

    if (status.status === "queued" || status.status === "running") {
      setWorking(true, `Working... (${status.status})`);
      await new Promise((resolve) => setTimeout(resolve, 1200));
      continue;
    }

    if (status.status === "error") {
      setStatus("error", `Generation failed: ${status.error || "Unknown error"}`);
      clearStatusMeta();
      ui.spinner.classList.add("hidden");
      ui.generateBtn.disabled = false;
      ui.generateBtn.classList.remove("opacity-50");
      return;
    }

    if (status.status === "done") {
      const duration = formatDuration(status.duration_seconds);
      setStatus(
        "done",
        `Ready - ${status.triangles?.toLocaleString() || "?"} triangles - cache ${status.cache_hit ? "hit" : "miss"} - ${duration}`,
      );
      setStatusMeta(status);
      ui.spinner.classList.add("hidden");
      ui.generateBtn.disabled = false;
      ui.generateBtn.classList.remove("opacity-50");
      ui.downloadLink.href = status.stl_url;
      ui.downloadLink.classList.remove("hidden");
      loadStl(`${status.stl_url}?t=${Date.now()}`);
      await refreshHistory();
      return;
    }

    setStatus("info", `Unknown status: ${status.status}`);
    clearStatusMeta();
    ui.spinner.classList.add("hidden");
    ui.generateBtn.disabled = false;
    ui.generateBtn.classList.remove("opacity-50");
    return;
  }
}

ui.generateBtn.addEventListener("click", async () => {
  setWorking(true, "Submitting job...");
  ui.downloadLink.classList.add("hidden");

  try {
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });

    if (!res.ok) {
      const body = await res.text();
      setStatus("error", `Submit failed: ${body}`);
      clearStatusMeta();
      ui.spinner.classList.add("hidden");
      ui.generateBtn.disabled = false;
      ui.generateBtn.classList.remove("opacity-50");
      return;
    }

    const payload = await res.json();
    await pollJob(payload.job_id);
  } catch (error) {
    setStatus("error", `Submit failed: ${error}`);
    clearStatusMeta();
    ui.spinner.classList.add("hidden");
    ui.generateBtn.disabled = false;
    ui.generateBtn.classList.remove("opacity-50");
  }
});

ui.historyTableBody.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const action = target.dataset.action;
  if (!action) {
    return;
  }

  const jobId = target.dataset.jobId;
  const job = jobId ? historyJobsById.get(jobId) : null;
  if (action === "display") {
    const url = target.dataset.url;
    if (!url) {
      return;
    }

    if (job?.request) {
      applyRequestToInputs(job.request);
      setStatusMeta(job);
    }

    loadStl(`${url}?t=${Date.now()}`);
    setStatus("info", "Displayed model and restored generation parameters.");
    return;
  }

  if (action === "delete") {
    if (!jobId) {
      return;
    }
    const ok = window.confirm("Delete STL file for this job?");
    if (!ok) {
      return;
    }

    try {
      const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
      if (!res.ok) {
        const body = await res.text();
        setStatus("error", `Delete failed: ${body}`);
        return;
      }
      const data = await res.json();
      setStatus("info", data.deleted ? "Model entry removed." : `No deletion needed: ${data.reason || "already missing"}`);
      clearStatusMeta();
      ui.downloadLink.classList.add("hidden");
      await refreshHistory();
    } catch (error) {
      setStatus("error", `Delete failed: ${error}`);
    }
  }
});

ui.clearCacheBtn?.addEventListener("click", async () => {
  const ok = window.confirm("Clear cached DEM files?");
  if (!ok) {
    return;
  }
  try {
    const res = await fetch("/api/cache", { method: "DELETE" });
    if (!res.ok) {
      const body = await res.text();
      setStatus("error", `Cache clear failed: ${body}`);
      return;
    }
    const data = await res.json();
    const skipped = Number(data.skipped_files ?? 0);
    const skippedText = skipped > 0 ? `, skipped ${skipped.toLocaleString()} locked` : "";
    setStatus("info", `Cache cleared: removed ${Number(data.deleted_files ?? 0).toLocaleString()} files (${formatFileSize(data.deleted_bytes ?? null)})${skippedText}.`);
    await refreshCacheStats();
  } catch (error) {
    setStatus("error", `Cache clear failed: ${error}`);
  }
});

ui.meshColor?.addEventListener("input", updateMeshColor);
ui.sunAzimuth?.addEventListener("input", updateSunDirection);
ui.sunElevation?.addEventListener("input", updateSunDirection);
ui.viewerBedSize?.addEventListener("input", updatePrintBedSize);

updateOverlay();
updateSunDirection();
updatePrintBedSize();
setStatus("idle", "Idle");
clearStatusMeta();
setTimeout(() => map.invalidateSize(), 0);
await loadParameterHelp();
await refreshHistory();
await refreshCacheStats();
