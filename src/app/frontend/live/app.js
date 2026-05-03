const EARTH_RADIUS_M = 6371000;
const EARTH_OMEGA = 7.292115e-5; // rad/s, Earth rotation rate

// Ground-relative speed: subtract atmosphere co-rotation (omega x position)
function groundRelativeSpeed(pos, vel) {
    // omega x pos = [-omega*py, omega*px, 0]
    const relVx = vel[0] - (-EARTH_OMEGA * pos[1]);
    const relVy = vel[1] - (EARTH_OMEGA * pos[0]);
    const relVz = vel[2];
    return Math.hypot(relVx, relVy, relVz);
}
const MAX_POINTS = 1500;
const ORBIT_RENDER_MIN_INTERVAL_MS = 50;
const ORBIT_INTERACTION_POLL_MS = 120;
const TELEMETRY_LOG_EVERY_N = 100;
const DEFAULT_ROCKET_TELEMETRY_INTERVAL_S = 0.5;

// ---------------------------------------------------------------------------
// Constellation configuration (dynamic vehicle discovery)
// ---------------------------------------------------------------------------

const DEFAULT_VEHICLE_COLORS = [
    "#66d6ff",
    "#f07f3c",
    "#a78bfa",
    "#78f0b2",
    "#ffd166",
    "#ff8d8d",
    "#7dd3fc",
    "#f9a8d4",
];

const constellationVehicleIds = [];
const vehicleColorMap = {};
// Plotly trace indices in constellation mode (Earth mesh is always index 0)
const vehicleTraceIndices = {};

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const runIdInput = document.getElementById("runId");
const cmdActionInput = document.getElementById("cmdAction");
const cmdExecuteAtInput = document.getElementById("cmdExecuteAt");
const cmdTargetPerigeeInput = document.getElementById("cmdTargetPerigee");
const connState = document.getElementById("connState");
const phaseValue = document.getElementById("phaseValue");
const altValue = document.getElementById("altValue");
const spdValue = document.getElementById("spdValue");
const logEl = document.getElementById("log");
const orbitPlotEl = document.getElementById("orbitPlot");

// ---------------------------------------------------------------------------
// Single-vehicle state
// ---------------------------------------------------------------------------

let ws = null;

const altData = { t: [], y: [] };
const spdData = { t: [], y: [] };
const orbitData = { x: [], y: [], z: [] };

// ---------------------------------------------------------------------------
// Constellation state
// ---------------------------------------------------------------------------

let constellationMode = false;
let activeVehicle = null;
const vehicleData = {};

function initVehicleData(vid) {
    vehicleData[vid] = {
        altData: { t: [], y: [] },
        spdData: { t: [], y: [] },
        orbit: { x: [], y: [], z: [] },
        orbitPlottedCount: 0,
        phaseSeen: [],
        logCount: 0,
    };
}

function getVehicleColor(vid) {
    return vehicleColorMap[vid] || "#66d6ff";
}

function registerConstellationVehicle(vid) {
    if (constellationVehicleIds.includes(vid)) {
        return;
    }

    constellationVehicleIds.push(vid);
    vehicleColorMap[vid] = DEFAULT_VEHICLE_COLORS[
        (constellationVehicleIds.length - 1) % DEFAULT_VEHICLE_COLORS.length
    ];
    const idx = constellationVehicleIds.length - 1;
    vehicleTraceIndices[vid] = {
        trajectory: 1 + idx * 2,
        marker: 2 + idx * 2,
    };
    initVehicleData(vid);
}

// ---------------------------------------------------------------------------
// Orbit plot state
// ---------------------------------------------------------------------------

let orbitDirty = false;
let orbitFramePending = false;
let orbitEnabled = false;
let orbitResizeBound = false;
let orbitLastRenderAt = 0;
let orbitRenderTimer = null;
let orbitUserInteracting = false;
let orbitInteractionDebounce = null;
let orbitPlottedCount = 0;
let telemetryLogCount = 0;
let savedOrbitCamera = null;

const PLOTLY_CDN_URLS = [
    "https://cdn.plot.ly/plotly-2.35.2.min.js",
    "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js",
];

// ---------------------------------------------------------------------------
// Chart.js setup
// ---------------------------------------------------------------------------

const commonChartOptions = {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    resizeDelay: 200,
    scales: {
        x: {
            title: { display: true, text: "Time (s)", color: "#9ab2df" },
            ticks: {
                color: "#9ab2df",
                maxTicksLimit: 8,
                maxRotation: 0,
                callback(value) {
                    const label = Number(this.getLabelForValue(value));
                    return Number.isFinite(label) ? Math.round(label).toString() : "";
                },
            },
            grid: { color: "rgba(88,122,177,0.25)" },
        },
        y: {
            ticks: { color: "#9ab2df" },
            grid: { color: "rgba(88,122,177,0.25)" },
        },
    },
    plugins: {
        legend: {
            labels: { color: "#cfe2ff" },
            // Keep the live dataset always visible; legend clicks can hide traces.
            onClick: () => { },
        },
    },
};

const altChart = new Chart(document.getElementById("altChart"), {
    type: "line",
    data: {
        labels: altData.t,
        datasets: [
            {
                label: "Altitude (km)",
                data: altData.y,
                borderColor: "#66d6ff",
                backgroundColor: "rgba(102,214,255,0.15)",
                pointRadius: 0,
                tension: 0.2,
            },
        ],
    },
    options: {
        ...commonChartOptions,
        scales: {
            ...commonChartOptions.scales,
            y: {
                ...commonChartOptions.scales.y,
                title: { display: true, text: "Altitude (km)", color: "#9ab2df" },
            },
        },
    },
});

const spdChart = new Chart(document.getElementById("spdChart"), {
    type: "line",
    data: {
        labels: spdData.t,
        datasets: [
            {
                label: "Speed (km/s)",
                data: spdData.y,
                borderColor: "#78f0b2",
                backgroundColor: "rgba(120,240,178,0.15)",
                pointRadius: 0,
                tension: 0.2,
            },
        ],
    },
    options: {
        ...commonChartOptions,
        scales: {
            ...commonChartOptions.scales,
            y: {
                ...commonChartOptions.scales.y,
                title: { display: true, text: "Speed (km/s)", color: "#9ab2df" },
            },
        },
    },
});

// ---------------------------------------------------------------------------
// Orbit plot helpers
// ---------------------------------------------------------------------------

function buildEarthSphereMesh() {
    const radiusKm = EARTH_RADIUS_M / 1000;
    const latSteps = 28;
    const lonSteps = 56;
    const x = [];
    const y = [];
    const z = [];

    for (let i = 0; i <= latSteps; i += 1) {
        const theta = (Math.PI * i) / latSteps;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);
        const rowX = [];
        const rowY = [];
        const rowZ = [];
        for (let j = 0; j <= lonSteps; j += 1) {
            const phi = (2 * Math.PI * j) / lonSteps;
            rowX.push(radiusKm * sinTheta * Math.cos(phi));
            rowY.push(radiusKm * sinTheta * Math.sin(phi));
            rowZ.push(radiusKm * cosTheta);
        }
        x.push(rowX);
        y.push(rowY);
        z.push(rowZ);
    }

    const fi = [];
    const fj = [];
    const fk = [];
    const rowLen = lonSteps + 1;
    for (let lat = 0; lat < latSteps; lat += 1) {
        for (let lon = 0; lon < lonSteps; lon += 1) {
            const a = lat * rowLen + lon;
            const b = a + 1;
            const c = a + rowLen;
            const d = c + 1;
            fi.push(a, b);
            fj.push(c, c);
            fk.push(b, d);
        }
    }

    return { x: x.flat(), y: y.flat(), z: z.flat(), i: fi, j: fj, k: fk };
}

function buildEarthTrace(earthMesh) {
    return {
        type: "mesh3d",
        x: earthMesh.x, y: earthMesh.y, z: earthMesh.z,
        i: earthMesh.i, j: earthMesh.j, k: earthMesh.k,
        color: "#2a5b88", opacity: 1.0, flatshading: true,
        lighting: { ambient: 0.85, diffuse: 0.45, specular: 0.05, roughness: 0.95, fresnel: 0.0 },
        lightposition: { x: 0, y: 0, z: 1e6 },
        hoverinfo: "skip", showlegend: false,
    };
}

function buildOrbitLayout() {
    const earthRadiusKm = EARTH_RADIUS_M / 1000;
    const axisRange = [-earthRadiusKm * 1.7, earthRadiusKm * 1.7];
    return {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "#0e1830",
        plot_bgcolor: "#0e1830",
        scene: {
            aspectmode: "data",
            uirevision: "orbit-camera",
            camera: { eye: { x: 1.45, y: 1.45, z: 0.95 } },
            xaxis: { title: "X (km)", color: "#9ab2df", range: axisRange, showspikes: false },
            yaxis: { title: "Y (km)", color: "#9ab2df", range: axisRange, showspikes: false },
            zaxis: { title: "Z (km)", color: "#9ab2df", range: axisRange, showspikes: false },
        },
        hovermode: false,
        showlegend: true,
        legend: {
            font: { color: "#cfe2ff", size: 11 },
            bgcolor: "rgba(14,24,48,0.7)",
            bordercolor: "#274068",
            borderwidth: 1,
        },
    };
}

function setOrbitPlaceholder(message, isError = false) {
    if (!orbitPlotEl) { return; }
    orbitPlotEl.innerHTML = "";
    const msg = document.createElement("div");
    msg.className = isError ? "orbit-placeholder error" : "orbit-placeholder";
    msg.textContent = message;
    orbitPlotEl.appendChild(msg);
}

function clearOrbitPlaceholder() {
    if (!orbitPlotEl) { return; }
    orbitPlotEl.innerHTML = "";
}

function loadScript(url) {
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = url;
        script.async = true;
        script.onload = () => resolve(url);
        script.onerror = () => reject(new Error(`failed to load ${url}`));
        document.head.appendChild(script);
    });
}

async function ensurePlotlyLoaded() {
    if (window.Plotly) { return true; }
    for (const url of PLOTLY_CDN_URLS) {
        try {
            await loadScript(url);
            if (window.Plotly) { return true; }
        } catch {
            // Try next mirror.
        }
    }
    return Boolean(window.Plotly);
}

// ---------------------------------------------------------------------------
// Single-vehicle orbit plot
// ---------------------------------------------------------------------------

function initOrbitPlot() {
    if (!window.Plotly || !orbitPlotEl) { return; }
    clearOrbitPlaceholder();

    const earthMesh = buildEarthSphereMesh();
    const traces = [
        buildEarthTrace(earthMesh),
        {
            type: "scatter3d", mode: "lines",
            x: [], y: [], z: [],
            line: { color: "#66d6ff", width: 5 },
            name: "Trajectory", hoverinfo: "skip", showlegend: false,
        },
        {
            type: "scatter3d", mode: "markers",
            x: [], y: [], z: [],
            marker: { size: 6, color: "#ffd166" },
            name: "Current", hoverinfo: "skip", showlegend: false,
        },
    ];

    try {
        window.Plotly.newPlot(orbitPlotEl, traces, buildOrbitLayout(), {
            displayModeBar: false,
            responsive: true,
        });
        orbitEnabled = true;
        bindOrbitInteractionEvents();
        requestAnimationFrame(() => { window.Plotly.Plots.resize(orbitPlotEl); });
    } catch (err) {
        orbitEnabled = false;
        setOrbitPlaceholder(`3D view unavailable: ${String(err)}`, true);
    }
}

// ---------------------------------------------------------------------------
// Constellation orbit plot (one trajectory + marker trace per satellite)
// ---------------------------------------------------------------------------

function reinitOrbitForConstellation() {
    if (!window.Plotly || !orbitPlotEl) { return; }

    const earthMesh = buildEarthSphereMesh();
    const traces = [buildEarthTrace(earthMesh)];

    for (const vid of constellationVehicleIds) {
        const color = getVehicleColor(vid);
        traces.push({
            type: "scatter3d", mode: "lines",
            x: [], y: [], z: [],
            line: { color, width: 4 },
            name: vid, hoverinfo: "skip",
        });
        traces.push({
            type: "scatter3d", mode: "markers",
            x: [], y: [], z: [],
            marker: { size: 8, color, symbol: "circle" },
            name: `${vid} pos`, hoverinfo: "skip", showlegend: false,
        });
    }

    // Preserve camera position/zoom if the user has moved the camera.
    const layout = buildOrbitLayout();
    if (savedOrbitCamera) {
        layout.scene.camera = savedOrbitCamera;
    }

    try {
        window.Plotly.newPlot(orbitPlotEl, traces, layout, {
            displayModeBar: false,
            responsive: true,
        });
        orbitEnabled = true;
        orbitPlottedCount = 0;
        bindOrbitInteractionEvents();
        requestAnimationFrame(() => { window.Plotly.Plots.resize(orbitPlotEl); });
    } catch (err) {
        orbitEnabled = false;
        setOrbitPlaceholder(`3D view unavailable: ${String(err)}`, true);
    }
}

// ---------------------------------------------------------------------------
// Orbit interaction event binding
// ---------------------------------------------------------------------------

function bindOrbitInteractionEvents() {
    if (!orbitEnabled || !orbitPlotEl || !orbitPlotEl.on) { return; }

    orbitPlotEl.on("plotly_relayouting", () => {
        orbitUserInteracting = true;
        if (orbitInteractionDebounce) {
            clearTimeout(orbitInteractionDebounce);
            orbitInteractionDebounce = null;
        }
    });

    orbitPlotEl.on("plotly_relayout", (eventData) => {
        if (eventData && eventData["scene.camera"]) {
            savedOrbitCamera = eventData["scene.camera"];
        }
        if (orbitInteractionDebounce) { clearTimeout(orbitInteractionDebounce); }
        orbitInteractionDebounce = setTimeout(() => {
            orbitUserInteracting = false;
            if (orbitDirty) { scheduleOrbitRender(); }
        }, 180);
    });
}

function bindOrbitResize() {
    if (orbitResizeBound) { return; }
    orbitResizeBound = true;
    window.addEventListener("resize", () => {
        if (!orbitEnabled || !window.Plotly || !orbitPlotEl) { return; }
        window.Plotly.Plots.resize(orbitPlotEl);
    });
}

// ---------------------------------------------------------------------------
// Orbit render — single-vehicle path
// ---------------------------------------------------------------------------

function renderSingleOrbit() {
    const total = orbitData.x.length;
    const lastIdx = total - 1;

    if (total > orbitPlottedCount) {
        const newX = orbitData.x.slice(orbitPlottedCount);
        const newY = orbitData.y.slice(orbitPlottedCount);
        const newZ = orbitData.z.slice(orbitPlottedCount);
        window.Plotly.extendTraces(orbitPlotEl, { x: [newX], y: [newY], z: [newZ] }, [1]);
        orbitPlottedCount = total;
    }

    const currentX = lastIdx >= 0 ? [orbitData.x[lastIdx]] : [];
    const currentY = lastIdx >= 0 ? [orbitData.y[lastIdx]] : [];
    const currentZ = lastIdx >= 0 ? [orbitData.z[lastIdx]] : [];
    window.Plotly.restyle(orbitPlotEl, { x: [currentX], y: [currentY], z: [currentZ] }, [2]);
}

// ---------------------------------------------------------------------------
// Orbit render — constellation path
// ---------------------------------------------------------------------------

function renderConstellationOrbit() {
    for (const vid of constellationVehicleIds) {
        if (!vehicleData[vid]) { continue; }
        const vd = vehicleData[vid];
        const idxs = vehicleTraceIndices[vid];
        if (!idxs) { continue; }
        const total = vd.orbit.x.length;

        if (total > vd.orbitPlottedCount) {
            const newX = vd.orbit.x.slice(vd.orbitPlottedCount);
            const newY = vd.orbit.y.slice(vd.orbitPlottedCount);
            const newZ = vd.orbit.z.slice(vd.orbitPlottedCount);
            window.Plotly.extendTraces(
                orbitPlotEl, { x: [newX], y: [newY], z: [newZ] }, [idxs.trajectory]
            );
            vd.orbitPlottedCount = total;
        }

        const lastIdx = total - 1;
        const cx = lastIdx >= 0 ? [vd.orbit.x[lastIdx]] : [];
        const cy = lastIdx >= 0 ? [vd.orbit.y[lastIdx]] : [];
        const cz = lastIdx >= 0 ? [vd.orbit.z[lastIdx]] : [];
        window.Plotly.restyle(orbitPlotEl, { x: [cx], y: [cy], z: [cz] }, [idxs.marker]);
    }
}

// ---------------------------------------------------------------------------
// Orbit render scheduler
// ---------------------------------------------------------------------------

function renderOrbitIfNeeded() {
    orbitFramePending = false;
    if (!orbitEnabled || !orbitDirty || !window.Plotly || !orbitPlotEl) { return; }

    if (orbitUserInteracting) {
        // Never restyle traces while dragging; this avoids camera snap-back.
        if (!orbitRenderTimer) {
            orbitRenderTimer = setTimeout(() => {
                orbitRenderTimer = null;
                scheduleOrbitRender();
            }, ORBIT_INTERACTION_POLL_MS);
        }
        return;
    }

    orbitDirty = false;
    orbitLastRenderAt = performance.now();

    if (constellationMode) {
        renderConstellationOrbit();
    } else {
        renderSingleOrbit();
    }
}

function scheduleOrbitRender() {
    if (!orbitEnabled) { return; }
    orbitDirty = true;
    if (orbitFramePending || orbitRenderTimer) { return; }

    if (orbitUserInteracting) {
        orbitRenderTimer = setTimeout(() => {
            orbitRenderTimer = null;
            scheduleOrbitRender();
        }, ORBIT_INTERACTION_POLL_MS);
        return;
    }

    const elapsed = performance.now() - orbitLastRenderAt;
    if (elapsed >= ORBIT_RENDER_MIN_INTERVAL_MS) {
        orbitFramePending = true;
        window.requestAnimationFrame(renderOrbitIfNeeded);
        return;
    }

    orbitRenderTimer = setTimeout(() => {
        orbitRenderTimer = null;
        orbitFramePending = true;
        window.requestAnimationFrame(renderOrbitIfNeeded);
    }, ORBIT_RENDER_MIN_INTERVAL_MS - elapsed);
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function setState(text, color = "#78f0b2") {
    connState.textContent = text;
    connState.style.color = color;
}

function appendLine(type, payload) {
    const line = document.createElement("div");
    line.className = `line-${type}`;
    line.textContent = `[${new Date().toISOString()}] ${JSON.stringify(payload)}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
}

function closeWs() {
    if (ws) { ws.close(); ws = null; }
}

// ---------------------------------------------------------------------------
// Vehicle tab UI (constellation mode)
// ---------------------------------------------------------------------------

function renderVehicleTabs() {
    const tabEl = document.getElementById("vehicleTabs");
    if (!tabEl) { return; }
    tabEl.innerHTML = "";
    tabEl.style.display = "flex";

    for (const vid of constellationVehicleIds) {
        const tab = document.createElement("button");
        tab.className = "vehicle-tab" + (vid === activeVehicle ? " active" : "");
        tab.textContent = vid;
        tab.style.setProperty("--tab-color", getVehicleColor(vid));
        tab.addEventListener("click", () => setActiveVehicle(vid));
        tabEl.appendChild(tab);
    }
}

function setActiveVehicle(vid) {
    activeVehicle = vid;
    renderVehicleTabs();
    if (!vehicleData[vid]) { return; }
    const vd = vehicleData[vid];
    const color = getVehicleColor(vid);

    altChart.data.labels = vd.altData.t;
    altChart.data.datasets[0].data = vd.altData.y;
    altChart.data.datasets[0].label = `Altitude (km) — ${vid}`;
    altChart.data.datasets[0].borderColor = color;
    altChart.data.datasets[0].backgroundColor = `${color}26`;
    altChart.data.datasets[0].hidden = false;
    altChart.update("none");

    spdChart.data.labels = vd.spdData.t;
    spdChart.data.datasets[0].data = vd.spdData.y;
    spdChart.data.datasets[0].label = `Speed (km/s) — ${vid}`;
    spdChart.data.datasets[0].borderColor = color;
    spdChart.data.datasets[0].backgroundColor = `${color}26`;
    spdChart.data.datasets[0].hidden = false;
    spdChart.update("none");

    const altArr = vd.altData.y;
    const spdArr = vd.spdData.y;
    altValue.textContent = altArr.length ? altArr[altArr.length - 1].toFixed(2) : "-";
    spdValue.textContent = spdArr.length ? spdArr[spdArr.length - 1].toFixed(3) : "-";
    phaseValue.textContent = vd.phaseSeen[vd.phaseSeen.length - 1] || "-";
}

// ---------------------------------------------------------------------------
// Phase strip
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

function pushPoint(buf, t, y) {
    buf.t.push(t);
    buf.y.push(y);
    if (buf.t.length > MAX_POINTS) {
        buf.t.shift();
        buf.y.shift();
    }
}

// ---------------------------------------------------------------------------
// Telemetry handlers
// ---------------------------------------------------------------------------

function handleSingleTelemetry(frame) {
    const t = Number(frame.time_s ?? 0);
    const pos = frame.position_m || [0, 0, 0];
    const vel = frame.velocity_ms || [0, 0, 0];

    const radius = Math.hypot(pos[0], pos[1], pos[2]);
    const altitudeKm = (radius - EARTH_RADIUS_M) / 1000;
    const speedKms = groundRelativeSpeed(pos, vel) / 1000;

    altValue.textContent = Number.isFinite(altitudeKm) ? altitudeKm.toFixed(2) : "-";
    spdValue.textContent = Number.isFinite(speedKms) ? speedKms.toFixed(3) : "-";
    phaseValue.textContent = frame.phase || "-";

    pushPoint(altData, t, altitudeKm);
    pushPoint(spdData, t, speedKms);

    orbitData.x.push(pos[0] / 1000);
    orbitData.y.push(pos[1] / 1000);
    orbitData.z.push(pos[2] / 1000);
    scheduleOrbitRender();

    altChart.update("none");
    spdChart.update("none");
}

function handleConstellationTelemetry(frame) {
    const vid = frame.vehicle_id;
    if (!vid) {
        return;
    }

    // On first constellation frame: initialize constellation mode
    if (!constellationMode) {
        constellationMode = true;
    }

    if (!vehicleData[vid]) {
        registerConstellationVehicle(vid);
        if (!activeVehicle) {
            activeVehicle = vid;
        }
        reinitOrbitForConstellation();
        renderVehicleTabs();
        setActiveVehicle(activeVehicle);
    }

    if (!vehicleData[vid]) { return; }
    const vd = vehicleData[vid];

    const t = Number(frame.time_s ?? 0);
    const pos = frame.position_m || [0, 0, 0];
    const vel = frame.velocity_ms || [0, 0, 0];

    const radius = Math.hypot(pos[0], pos[1], pos[2]);
    const altKm = (radius - EARTH_RADIUS_M) / 1000;
    const spdKms = groundRelativeSpeed(pos, vel) / 1000;

    pushPoint(vd.altData, t, altKm);
    pushPoint(vd.spdData, t, spdKms);
    vd.orbit.x.push(pos[0] / 1000);
    vd.orbit.y.push(pos[1] / 1000);
    vd.orbit.z.push(pos[2] / 1000);

    if (frame.phase && !vd.phaseSeen.includes(frame.phase)) {
        vd.phaseSeen.push(frame.phase);
    }

    // Update KPIs and 2D charts only for the selected vehicle
    if (vid === activeVehicle) {
        altValue.textContent = Number.isFinite(altKm) ? altKm.toFixed(2) : "-";
        spdValue.textContent = Number.isFinite(spdKms) ? spdKms.toFixed(3) : "-";
        phaseValue.textContent = frame.phase || "-";
        altChart.update("none");
        spdChart.update("none");
    }

    scheduleOrbitRender();
}

function handleTelemetry(frame) {
    if (frame.vehicle_id) {
        handleConstellationTelemetry(frame);
    } else {
        handleSingleTelemetry(frame);
    }
}

// ---------------------------------------------------------------------------
// Plot reset
// ---------------------------------------------------------------------------

function resetPlots() {
    altData.t.length = 0;
    altData.y.length = 0;
    spdData.t.length = 0;
    spdData.y.length = 0;
    orbitData.x.length = 0;
    orbitData.y.length = 0;
    orbitData.z.length = 0;
    orbitPlottedCount = 0;
    telemetryLogCount = 0;

    // Restore chart references to single-vehicle arrays and default styling
    altChart.data.labels = altData.t;
    altChart.data.datasets[0].data = altData.y;
    altChart.data.datasets[0].label = "Altitude (km)";
    altChart.data.datasets[0].borderColor = "#66d6ff";
    altChart.data.datasets[0].backgroundColor = "rgba(102,214,255,0.15)";
    altChart.update("none");

    spdChart.data.labels = spdData.t;
    spdChart.data.datasets[0].data = spdData.y;
    spdChart.data.datasets[0].label = "Speed (km/s)";
    spdChart.data.datasets[0].borderColor = "#78f0b2";
    spdChart.data.datasets[0].backgroundColor = "rgba(120,240,178,0.15)";
    spdChart.update("none");

    phaseValue.textContent = "-";
    altValue.textContent = "-";
    spdValue.textContent = "-";

    altChart.data.datasets[0].hidden = false;
    spdChart.data.datasets[0].hidden = false;

    // Reset constellation state
    constellationMode = false;
    activeVehicle = null;
    for (const vid of constellationVehicleIds) { delete vehicleData[vid]; }
    constellationVehicleIds.length = 0;
    for (const key of Object.keys(vehicleColorMap)) { delete vehicleColorMap[key]; }
    for (const key of Object.keys(vehicleTraceIndices)) { delete vehicleTraceIndices[key]; }
    const tabEl = document.getElementById("vehicleTabs");
    if (tabEl) { tabEl.style.display = "none"; tabEl.innerHTML = ""; }

    // Reinit orbit plot to single-vehicle mode
    if (orbitEnabled && window.Plotly && orbitPlotEl) {
        try { window.Plotly.purge(orbitPlotEl); } catch { /* ignore */ }
        orbitEnabled = false;
    }
    orbitDirty = false;
    orbitUserInteracting = false;
    if (orbitRenderTimer) { clearTimeout(orbitRenderTimer); orbitRenderTimer = null; }
    if (orbitInteractionDebounce) { clearTimeout(orbitInteractionDebounce); orbitInteractionDebounce = null; }
    orbitFramePending = false;

    if (window.Plotly) { initOrbitPlot(); }
}

// ---------------------------------------------------------------------------
// WebSocket connection
// ---------------------------------------------------------------------------

function connectRun(runId) {
    closeWs();

    const wsProto = location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${wsProto}://${location.host}/simulations/live/${runId}/ws`;
    ws = new WebSocket(wsUrl);

    setState("connecting", "#ffc47c");
    appendLine("status", { event: "connecting", run_id: runId, ws_url: wsUrl });

    ws.onopen = () => setState("connected", "#78f0b2");
    ws.onclose = () => setState("closed", "#ffc47c");
    ws.onerror = () => setState("error", "#ff8d8d");

    ws.onmessage = (evt) => {
        try {
            const msg = JSON.parse(evt.data);
            if (msg.type === "telemetry") {
                handleTelemetry(msg.data);
                const d = msg.data;
                const pos = d.position_m || [0, 0, 0];
                const vel = d.velocity_ms || [0, 0, 0];
                const _altKm = (Math.hypot(pos[0], pos[1], pos[2]) - EARTH_RADIUS_M) / 1000;
                const _spdKms = groundRelativeSpeed(pos, vel) / 1000;
                const _logEntry = {
                    seq: d.seq,
                    time_s: d.time_s,
                    ...(d.vehicle_id ? { vehicle_id: d.vehicle_id } : {}),
                    alt_km: Number.isFinite(_altKm) ? +_altKm.toFixed(2) : null,
                    spd_km_s: Number.isFinite(_spdKms) ? +_spdKms.toFixed(3) : null,
                };
                const _vid = d.vehicle_id;
                if (_vid && vehicleData[_vid]) {
                    vehicleData[_vid].logCount += 1;
                    if (vehicleData[_vid].logCount % TELEMETRY_LOG_EVERY_N === 0) {
                        appendLine("telemetry", _logEntry);
                    }
                } else {
                    telemetryLogCount += 1;
                    if (telemetryLogCount % TELEMETRY_LOG_EVERY_N === 0) {
                        appendLine("telemetry", _logEntry);
                    }
                }
            } else if (msg.type === "command") {
                appendLine("status", msg.data);
            } else if (msg.type === "status") {
                appendLine("status", msg.data);
                const status = msg.data?.status || "unknown";
                setState(
                    `finished: ${status}`,
                    status === "completed" ? "#78f0b2" : "#ff8d8d"
                );
            } else {
                appendLine("status", msg);
            }
        } catch (err) {
            appendLine("error", { event: "parse_error", raw: evt.data, error: String(err) });
        }
    };
}

// ---------------------------------------------------------------------------
// Start actions
// ---------------------------------------------------------------------------

async function startAndConnect() {
    try {
        const resp = await fetch(
            `/simulations/live/start?telemetry_interval=${DEFAULT_ROCKET_TELEMETRY_INTERVAL_S}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: "{}",
            }
        );

        if (!resp.ok) { throw new Error(`start failed: ${resp.status}`); }

        const data = await resp.json();
        runIdInput.value = data.run_id;
        appendLine("status", { event: "run_started", ...data });
        resetPlots();
        connectRun(data.run_id);
    } catch (err) {
        setState("error", "#ff8d8d");
        appendLine("error", { event: "start_error", error: String(err) });
    }
}

async function startConstellation() {
    try {
        const resp = await fetch("/simulations/live/constellation/start", { method: "POST" });
        if (!resp.ok) { throw new Error(`constellation start failed: ${resp.status}`); }
        const data = await resp.json();
        runIdInput.value = data.run_id;
        appendLine("status", { event: "constellation_started", ...data });
        resetPlots();
        connectRun(data.run_id);
    } catch (err) {
        setState("error", "#ff8d8d");
        appendLine("error", { event: "start_error", error: String(err) });
    }
}

async function sendDeorbitCommand() {
    try {
        const runId = (runIdInput.value || "").trim();
        const vehicleId = activeVehicle;
        const action = (cmdActionInput?.value || "deorbit_burn").trim();
        const executeAt = Number(cmdExecuteAtInput.value || "0");
        const targetPerigee = Number(cmdTargetPerigeeInput.value || "0");

        if (!runId) {
            appendLine("error", { event: "validation", message: "run_id is required" });
            return;
        }
        if (!vehicleId) {
            appendLine("error", {
                event: "validation",
                message: "No active satellite selected. Launch/connect to a constellation run and pick a satellite tab.",
            });
            return;
        }

        const resp = await fetch("/simulations/live/constellation/command", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                run_id: runId,
                vehicle_id: vehicleId,
                action,
                execute_at_sim_time_s: executeAt,
                target_perigee_alt_km: targetPerigee,
            }),
        });

        const data = await resp.json();
        appendLine("status", { event: "command_upload", ...data });
    } catch (err) {
        appendLine("error", { event: "command_upload_error", error: String(err) });
    }
}

// ---------------------------------------------------------------------------
// Button wiring
// ---------------------------------------------------------------------------

document.getElementById("startBtn").addEventListener("click", startAndConnect);
document.getElementById("constellationBtn").addEventListener("click", startConstellation);
document.getElementById("sendCommandBtn").addEventListener("click", sendDeorbitCommand);

document.getElementById("connectBtn").addEventListener("click", () => {
    const runId = (runIdInput.value || "").trim();
    if (!runId) {
        appendLine("error", { event: "validation", message: "run_id is required" });
        return;
    }
    resetPlots();
    connectRun(runId);
});

document.getElementById("disconnectBtn").addEventListener("click", closeWs);

document.getElementById("clearBtn").addEventListener("click", () => {
    logEl.textContent = "";
});

// ---------------------------------------------------------------------------
// Orbit section init
// ---------------------------------------------------------------------------

async function initOrbitSection() {
    setOrbitPlaceholder("Initializing 3D orbit view...");
    const loaded = await ensurePlotlyLoaded();
    if (!loaded) {
        setOrbitPlaceholder(
            "Could not load Plotly from CDN mirrors. Check internet/proxy and hard refresh.",
            true
        );
        return;
    }

    initOrbitPlot();
    if (!orbitEnabled) { return; }

    bindOrbitResize();
    scheduleOrbitRender();
}

initOrbitSection();
