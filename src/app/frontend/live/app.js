const EARTH_RADIUS_M = 6371000; // Approximate Earth radius for live altitude conversion.
const MAX_POINTS = 1500;
const ORBIT_RENDER_MIN_INTERVAL_MS = 50;
const ORBIT_INTERACTION_POLL_MS = 120;
const TELEMETRY_LOG_EVERY_N = 100;

const runIdInput = document.getElementById("runId");
const intervalInput = document.getElementById("interval");
const connState = document.getElementById("connState");
const phaseValue = document.getElementById("phaseValue");
const altValue = document.getElementById("altValue");
const spdValue = document.getElementById("spdValue");
const phaseStrip = document.getElementById("phaseStrip");
const logEl = document.getElementById("log");
const orbitPlotEl = document.getElementById("orbitPlot");

let ws = null;
let phaseSeen = [];

const altData = { t: [], y: [] };
const spdData = { t: [], y: [] };
const orbitData = { x: [], y: [], z: [] };

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

const PLOTLY_CDN_URLS = [
    "https://cdn.plot.ly/plotly-2.35.2.min.js",
    "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js",
];

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
        legend: { labels: { color: "#cfe2ff" } },
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

    const i = [];
    const j = [];
    const k = [];

    const rowLen = lonSteps + 1;
    for (let lat = 0; lat < latSteps; lat += 1) {
        for (let lon = 0; lon < lonSteps; lon += 1) {
            const a = lat * rowLen + lon;
            const b = a + 1;
            const c = a + rowLen;
            const d = c + 1;

            i.push(a, b);
            j.push(c, c);
            k.push(b, d);
        }
    }

    const flatX = x.flat();
    const flatY = y.flat();
    const flatZ = z.flat();
    return { x: flatX, y: flatY, z: flatZ, i, j, k };
}

function setOrbitPlaceholder(message, isError = false) {
    if (!orbitPlotEl) {
        return;
    }

    orbitPlotEl.innerHTML = "";
    const msg = document.createElement("div");
    msg.className = isError ? "orbit-placeholder error" : "orbit-placeholder";
    msg.textContent = message;
    orbitPlotEl.appendChild(msg);
}

function clearOrbitPlaceholder() {
    if (!orbitPlotEl) {
        return;
    }
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
    if (window.Plotly) {
        return true;
    }

    for (const url of PLOTLY_CDN_URLS) {
        try {
            await loadScript(url);
            if (window.Plotly) {
                return true;
            }
        } catch {
            // Try next mirror.
        }
    }

    return Boolean(window.Plotly);
}

function initOrbitPlot() {
    if (!window.Plotly || !orbitPlotEl) {
        return;
    }

    clearOrbitPlaceholder();

    const earthMesh = buildEarthSphereMesh();
    const earthRadiusKm = EARTH_RADIUS_M / 1000;

    const traces = [
        {
            type: "mesh3d",
            x: earthMesh.x,
            y: earthMesh.y,
            z: earthMesh.z,
            i: earthMesh.i,
            j: earthMesh.j,
            k: earthMesh.k,
            color: "#2a5b88",
            opacity: 1.0,
            flatshading: true,
            lighting: {
                ambient: 0.85,
                diffuse: 0.45,
                specular: 0.05,
                roughness: 0.95,
                fresnel: 0.0,
            },
            lightposition: { x: 0, y: 0, z: 1e6 },
            hoverinfo: "skip",
        },
        {
            type: "scatter3d",
            mode: "lines",
            x: [],
            y: [],
            z: [],
            line: {
                color: "#66d6ff",
                width: 5,
            },
            name: "Trajectory",
            hoverinfo: "skip",
        },
        {
            type: "scatter3d",
            mode: "markers",
            x: [],
            y: [],
            z: [],
            marker: {
                size: 6,
                color: "#ffd166",
            },
            name: "Current",
            hoverinfo: "skip",
        },
    ];

    const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "#0e1830",
        plot_bgcolor: "#0e1830",
        scene: {
            aspectmode: "data",
            uirevision: "orbit-camera",
            camera: {
                eye: { x: 1.45, y: 1.45, z: 0.95 },
            },
            xaxis: {
                title: "X (km)",
                color: "#9ab2df",
                range: [-earthRadiusKm * 1.7, earthRadiusKm * 1.7],
                showspikes: false,
            },
            yaxis: {
                title: "Y (km)",
                color: "#9ab2df",
                range: [-earthRadiusKm * 1.7, earthRadiusKm * 1.7],
                showspikes: false,
            },
            zaxis: {
                title: "Z (km)",
                color: "#9ab2df",
                range: [-earthRadiusKm * 1.7, earthRadiusKm * 1.7],
                showspikes: false,
            },
        },
        hovermode: false,
        showlegend: false,
    };

    try {
        window.Plotly.newPlot(orbitPlotEl, traces, layout, {
            displayModeBar: false,
            responsive: true,
        });
        orbitEnabled = true;
        bindOrbitInteractionEvents();
        requestAnimationFrame(() => {
            window.Plotly.Plots.resize(orbitPlotEl);
        });
    } catch (err) {
        orbitEnabled = false;
        setOrbitPlaceholder(`3D view unavailable: ${String(err)}`, true);
    }
}

function bindOrbitInteractionEvents() {
    if (!orbitEnabled || !orbitPlotEl || !orbitPlotEl.on) {
        return;
    }

    orbitPlotEl.on("plotly_relayouting", () => {
        orbitUserInteracting = true;
        if (orbitInteractionDebounce) {
            clearTimeout(orbitInteractionDebounce);
            orbitInteractionDebounce = null;
        }
    });

    orbitPlotEl.on("plotly_relayout", () => {
        if (orbitInteractionDebounce) {
            clearTimeout(orbitInteractionDebounce);
        }
        orbitInteractionDebounce = setTimeout(() => {
            orbitUserInteracting = false;
            if (orbitDirty) {
                scheduleOrbitRender();
            }
        }, 180);
    });
}

function bindOrbitResize() {
    if (orbitResizeBound) {
        return;
    }
    orbitResizeBound = true;

    window.addEventListener("resize", () => {
        if (!orbitEnabled || !window.Plotly || !orbitPlotEl) {
            return;
        }
        window.Plotly.Plots.resize(orbitPlotEl);
    });
}

function renderOrbitIfNeeded() {
    orbitFramePending = false;
    if (!orbitEnabled || !orbitDirty || !window.Plotly || !orbitPlotEl) {
        return;
    }

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
    window.Plotly.restyle(
        orbitPlotEl,
        { x: [currentX], y: [currentY], z: [currentZ] },
        [2]
    );
}

function scheduleOrbitRender() {
    if (!orbitEnabled) {
        return;
    }

    orbitDirty = true;
    if (orbitFramePending || orbitRenderTimer) {
        return;
    }

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
    if (ws) {
        ws.close();
        ws = null;
    }
}

function resetPlots() {
    altData.t.length = 0;
    altData.y.length = 0;
    spdData.t.length = 0;
    spdData.y.length = 0;
    altChart.update("none");
    spdChart.update("none");
    phaseSeen = [];
    phaseStrip.innerHTML = "";
    phaseValue.textContent = "-";
    altValue.textContent = "-";
    spdValue.textContent = "-";

    orbitData.x.length = 0;
    orbitData.y.length = 0;
    orbitData.z.length = 0;
    orbitPlottedCount = 0;
    telemetryLogCount = 0;

    if (orbitEnabled && window.Plotly && orbitPlotEl) {
        window.Plotly.restyle(orbitPlotEl, { x: [[]], y: [[]], z: [[]] }, [1]);
        window.Plotly.restyle(orbitPlotEl, { x: [[]], y: [[]], z: [[]] }, [2]);
    }

    if (orbitRenderTimer) {
        clearTimeout(orbitRenderTimer);
        orbitRenderTimer = null;
    }

    orbitUserInteracting = false;
    if (orbitInteractionDebounce) {
        clearTimeout(orbitInteractionDebounce);
        orbitInteractionDebounce = null;
    }

    scheduleOrbitRender();
}

function upsertPhase(phase) {
    if (!phase) {
        return;
    }

    if (!phaseSeen.includes(phase)) {
        phaseSeen.push(phase);
        const chip = document.createElement("span");
        chip.className = "phase-chip";
        chip.dataset.phase = phase;
        chip.textContent = phase;
        phaseStrip.appendChild(chip);
    }

    for (const chip of phaseStrip.querySelectorAll(".phase-chip")) {
        chip.classList.toggle("active", chip.dataset.phase === phase);
    }

    phaseValue.textContent = phase;
}

function pushPoint(buf, t, y) {
    buf.t.push(t);
    buf.y.push(y);
    if (buf.t.length > MAX_POINTS) {
        buf.t.shift();
        buf.y.shift();
    }
}

function handleTelemetry(frame) {
    const t = Number(frame.time_s ?? 0);
    const pos = frame.position_m || [0, 0, 0];
    const vel = frame.velocity_ms || [0, 0, 0];

    const radius = Math.hypot(pos[0], pos[1], pos[2]);
    const speed = Math.hypot(vel[0], vel[1], vel[2]);
    const altitudeKm = (radius - EARTH_RADIUS_M) / 1000;
    const speedKms = speed / 1000;

    altValue.textContent = Number.isFinite(altitudeKm) ? altitudeKm.toFixed(2) : "-";
    spdValue.textContent = Number.isFinite(speedKms) ? speedKms.toFixed(3) : "-";
    upsertPhase(frame.phase);

    pushPoint(altData, t, altitudeKm);
    pushPoint(spdData, t, speedKms);

    orbitData.x.push(pos[0] / 1000);
    orbitData.y.push(pos[1] / 1000);
    orbitData.z.push(pos[2] / 1000);
    scheduleOrbitRender();

    altChart.update("none");
    spdChart.update("none");
}

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
                telemetryLogCount += 1;
                if (telemetryLogCount % TELEMETRY_LOG_EVERY_N === 0) {
                    appendLine("telemetry", {
                        seq: msg.data.seq,
                        time_s: msg.data.time_s,
                        phase: msg.data.phase,
                    });
                }
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

async function startAndConnect() {
    try {
        const telemetryInterval = Number(intervalInput.value || "0.5");
        const resp = await fetch(
            `/simulations/live/start?telemetry_interval=${telemetryInterval}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: "{}",
            }
        );

        if (!resp.ok) {
            throw new Error(`start failed: ${resp.status}`);
        }

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

document.getElementById("startBtn").addEventListener("click", startAndConnect);

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
    if (!orbitEnabled) {
        return;
    }

    bindOrbitResize();

    scheduleOrbitRender();
}

initOrbitSection();
