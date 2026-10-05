const CHART_POINTS = 100;
const PPG_HISTORY_SIZE = 600;

// Fixed graph rendering speed.
const GRAPH_UPDATE_INTERVAL = 100;

let lastGraphUpdateTime = 0;
let latestECGValue = null;
let latestPPGValue = null;

// HRV becomes fixed after the final 30-second assessment.
let finalHRVLocked = false;

// --------------------------------------------------
// Chart helper
// --------------------------------------------------

function initChart(canvasId, color, label, yTitle) {
  const canvas = document.getElementById(canvasId);

  if (!canvas) {
    console.error("Canvas not found:", canvasId);
    return null;
  }

  const ctx = canvas.getContext("2d");

  return new Chart(ctx, {
    type: "line",

    data: {
      labels: Array.from(
        { length: CHART_POINTS },
        (_, i) => i
      ),

      datasets: [{
        label: label,
        data: Array(CHART_POINTS).fill(null),
        borderColor: color,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.05,
        fill: false
      }]
    },

    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,

      plugins: {
        legend: {
          display: false
        },

        tooltip: {
          enabled: true,
          mode: "index",
          intersect: false
        }
      },

      scales: {
        x: {
          display: true,

          title: {
            display: true,
            text: "Sample"
          },

          ticks: {
            display: true,
            maxTicksLimit: 10,
            color: "#9bb7c4"
          },

          grid: {
            color: "rgba(111, 202, 205, 0.08)"
          }
        },

        y: {
          display: true,

          title: {
            display: true,
            text: yTitle
          },

          ticks: {
            display: true,
            color: "#9bb7c4",
            callback: function(value) {
              return Number(value).toFixed(0);
            }
          },

          grid: {
            color: "rgba(111, 202, 205, 0.08)"
          }
        }
      }
    }
  });
}


// --------------------------------------------------
// Utility functions
// --------------------------------------------------

function mean(values) {
  if (!values.length) return 0;

  return values.reduce((a, b) => a + b, 0) / values.length;
}


function standardDeviation(values) {
  if (values.length < 2) return 0;

  const avg = mean(values);

  const variance =
    values.reduce(
      (sum, value) => sum + Math.pow(value - avg, 2),
      0
    ) / values.length;

  return Math.sqrt(variance);
}


function median(values) {
  if (!values.length) return 0;

  const sorted = [...values].sort((a, b) => a - b);

  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }

  return sorted[middle];
}


// --------------------------------------------------
// PPG peak detection
// PPG-derived RR intervals
// --------------------------------------------------

function calculatePPGPeaks(ppgSamples) {
  if (ppgSamples.length < 30) {
    return [];
  }

  const values = ppgSamples.map(x => x.value);

  const avg = mean(values);
  const std = standardDeviation(values);
  const med = median(values);

  if (std < 0.5) {
    return [];
  }

  /*
     Dynamic threshold.

     This is an engineering peak detector for the
     prototype. It is NOT a clinical algorithm.
  */
  const threshold = med + (0.20 * std);

  const peaks = [];

  // Minimum separation between heart beats.
  // 300 ms ~= maximum 200 BPM.
  const MIN_DISTANCE_MS = 300;

  let lastPeakTime = -Infinity;

  for (let i = 1; i < ppgSamples.length - 1; i++) {

    const previous = ppgSamples[i - 1].value;
    const current = ppgSamples[i].value;
    const next = ppgSamples[i + 1].value;

    if (
      current > previous &&
      current >= next &&
      current > threshold
    ) {

      const currentTime = ppgSamples[i].time;

      if (
        currentTime - lastPeakTime >= MIN_DISTANCE_MS
      ) {
        peaks.push(ppgSamples[i]);
        lastPeakTime = currentTime;
      }
    }
  }

  return peaks;
}


function calculateHRV(ppgSamples) {

  const peaks = calculatePPGPeaks(ppgSamples);

  if (peaks.length < 3) {
    return {
      meanRR: 0,
      sdnn: 0,
      rmssd: 0,
      peaks: peaks
    };
  }

  const rrIntervals = [];

  for (let i = 1; i < peaks.length; i++) {

    const rr =
      peaks[i].time -
      peaks[i - 1].time;

    // Accept approximately 30–200 BPM.
    if (rr >= 300 && rr <= 2000) {
      rrIntervals.push(rr);
    }
  }

  if (rrIntervals.length < 2) {
    return {
      meanRR: 0,
      sdnn: 0,
      rmssd: 0,
      peaks: peaks
    };
  }

  const avgRR = mean(rrIntervals);

  const sdnn = standardDeviation(rrIntervals);

  const differences = [];

  for (let i = 1; i < rrIntervals.length; i++) {
    differences.push(
      rrIntervals[i] - rrIntervals[i - 1]
    );
  }

  let rmssd = 0;

  if (differences.length > 0) {

    const squared =
      differences.map(x => x * x);

    rmssd =
      Math.sqrt(mean(squared));
  }

  return {
    meanRR: avgRR,
    sdnn: sdnn,
    rmssd: rmssd,
    peaks: peaks
  };
}


// --------------------------------------------------
// Engineering signal quality
// --------------------------------------------------

function calculateSignalQuality(
  ppgValues,
  leadOff,
  spo2
) {

  if (leadOff !== 0) {
    return 0;
  }

  if (ppgValues.length < 20) {
    return 0;
  }

  const std = standardDeviation(ppgValues);

  const min = Math.min(...ppgValues);
  const max = Math.max(...ppgValues);

  const amplitude = max - min;

  if (!Number.isFinite(std) || !Number.isFinite(amplitude)) {
    return 0;
  }

  /*
     Engineering-quality estimate.

     It considers:
     - PPG amplitude
     - variation
     - valid SpO2
     - lead status

     This is NOT a medical signal-quality index.
  */

  let score = 0;

  if (amplitude > 5) score += 25;
  if (amplitude > 20) score += 15;
  if (amplitude > 50) score += 15;
  if (amplitude > 100) score += 10;

  if (std > 2) score += 10;
  if (std > 5) score += 10;
  if (std > 10) score += 5;

  if (
    Number.isFinite(spo2) &&
    spo2 > 70 &&
    spo2 <= 100
  ) {
    score += 10;
  }

  return Math.max(0, Math.min(100, score));
}


// --------------------------------------------------
// Update graph
// --------------------------------------------------

function updateChart(chart, values) {

  if (!chart || values.length < 2) {
    return;
  }

  const visibleValues =
    values.slice(-CHART_POINTS);

  const startIndex =
    Math.max(0, values.length - CHART_POINTS);

  const labels =
    visibleValues.map(
      (_, index) => startIndex + index
    );

  const minValue =
    Math.min(...visibleValues);

  const maxValue =
    Math.max(...visibleValues);

  const range =
    Math.max(maxValue - minValue, 10);

  const margin =
    range * 0.15;

  chart.data.labels = labels;

  chart.data.datasets[0].data =
    visibleValues;

  chart.options.scales.y.min =
    minValue - margin;

  chart.options.scales.y.max =
    maxValue + margin;

  chart.update("none");
}


// --------------------------------------------------
// Main
// --------------------------------------------------

document.addEventListener(
  "DOMContentLoaded",
  () => {

    const ecgChart =
      initChart(
        "ecgChart",
        "#63d8df",
        "ECG",
        "ECG Value"
      );

    const ppgChart =
      initChart(
        "ppgChart",
        "#66a9ef",
        "PPG IR",
        "IR Value"
      );


    // -----------------------------
    // Dashboard elements
    // -----------------------------

    const hrElement =
      document.getElementById("hrValue");

    const spo2Element =
      document.getElementById("spo2Value");

    const rmssdVitalElement =
      document.getElementById("feat-rmssd-vital");

    const rrElement =
      document.getElementById("feat-rr");

    const sdnnElement =
      document.getElementById("feat-sdnn");

    const rmssdElement =
      document.getElementById("feat-rmssd");

    const p2pElement =
      document.getElementById("feat-p2p");

    const qualityElement =
      document.querySelector(".quality-value");

    const qualityStatusElement =
      document.querySelector(
        ".quality-value + .vital-status"
      );

    const qualityBars =
      document.querySelectorAll(".quality-fill");

    const lastUpdate =
      document.getElementById("lastUpdate");


    // -----------------------------
    // Data buffers
    // -----------------------------

    const ecgValues = [];

    const ppgValues = [];

    const ppgHistory = [];

    let sampleCounter = 0;


    // -----------------------------
    // Socket
    // -----------------------------

    const socket = io();


    socket.on("connect", () => {

      console.log(
        "Dashboard connected"
      );

    });


    socket.on("disconnect", () => {

      console.log(
        "Dashboard disconnected"
      );

    });

    // ==================================================
// Cardiovascular Health
// ==================================================

socket.on(
  "health_prediction",
  (health) => {

    if (!health) {
      return;
    }

    // ------------------------------------------------
    // Main cardiovascular health status
    // ------------------------------------------------

    const mlStatus =
      document.getElementById(
        "mlStatus"
      );

    if (mlStatus) {
      mlStatus.textContent =
        health.status ||
        "Waiting for data";
    }


    // ------------------------------------------------
    // Overall health score
    // ------------------------------------------------

    const healthScore =
      Number(health.score);

    const confidenceValue =
      document.getElementById(
        "confidenceValue"
      );

    if (Number.isFinite(healthScore)) {

      if (confidenceValue) {
        confidenceValue.textContent =
          healthScore.toFixed(1) +
          "/100";
      }
    }


    // ------------------------------------------------
    // Existing ring is now an overall health-score ring
    // ------------------------------------------------

    const confidenceRing =
      document.getElementById(
        "confidenceRing"
      );

    if (
      confidenceRing &&
      Number.isFinite(healthScore)
    ) {

      const degrees =
        Math.max(
          0,
          Math.min(
            360,
            healthScore * 3.6
          )
        );

      confidenceRing.style.setProperty(
        "--confidence",
        degrees + "deg"
      );
    }


    // ------------------------------------------------
    // Individual component scores
    // ------------------------------------------------

    const components =
      health.components ||
      {};

    const hrScore =
      Number(
        components.heart_rate
      );

    const spo2Score =
      Number(
        components.spo2
      );

    const hrvScore =
      Number(
        components.hrv
      );

    const signalScore =
      Number(
        components.signal_quality
      );


    // ------------------------------------------------
    // Actual final physiological values
    // ------------------------------------------------
    // Keep the overall Health Score as /100, but do NOT
    // show component values as percentages. The dashboard
    // should show the measured/averaged physiological values.

    const averages = health.averages || {};

    const finalHR = Number(averages.heart_rate);
    const finalSpO2 = Number(averages.spo2);
    const finalRMSSDForCard = Number(averages.rmssd);
    const finalSignalQuality = Number(averages.signal_quality);

    const hrHealthScore =
      document.getElementById(
        "hrHealthScore"
      );

    if (
      hrHealthScore &&
      Number.isFinite(finalHR) &&
      finalHR > 0
    ) {
      hrHealthScore.textContent =
        finalHR.toFixed(1) +
        " BPM";
    }


    const spo2HealthScore =
      document.getElementById(
        "spo2HealthScore"
      );

    if (
      spo2HealthScore &&
      Number.isFinite(finalSpO2) &&
      finalSpO2 > 0
    ) {
      spo2HealthScore.textContent =
        finalSpO2.toFixed(1) +
        "%";
    }


    const hrvHealthScore =
      document.getElementById(
        "hrvHealthScore"
      );

    if (
      hrvHealthScore &&
      Number.isFinite(finalRMSSDForCard) &&
      finalRMSSDForCard > 0
    ) {
      hrvHealthScore.textContent =
        finalRMSSDForCard.toFixed(1) +
        " ms";
    }


    const signalHealthScore =
      document.getElementById(
        "signalHealthScore"
      );

    if (
      signalHealthScore &&
      Number.isFinite(finalSignalQuality) &&
      finalSignalQuality >= 0
    ) {
      signalHealthScore.textContent =
        finalSignalQuality.toFixed(0) +
        "%";
    }


    // ------------------------------------------------
    // Lock final averaged HRV values
    // ------------------------------------------------

    if (health.locked) {

      finalHRVLocked = true;

      const finalRMSSD = Number(averages.rmssd);
      const finalSDNN = Number(averages.sdnn);
      const finalRR = Number(
        averages.mean_rr ||
        (
          Number(averages.heart_rate) > 0
            ? 60000 / Number(averages.heart_rate)
            : 0
        )
      );

      if (rrElement && Number.isFinite(finalRR) && finalRR > 0) {
        rrElement.textContent = finalRR.toFixed(0) + " ms";
      }

      if (sdnnElement && Number.isFinite(finalSDNN) && finalSDNN > 0) {
        sdnnElement.textContent = finalSDNN.toFixed(1) + " ms";
      }

      if (rmssdElement && Number.isFinite(finalRMSSD) && finalRMSSD > 0) {
        rmssdElement.textContent = finalRMSSD.toFixed(1) + " ms";
      }

      if (rmssdVitalElement && Number.isFinite(finalRMSSD) && finalRMSSD > 0) {
        rmssdVitalElement.textContent = finalRMSSD.toFixed(1);
      }

      console.log("Final HRV locked:", finalRMSSD, "ms");
    }


    // ------------------------------------------------
    // Debug
    // ------------------------------------------------

    console.log(
      "Cardiovascular Health:",
      health.status,
      "| Score:",
      health.score + "/100",
      "| HR:",
      hrScore,
      "| SpO2:",
      spo2Score,
      "| HRV:",
      hrvScore,
      "| Signal:",
      signalScore
    );
  }
);

    // -----------------------------
    // Sensor data
    // -----------------------------

    socket.on(
      "sensor_data",
      (data) => {

        if (!data) return;


        // ==================================================
        // ECG
        // ==================================================

        const ecg =
          Number(data.ecg_value);

        if (Number.isFinite(ecg)) {
          latestECGValue = ecg;
        }


        // ==================================================
        // PPG IR
        // ==================================================

        const ppg =
          Number(data.ppg_ir);


        if (
          Number.isFinite(ppg) &&
          ppg > 0
        ) {

          sampleCounter++;

          latestPPGValue = ppg;

          // Keep raw history for calculations.
          ppgHistory.push({
            value: ppg,
            time: performance.now()
          });

          if (
            ppgHistory.length >
            PPG_HISTORY_SIZE
          ) {
            ppgHistory.shift();
          }
        }


        // ==================================================
        // Fixed-speed graph rendering
        // ==================================================

        const graphNow = performance.now();

        if (graphNow - lastGraphUpdateTime >= GRAPH_UPDATE_INTERVAL) {

          lastGraphUpdateTime = graphNow;

          if (Number.isFinite(latestECGValue)) {

            ecgValues.push(latestECGValue);

            if (ecgValues.length > CHART_POINTS) {
              ecgValues.shift();
            }

            updateChart(ecgChart, ecgValues);
          }

          if (Number.isFinite(latestPPGValue)) {

            ppgValues.push(latestPPGValue);

            if (ppgValues.length > CHART_POINTS) {
              ppgValues.shift();
            }

            updateChart(ppgChart, ppgValues);
          }
        }


        // ==================================================
        // Vitals
        // ==================================================

        let hr = 0;
        let spo2 = 0;

        if (data.vitals) {

          hr =
            Number(data.vitals.hr);

          spo2 =
            Number(data.vitals.spo2);


          if (
            Number.isFinite(hr) &&
            hr > 0
          ) {

            if (hrElement) {
              hrElement.textContent =
                hr.toFixed(1);
            }
          }


          if (
            Number.isFinite(spo2) &&
            spo2 > 0 &&
            spo2 <= 100
          ) {

            if (spo2Element) {
              spo2Element.textContent =
                spo2.toFixed(1);
            }
          }
        }


        // ==================================================
        // Lead status
        // ==================================================

        const leadOff =
          Number(data.lead_off);


        // ==================================================
        // PPG-derived HRV
        // ==================================================
        // Keep HRV stable during collection. The final
        // averaged value is supplied by health_prediction.

        if (!finalHRVLocked) {

          if (rrElement) {
            rrElement.textContent = "-- ms";
          }

          if (sdnnElement) {
            sdnnElement.textContent = "-- ms";
          }

          if (rmssdElement) {
            rmssdElement.textContent = "-- ms";
          }

          if (rmssdVitalElement) {
            rmssdVitalElement.textContent = "--";
          }
        }


        // PPG peak-to-peak
        // ==================================================

        if (
          ppgHistory.length >= 20
        ) {

          const recentPPG =
            ppgValues
              .slice(-100);

          const p2p =
            Math.max(...recentPPG) -
            Math.min(...recentPPG);


          if (p2pElement) {

            p2pElement.textContent =
              p2p.toFixed(1);
          }
        }


        // ==================================================
        // Signal quality
        // ==================================================

        const recentPPG =
          ppgValues
            .slice(-100);

        const quality =
          calculateSignalQuality(
            recentPPG,
            leadOff,
            spo2
          );


        if (qualityElement) {

          qualityElement.textContent =
            quality.toFixed(0) + "%";
        }


        if (qualityStatusElement) {

          const statusBar =
            qualityStatusElement.querySelector(
              ".status-bar"
            );

          if (statusBar) {

            statusBar.classList.remove(
              "pending"
            );

            statusBar.classList.add(
              quality >= 70
                ? "good"
                : "pending"
            );
          }


          if (leadOff !== 0) {

            qualityStatusElement.innerHTML =
              '<span class="status-bar pending"></span> Lead Off';

          } else if (quality >= 70) {

            qualityStatusElement.innerHTML =
              '<span class="status-bar good"></span> Good signal';

          } else if (quality >= 40) {

            qualityStatusElement.innerHTML =
              '<span class="status-bar pending"></span> Moderate signal';

          } else {

            qualityStatusElement.innerHTML =
              '<span class="status-bar pending"></span> Weak signal';
          }
        }


        // ==================================================
        // Telemetry quality bars
        // ==================================================

        if (qualityBars.length >= 3) {

          const ecgQuality =
            leadOff === 0
              ? Math.max(30, quality)
              : 0;

          const ppgQuality =
            quality;

          const streamQuality =
            100;


          qualityBars[0].style.width =
            ecgQuality + "%";

          qualityBars[1].style.width =
            ppgQuality + "%";

          qualityBars[2].style.width =
            streamQuality + "%";
        }


        // ==================================================
        // Last update
        // ==================================================

        if (lastUpdate) {

          lastUpdate.textContent =
            new Date().toLocaleTimeString(
              [],
              {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
              }
            );
        }


        // ==================================================
        // Debug
        // ==================================================

        if (sampleCounter % 50 === 0) {

          console.log(
            "ECG:",
            ecg,
            "| PPG IR:",
            ppg,
            "| HR:",
            hr,
            "| SpO2:",
            spo2,
            "| HRV RMSSD:",
            rmssdVitalElement
              ? rmssdVitalElement.textContent
              : "--",
            "| Quality:",
            quality.toFixed(0) + "%"
          );
        }

      }
    );

  }
);