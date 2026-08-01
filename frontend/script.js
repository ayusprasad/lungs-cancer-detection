const CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n"];
const FEATURES = [
  ["GENDER", "Gender", "select", [[1, "Male"], [0, "Female"]]], ["AGE", "Age", "number"],
  ["SMOKING", "Smoking", "binary"], ["YELLOW_FINGERS", "Yellow fingers", "binary"], ["ANXIETY", "Anxiety", "binary"],
  ["PEER_PRESSURE", "Peer pressure", "binary"], ["CHRONIC DISEASE", "Chronic disease", "binary"], ["FATIGUE", "Fatigue", "binary"],
  ["ALLERGY", "Allergy", "binary"], ["WHEEZING", "Wheezing", "binary"], ["ALCOHOL CONSUMING", "Alcohol consuming", "binary"],
  ["COUGHING", "Coughing", "binary"], ["SHORTNESS OF BREATH", "Shortness of breath", "binary"], ["SWALLOWING DIFFICULTY", "Swallowing difficulty", "binary"], ["CHEST PAIN", "Chest pain", "binary"]
];

const $ = (id) => document.getElementById(id);
const api = async (url, options = {}) => {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({ ok: false, error: "Invalid server response" }));
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
};
const jsonOptions = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function showError(error) { window.alert(error.message || String(error)); }
function setButtonBusy(button, busy, busyText) { if (!button.dataset.label) button.dataset.label = button.textContent; button.disabled = busy; button.textContent = busy ? busyText : button.dataset.label; }
function metricMarkup(metrics) {
  if (!metrics) return "<span>Train this model to load metrics.</span>";
  const entries = Object.entries(metrics).filter(([key, value]) => ["accuracy", "precision", "recall", "f1", "loss", "epochs_ran", "sample_per_class", "sample_rows"].includes(key) && typeof value !== "object");
  return entries.map(([key, value]) => `<span>${key.replaceAll("_", " ")} <b>${typeof value === "number" && ["accuracy", "precision", "recall", "f1"].includes(key) ? `${(value * 100).toFixed(1)}%` : value}</b></span>`).join("");
}

function buildForm() {
  $("tabular-form").innerHTML = FEATURES.map(([key, label, type, options]) => {
    if (type === "number") return `<label>${label}<input name="${key}" type="number" min="18" max="100" value="60" required></label>`;
    const choices = options || [[1, "No"], [2, "Yes"]];
    return `<label>${label}<select name="${key}">${choices.map(([value, text]) => `<option value="${value}">${text}</option>`).join("")}</select></label>`;
  }).join("");
}
function readForm() { return Object.fromEntries([...$("tabular-form").elements].map((input) => [input.name, Number(input.value)])); }

async function refresh() {
  const [health, dataset, metrics] = await Promise.all([api("/api/health"), api("/api/dataset"), api("/api/metrics")]);
  $("status-dot").classList.add("live"); $("health-label").textContent = "Local API online"; $("health-detail").textContent = "Ready for testing";
  $("tabular-rows").textContent = dataset.data.tabular_rows; $("image-classes").textContent = Object.keys(dataset.data.image_classes).length;
  $("image-source-path").textContent = dataset.data.image_archive;
  $("image-ready").textContent = health.image_model_ready ? "Ready" : "Not trained";
  $("tabular-metrics").innerHTML = metricMarkup(metrics.tabular); $("image-metrics").innerHTML = metricMarkup(metrics.image);
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => { document.querySelectorAll(".tab, .panel").forEach((item) => item.classList.remove("active")); tab.classList.add("active"); $(tab.dataset.panel).classList.add("active"); }));
$("train-tabular").addEventListener("click", async (event) => { const button = event.currentTarget; setButtonBusy(button, true, "Training..."); try { await api("/api/train/tabular", jsonOptions({ sample_size: Number($("tabular-sample").value) })); await refresh(); } catch (error) { showError(error); } finally { setButtonBusy(button, false); } });
$("predict-tabular").addEventListener("click", async () => { try { const response = await api("/api/predict/tabular", jsonOptions(readForm())); const result = response.result; $("tabular-result").hidden = false; $("tabular-result").innerHTML = `<strong>${result.prediction}</strong><p>Probability of a positive survey label: ${(result.probability_yes * 100).toFixed(1)}%</p>`; } catch (error) { showError(error); } });
$("train-image").addEventListener("click", async (event) => { const button = event.currentTarget; setButtonBusy(button, true, "Training CNN..."); try { await api("/api/train/image", jsonOptions({ sample_per_class: Number($("image-sample").value), epochs: Number($("image-epochs").value), fine_tune: $("fine-tune").checked })); await refresh(); } catch (error) { showError(error); } finally { setButtonBusy(button, false); } });
$("image-file").addEventListener("change", (event) => { const file = event.target.files[0]; if (!file) return; const preview = $("preview"); preview.src = URL.createObjectURL(file); preview.hidden = false; document.querySelector(".upload-copy").hidden = true; });
$("predict-image").addEventListener("click", async () => { const file = $("image-file").files[0]; if (!file) return showError(new Error("Choose an image first.")); const form = new FormData(); form.append("image", file); try { const response = await api("/api/predict/image", { method: "POST", body: form }); const result = response.result; $("image-result").hidden = false; const bars = Object.entries(result.probabilities).map(([name, probability]) => `<div class="bar"><span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${probability * 100}%"></div></div><span>${(probability * 100).toFixed(1)}%</span></div>`).join(""); $("image-result").innerHTML = `<strong>${result.prediction}</strong><div>${bars}</div>`; } catch (error) { showError(error); } });

buildForm();
refresh().catch((error) => { $("health-label").textContent = "API unavailable"; $("health-detail").textContent = error.message; });
