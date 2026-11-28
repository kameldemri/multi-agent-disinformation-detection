const form = document.getElementById("analyze-form");
const textInput = document.getElementById("text");
const titleInput = document.getElementById("title");
const userInput = document.getElementById("user-id");
const prefixInput = document.getElementById("tweet-prefix");
const tweetsInput = document.getElementById("tweets");
const sampleButton = document.getElementById("sample-button");
const clearButton = document.getElementById("clear-button");
const submitButton = document.getElementById("submit-button");
const healthBadge = document.getElementById("health");
const llmBadge = document.getElementById("llm-badge");
const errorBox = document.getElementById("error");
const resultBox = document.getElementById("result");

const sampleText = "Breaking: officials secretly changed the election results overnight, and the mainstream media refuses to report the truth.";
const sampleTitle = "Election results changed overnight";
const sampleTimestamps = [
  "2026-11-20T10:30:00Z",
  "2026-11-20T10:35:00Z",
  "2026-11-20T11:00:00Z",
];

function makeElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setHealth(ok, text) {
  healthBadge.textContent = text;
  healthBadge.className = `health ${ok ? "ok" : "error"}`;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`status ${response.status}`);
    setHealth(true, "API online");
  } catch (error) {
    setHealth(false, "API offline");
  }
}

function parseTweets() {
  const lines = tweetsInput.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 3) {
    throw new Error("Enter at least three tweet timestamps, one per line.");
  }

  const invalid = lines.find((line) => Number.isNaN(Date.parse(line)));
  if (invalid) {
    throw new Error(`Invalid timestamp: ${invalid}`);
  }

  const prefix = prefixInput.value.trim() || "tweet";
  return lines.map((createdAt, index) => ({
    tweet_id: `${prefix}-${index + 1}`,
    created_at: createdAt,
  }));
}

function buildPayload() {
  const text = textInput.value.trim();
  const userId = userInput.value.trim();

  if (!text) throw new Error("Text is required.");
  if (!userId) throw new Error("User ID is required.");

  return {
    text,
    title: titleInput.value.trim(),
    account: {
      user_id: userId,
      tweets: parseTweets(),
    },
  };
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

function setBusy(busy) {
  submitButton.disabled = busy;
  submitButton.textContent = busy ? "Analyzing..." : "Analyze";
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function addMetric(container, label, value) {
  const metric = makeElement("div", "metric");
  metric.append(makeElement("span", "", label), makeElement("strong", "", value));
  container.append(metric);
}

function renderResult(payload) {
  const agent1 = payload.agent_1;
  const agent2 = payload.agent_2;
  const agent3 = payload.agent_3;

  resultBox.classList.remove("result-empty");
  resultBox.replaceChildren();

  const decision = makeElement("div", `decision ${agent3.decision.toLowerCase()}`, agent3.decision);
  resultBox.append(decision);

  const metrics = makeElement("div", "metrics");
  addMetric(metrics, "Final risk", formatPercent(agent3.final_risk_probability));
  addMetric(metrics, "Content risk", formatPercent(agent1.fake_news_probability));
  addMetric(metrics, "Behavior risk", formatPercent(agent2.bot_probability));
  addMetric(
    metrics,
    "Agent labels",
    `${agent1.predicted_label} / ${agent2.predicted_label}`,
  );
  resultBox.append(metrics);

  const reason = makeElement("div", "reason");
  reason.append(makeElement("h3", "", "Explanation"), makeElement("p", "", agent3.reason));
  resultBox.append(reason);

  const raw = makeElement("details", "raw");
  raw.append(makeElement("summary", "", "Raw response"));
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(payload, null, 2);
  raw.append(pre);
  resultBox.append(raw);

  llmBadge.textContent = `LLM: ${agent3.llm_status}`;
  llmBadge.className = `badge llm-${agent3.llm_status}`;
}

sampleButton.addEventListener("click", () => {
  textInput.value = sampleText;
  titleInput.value = sampleTitle;
  userInput.value = "account_001";
  prefixInput.value = "tweet";
  tweetsInput.value = sampleTimestamps.join("\n");
  clearError();
});

clearButton.addEventListener("click", () => {
  form.reset();
  userInput.value = "account_001";
  prefixInput.value = "tweet";
  resultBox.classList.add("result-empty");
  resultBox.textContent = "Submit evidence to see the agent outputs.";
  llmBadge.textContent = "Not run";
  llmBadge.className = "badge";
  clearError();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  try {
    const payload = buildPayload();
    setBusy(true);

    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail ?? data);
      throw new Error(detail || `Request failed with status ${response.status}`);
    }

    renderResult(data);
  } catch (error) {
    showError(error.message || String(error));
  } finally {
    setBusy(false);
  }
});

checkHealth();
