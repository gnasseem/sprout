import { storyPrompt } from "./tokenizer.js";

const form = document.querySelector("#story-form");
const write = document.querySelector("#write");
const stop = document.querySelector("#stop");
const story = document.querySelector("#story");
const copy = document.querySelector("#copy");
const status = document.querySelector("#status");
const progress = document.querySelector("#progress");
const meta = document.querySelector("#story-meta");
let worker;
let loaded = false;
let running = false;

function message(text, error = false) {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function finish() {
  running = false;
  write.disabled = false;
  write.textContent = loaded ? "Write a story" : "Load model & write";
  stop.hidden = true;
  progress.hidden = true;
  form.querySelectorAll("input, textarea, select, [data-example]").forEach(input => input.disabled = false);
  story.removeAttribute("aria-busy");
}

function startWorker() {
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "status") message(data.text);
    if (data.type === "progress") {
      progress.hidden = false;
      progress.value = data.value;
      message(`Loading weights… ${Math.round(data.value * 100)}%`);
    }
    if (data.type === "loaded") {
      loaded = true;
      progress.hidden = true;
      document.querySelector("#download-note").textContent = "Model loaded. Stories are generated on your device; prompts are never sent to a server.";
    }
    if (data.type === "text") {
      story.textContent = data.text;
      copy.disabled = !data.text;
      meta.textContent = `${data.tokens} tokens written`;
    }
    if (data.type === "done") {
      meta.textContent = `${data.reason} · ${data.tokens} tokens${data.seconds ? ` · ${data.seconds.toFixed(1)}s` : ""}`;
      message(data.reason === "Stopped" ? "Stopped. You can write another story." : "Ready for another idea.");
      finish();
    }
    if (data.type === "error") {
      meta.textContent = "Could not write this story";
      message(data.text, true);
      finish();
    }
  };
  worker.onerror = () => {
    worker.terminate();
    worker = undefined;
    loaded = false;
    meta.textContent = "Model unavailable";
    message("The browser model could not start. Reload and try again, or use the local Python demo linked below.", true);
    finish();
  };
}

form.addEventListener("submit", event => {
  event.preventDefault();
  if (running) return;
  const fields = Object.fromEntries(new FormData(form));
  try { storyPrompt(fields); }
  catch (error) { message(error.message, true); return; }
  if (!worker) startWorker();
  running = true;
  write.disabled = true;
  write.textContent = loaded ? "Writing…" : "Loading…";
  stop.hidden = false;
  stop.disabled = false;
  stop.textContent = "Stop";
  copy.disabled = true;
  story.textContent = "";
  story.setAttribute("aria-busy", "true");
  meta.textContent = "Getting ready to write";
  form.querySelectorAll("input, textarea, select, [data-example]").forEach(input => input.disabled = true);
  message(loaded ? "Writing your story…" : "Loading the model for the first story…");
  worker.postMessage({ type: "generate", fields,
    temperature: Number(document.querySelector("#temperature").value),
    maxTokens: Number(document.querySelector("#length").value) });
});

stop.addEventListener("click", () => {
  if (!loaded) {
    worker.terminate();
    worker = undefined;
    message("Loading stopped. Click Load model & write to try again.");
    meta.textContent = "Loading stopped";
    finish();
  } else {
    stop.disabled = true;
    stop.textContent = "Stopping…";
    worker.postMessage({ type: "stop" });
  }
});

document.querySelectorAll("[data-example]").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelector("#summary").value = button.dataset.example;
    document.querySelector("#summary").focus();
  });
});
document.querySelector("#temperature").addEventListener("input", event => {
  document.querySelector("#temperature-value").value = Number(event.target.value).toFixed(1);
});
copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(story.textContent);
    message("Story copied.");
  } catch { message("Select the story text and copy it with your keyboard.", true); }
});
