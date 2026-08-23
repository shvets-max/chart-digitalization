/* Reviewer page for eval/ground_truth/staging/ entries (see
   docs/accuracy-monitoring-design.md §5): approve promotes a chart into the
   canonical dataset (eval.promote.approve), reject discards it. Independent
   of app.js -- this page only lists/approves/rejects, it never re-extracts
   or edits a chart. */

const el = (id) => document.getElementById(id);

function setStatus(message, kind = "") {
  const status = el("status");
  status.textContent = message;
  status.className = `status${kind ? ` is-${kind}` : ""}`;
}

// Every field below can hold user-typed text (category/notes from the "Save
// to test set" form), so it's rendered as text, not raw HTML.
function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function formatEntry(entry) {
  const added = entry.added_at ? new Date(entry.added_at).toLocaleString() : "unknown time";
  const fraction =
    entry.correction_fraction === null || entry.correction_fraction === undefined
      ? "n/a"
      : `${Math.round(entry.correction_fraction * 100)}%`;
  return { added, fraction };
}

function renderEntries(entries) {
  const list = el("staged-list");
  list.innerHTML = "";
  if (!entries.length) {
    list.innerHTML = '<p class="empty-note">No staged entries.</p>';
    return;
  }
  for (const entry of entries) {
    const { added, fraction } = formatEntry(entry);
    const id = escapeHtml(entry.id);
    const card = document.createElement("article");
    card.className = "staged-card";
    card.innerHTML = `
      <img class="staged-thumb" src="/api/testset/staged/${id}/image" alt="">
      <div class="staged-body">
        <p class="staged-title">${id}</p>
        <p class="staged-meta">
          category: <strong>${escapeHtml(entry.category)}</strong> &middot;
          source: ${escapeHtml(entry.source)} &middot;
          series: ${escapeHtml(entry.n_series)} &middot;
          corrected: ${fraction} &middot;
          added ${escapeHtml(added)}
          ${entry.annotator ? `&middot; by ${escapeHtml(entry.annotator)}` : ""}
        </p>
        ${entry.notes ? `<p class="staged-notes">&ldquo;${escapeHtml(entry.notes)}&rdquo;</p>` : ""}
        <div class="staged-actions">
          <button type="button" class="button button-primary" data-action="approve" data-id="${id}">Approve</button>
          <button type="button" class="button" data-action="reject" data-id="${id}">Reject</button>
        </div>
      </div>
    `;
    list.appendChild(card);
  }
}

async function loadStaged() {
  setStatus("Loading…", "busy");
  try {
    const response = await fetch("/api/testset/staged");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    renderEntries(payload.entries);
    setStatus(payload.entries.length ? "" : "No staged entries.");
  } catch (error) {
    setStatus(error.message || "Could not load staged entries.", "error");
  }
}

async function approveEntry(id) {
  setStatus(`Approving ${id}…`, "busy");
  try {
    const response = await fetch(`/api/testset/staged/${id}/approve`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    setStatus(`Approved ${id}.`);
    await loadStaged();
  } catch (error) {
    setStatus(error.message || "Approving failed.", "error");
  }
}

async function rejectEntry(id) {
  if (!window.confirm(`Reject and permanently discard "${id}"?`)) return;
  setStatus(`Rejecting ${id}…`, "busy");
  try {
    const response = await fetch(`/api/testset/staged/${id}`, { method: "DELETE" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    setStatus(`Rejected ${id}.`);
    await loadStaged();
  } catch (error) {
    setStatus(error.message || "Rejecting failed.", "error");
  }
}

el("staged-list").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const { action, id } = button.dataset;
  if (action === "approve") approveEntry(id);
  else if (action === "reject") rejectEntry(id);
});
el("refresh-button").addEventListener("click", loadStaged);

loadStaged();
