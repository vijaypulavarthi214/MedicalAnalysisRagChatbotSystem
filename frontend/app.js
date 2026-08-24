(function () {
  const API_BASE_URL = window.APP_CONFIG.API_BASE_URL;

  const fileInput = document.getElementById("file-input");
  const uploadStatus = document.getElementById("upload-status");
  const documentList = document.getElementById("document-list");
  const chatWindow = document.getElementById("chat-window");
  const chatForm = document.getElementById("chat-form");
  const questionInput = document.getElementById("question-input");
  const sendButton = document.getElementById("send-button");
  const errorBanner = document.getElementById("error-banner");

  let selectedDocumentId = localStorage.getItem("selectedDocumentId") || null;

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.remove("hidden");
  }

  function clearError() {
    errorBanner.classList.add("hidden");
    errorBanner.textContent = "";
  }

  function setChatEnabled(enabled) {
    questionInput.disabled = !enabled;
    sendButton.disabled = !enabled;
  }

  function appendMessage(role, text, sources) {
    const wrapper = document.createElement("div");
    wrapper.className = role === "user" ? "flex justify-end" : "flex justify-start";

    const bubble = document.createElement("div");
    bubble.className =
      role === "user"
        ? "bg-slate-800 text-white rounded-lg px-4 py-2 max-w-lg text-sm"
        : "bg-white border border-slate-200 rounded-lg px-4 py-2 max-w-lg text-sm";
    bubble.textContent = text;
    wrapper.appendChild(bubble);

    if (sources && sources.length > 0) {
      const citationBox = document.createElement("div");
      citationBox.className = "mt-2 flex flex-col gap-1";
      sources.forEach((source) => {
        const chip = document.createElement("details");
        chip.className = "bg-slate-100 rounded px-2 py-1 text-xs text-slate-600 cursor-pointer";
        const pageLabel =
          source.page_start === source.page_end
            ? `Page ${source.page_start}`
            : `Pages ${source.page_start}-${source.page_end}`;
        chip.innerHTML = `<summary>${source.section_title} · ${pageLabel} · relevance ${source.relevance_score.toFixed(2)}</summary>
          <p class="mt-1 text-slate-500">${source.excerpt}</p>`;
        citationBox.appendChild(chip);
      });
      bubble.appendChild(citationBox);
    }

    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  async function loadDocuments() {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`);
      if (!response.ok) throw new Error(`Failed to load documents (${response.status})`);
      const documents = await response.json();
      renderDocumentList(documents);
    } catch (err) {
      showError(err.message);
    }
  }

  function renderDocumentList(documents) {
    documentList.innerHTML = "";
    if (documents.length === 0) {
      documentList.innerHTML = '<li class="text-xs text-slate-400">No documents uploaded yet.</li>';
      return;
    }
    documents.forEach((doc) => {
      const item = document.createElement("li");
      const isSelected = doc.document_id === selectedDocumentId;
      item.className = `border rounded px-2 py-2 text-xs flex items-center justify-between gap-2 ${
        isSelected ? "border-slate-800 bg-slate-50" : "border-slate-200"
      }`;

      const label = document.createElement("button");
      label.type = "button";
      label.className = "text-left flex-1 truncate";
      label.innerHTML = `<span class="font-medium block truncate">${doc.filename}</span><span class="text-slate-400">${doc.chunk_count} chunks · ${doc.page_count} pages</span>`;
      label.addEventListener("click", () => selectDocument(doc.document_id));

      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.textContent = "✕";
      deleteBtn.className = "text-slate-400 hover:text-red-600 px-1";
      deleteBtn.addEventListener("click", () => deleteDocument(doc.document_id));

      item.appendChild(label);
      item.appendChild(deleteBtn);
      documentList.appendChild(item);
    });
  }

  function selectDocument(documentId) {
    selectedDocumentId = documentId;
    localStorage.setItem("selectedDocumentId", documentId);
    setChatEnabled(true);
    chatWindow.innerHTML = "";
    loadDocuments();
  }

  async function deleteDocument(documentId) {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, { method: "DELETE" });
      if (!response.ok) throw new Error(`Failed to delete document (${response.status})`);
      if (selectedDocumentId === documentId) {
        selectedDocumentId = null;
        localStorage.removeItem("selectedDocumentId");
        setChatEnabled(false);
        chatWindow.innerHTML = "";
      }
      await loadDocuments();
    } catch (err) {
      showError(err.message);
    }
  }

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;
    clearError();
    uploadStatus.textContent = "Uploading…";

    const formData = new FormData();
    formData.append("file", file);

    try {
      uploadStatus.textContent = "Processing (chunking, embedding)…";
      const response = await fetch(`${API_BASE_URL}/upload`, { method: "POST", body: formData });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Upload failed (${response.status})`);
      }
      const result = await response.json();
      uploadStatus.textContent = `Ready — ${result.chunk_count} chunks indexed.`;
      selectDocument(result.document_id);
    } catch (err) {
      uploadStatus.textContent = "";
      showError(err.message);
    } finally {
      fileInput.value = "";
    }
  });

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question || !selectedDocumentId) return;

    clearError();
    appendMessage("user", question);
    questionInput.value = "";
    setChatEnabled(false);

    const thinkingBubble = document.createElement("div");
    thinkingBubble.className = "flex justify-start";
    thinkingBubble.innerHTML = '<div class="bg-white border border-slate-200 rounded-lg px-4 py-2 text-sm text-slate-400">Thinking…</div>';
    chatWindow.appendChild(thinkingBubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, document_id: selectedDocumentId }),
      });
      thinkingBubble.remove();
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Query failed (${response.status})`);
      }
      const result = await response.json();
      appendMessage("assistant", result.answer, result.sources);
    } catch (err) {
      thinkingBubble.remove();
      showError(err.message);
    } finally {
      setChatEnabled(true);
      questionInput.focus();
    }
  });

  setChatEnabled(!!selectedDocumentId);
  loadDocuments();
})();
