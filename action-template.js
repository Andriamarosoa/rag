(() => {
  function templateLabel(action) {
    return String(action?.label || humanize(action?.agent || action?.type || 'action'));
  }

  function expandTemplateVariables(value, action) {
    const label = templateLabel(action);
    return String(value || '')
      .replace(/\{\{\s*label\s*\}\}/gi, label)
      .replace(/\{\{\s*labe\s*\}\}/gi, label);
  }

  function appendTemplateText(parent, value, action) {
    const text = expandTemplateVariables(value, action);
    if (text) parent.appendChild(document.createTextNode(text));
  }

  function createInlineActionLink(action, rawLabel) {
    const button = createActionButton(action);
    button.classList.add('action-template-link');
    button.textContent = expandTemplateVariables(rawLabel, action).trim() || templateLabel(action);
    return button;
  }

  function renderTemplateBody(parent, body, action) {
    const linkPattern = /<link\b[^>]*>([\s\S]*?)<\/link>/gi;
    let cursor = 0;
    let match;

    while ((match = linkPattern.exec(body)) !== null) {
      appendTemplateText(parent, body.slice(cursor, match.index), action);
      parent.appendChild(createInlineActionLink(action, match[1]));
      cursor = match.index + match[0].length;
    }

    appendTemplateText(parent, body.slice(cursor), action);
  }

  function createActionTemplate(action) {
    const template = String(action?.template || '').trim();
    const wrapper = document.createElement('div');
    wrapper.className = 'action-template-block';

    const paragraphPattern = /<p\b[^>]*>([\s\S]*?)<\/p>/gi;
    const paragraphs = [];
    let match;
    while ((match = paragraphPattern.exec(template)) !== null) {
      paragraphs.push(match[1]);
    }

    const bodies = paragraphs.length ? paragraphs : [template];
    bodies.forEach(body => {
      const paragraph = document.createElement('p');
      paragraph.className = 'action-template-paragraph';
      renderTemplateBody(paragraph, body, action);
      wrapper.appendChild(paragraph);
    });

    return wrapper;
  }

  function installActionTemplateStyles() {
    if (document.getElementById('actionTemplateStyles')) return;
    const style = document.createElement('style');
    style.id = 'actionTemplateStyles';
    style.textContent = `
      .action-template-block { width:100%; }
      .action-template-paragraph { margin:0 0 8px; line-height:1.6; }
      .action-template-paragraph:last-child { margin-bottom:0; }
      .action-template-link {
        display:inline;
        border:0;
        background:transparent;
        color:inherit;
        padding:0;
        margin:0 2px;
        border-radius:0;
        font:inherit;
        font-weight:650;
        text-decoration:underline;
        text-underline-offset:2px;
        cursor:pointer;
      }
      .action-template-link:hover { background:transparent; opacity:.82; }
    `;
    document.head.appendChild(style);
  }

  function completeAssistantWithTemplates(payload) {
    const d = payload.data || {};
    const text = String(d.answer ?? d.response ?? d.text ?? '');
    const view = ensureFlowView(payload.request_id);
    finishFlow(payload.request_id, false);
    view.content.textContent = text;
    view.actions.innerHTML = '';

    const actions = Array.isArray(d.actions) ? d.actions : [];
    const hasTemplate = actions.some(action => String(action?.template || '').trim());
    view.actions.style.display = hasTemplate ? 'block' : 'flex';

    actions.forEach(action => {
      const template = String(action?.template || '').trim();
      view.actions.appendChild(
        template ? createActionTemplate(action) : createActionButton(action)
      );
    });
    scrollChat();
  }

  installActionTemplateStyles();
  RESPONSE_HANDLERS.set('assistant.completed', completeAssistantWithTemplates);
})();
