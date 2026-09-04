from html.parser import HTMLParser
from pathlib import Path
import re


INDEX_HTML = Path(__file__).resolve().parents[1] / "index.html"
ACTION_TEMPLATE_JS = Path(__file__).resolve().parents[1] / "action-template.js"
MAIN_PY = Path(__file__).resolve().parents[1] / "app" / "main.py"


class _ModuleSelectParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_module_select = False
        self.current_option: dict[str, str | bool] | None = None
        self.options: list[dict[str, str | bool]] = []
        self.module_label_for: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "label" and attributes.get("for") == "moduleSelect":
            self.module_label_for = "moduleSelect"
        if tag == "select" and attributes.get("id") == "moduleSelect":
            self.in_module_select = True
        elif tag == "option" and self.in_module_select:
            self.current_option = {
                "value": attributes.get("value") or "",
                "selected": "selected" in attributes,
                "label": "",
            }

    def handle_data(self, data: str) -> None:
        if self.current_option is not None:
            self.current_option["label"] = str(self.current_option["label"]) + data

    def handle_endtag(self, tag: str) -> None:
        if tag == "option" and self.current_option is not None:
            self.current_option["label"] = str(self.current_option["label"]).strip()
            self.options.append(self.current_option)
            self.current_option = None
        elif tag == "select" and self.in_module_select:
            self.in_module_select = False


def test_module_selector_is_accessible_and_has_expected_namespaces():
    parser = _ModuleSelectParser()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))

    assert parser.module_label_for == "moduleSelect"
    assert parser.options == [
        {"value": "", "selected": True, "label": "Tous les modules"},
        {"value": "spay", "selected": False, "label": "Payroll"},
        {"value": "shrm", "selected": False, "label": "Human Resources"},
        {"value": "sgc", "selected": False, "label": "Gestion Commerciale"},
        {"value": "sacc", "selected": False, "label": "Accounting"},
        {"value": "sfar", "selected": False, "label": "Fixed Asset"},
        {"value": "sef", "selected": False, "label": "Equipment Follow-up"},
        {"value": "sim", "selected": False, "label": "Incident Management"},
        {"value": "seam", "selected": False, "label": "SEAM"},
        {"value": "pms", "selected": False, "label": "PMS"},
        {"value": "sess", "selected": False, "label": "SESS"},
    ]


def test_chat_message_includes_nullable_module_and_reset_preserves_selection():
    html = INDEX_HTML.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", html)

    assert "namespace:namespace||null" in compact
    assert "envelope('chat.message',{text,module:moduleScope.namespace})" in compact
    assert "addUserMessage(text,moduleScope.label)" in compact

    reset = re.search(r"function\s+resetConversation\(\)\s*\{(?P<body>.*?)\n\s*\}", html, re.S)
    assert reset is not None
    assert "moduleSelect" not in reset.group("body")


def test_refresh_restores_and_persists_module_and_chat_id_with_guarded_storage():
    html = INDEX_HTML.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", html)

    # Browser storage can be unavailable (for example in a restricted/private
    # context), so every read/write/remove operation must remain guarded.
    assert (
        "functionreadStoredValue(key){"
        "try{returnlocalStorage.getItem(key)||'';}"
        "catch{return'';}"
        "}" in compact
    )
    assert (
        "functionwriteStoredValue(key,value){"
        "try{"
        "if(value)localStorage.setItem(key,value);"
        "elselocalStorage.removeItem(key);"
        "}catch{}"
        "}" in compact
    )

    assert "constsavedModule=readStoredValue(STORAGE_KEYS.module)" in compact
    assert "option.value===savedModule" in compact
    assert "if(validModule)moduleSelect.value=savedModule" in compact
    assert "$('chatId').value=readStoredValue(STORAGE_KEYS.chatId)" in compact

    assert (
        "moduleSelect.addEventListener('change',()=>{"
        "writeStoredValue(STORAGE_KEYS.module,moduleSelect.value);"
        "})" in compact
    )
    assert (
        "$('chatId').value=p.chat_id;"
        "writeStoredValue(STORAGE_KEYS.chatId,p.chat_id);"
        "syncSession()" in compact
    )

    # Restoration has to happen during startup, before the refreshed client
    # reconnects and starts creating envelopes with the persisted chat id.
    restore_call = compact.rfind("restorePersistentState();")
    connect_call = compact.rfind("connect();")
    assert restore_call != -1
    assert connect_call != -1
    assert restore_call < connect_call


def test_new_conversation_forgets_persisted_chat_but_keeps_module_selection():
    html = INDEX_HTML.read_text(encoding="utf-8")
    reset = re.search(r"function\s+resetConversation\(\)\s*\{(?P<body>.*?)\n\s*\}", html, re.S)

    assert reset is not None
    body = re.sub(r"\s+", "", reset.group("body"))
    assert "$('chatId').value=''" in body
    assert "writeStoredValue(STORAGE_KEYS.chatId,'')" in body
    assert "STORAGE_KEYS.module" not in body
    assert "moduleSelect" not in body


def test_select_module_action_validates_namespace_and_continues_same_request():
    html = INDEX_HTML.read_text(encoding="utf-8")
    handler = re.search(
        r"registerActionHandler\(\s*['\"]select_module['\"]\s*,\s*action\s*=>\s*\{"
        r"(?P<body>.*?)\n\s*\}\s*\);",
        html,
        re.S,
    )

    assert handler is not None
    body = re.sub(r"\s+", "", handler.group("body"))

    # Never accept an arbitrary namespace supplied by an action: it must match
    # one of the values already declared by the trusted module selector.
    assert "action.module" in body
    assert "moduleSelect.options" in body
    assert "option.value===namespace" in body

    validation_position = body.find("if(!validModule||!question)")
    select_position = body.find("moduleSelect.value=namespace")
    persist_position = body.find("writeStoredValue(STORAGE_KEYS.module,namespace)")
    continuation_position = body.find("envelope('chat.continue',{question,module:namespace})")
    send_position = body.rfind("send(payload)")

    assert validation_position != -1
    assert select_position != -1
    assert persist_position != -1
    assert continuation_position != -1
    assert send_position != -1
    assert "input.value=question" not in body
    assert "sendMessage()" not in body
    assert validation_position < select_position < persist_position < continuation_position < send_position


def test_assistant_sources_use_safe_dom_links_without_rendering_model_html():
    html = INDEX_HTML.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", html)

    assert "view.content.textContent=text" in compact
    assert "renderSources(view.sources,d.sources,actions)" in compact
    assert "safeHttpUrl(source.url)" in compact
    assert "consturl=newURL(value,`${apiOrigin}/`)" in compact
    assert "url.protocol==='http:'||url.protocol==='https:'" in compact
    assert "url.hostname.toLowerCase()==='skb.uniconsults.mu'" in compact
    assert "constsafeFile=url.origin===apiOrigin" in compact
    assert "if(!href)return" in compact
    assert "visitLink.href=href" in compact
    assert "visitLink.href=source.url" not in compact
    assert "visitLink.target='_blank'" in compact
    assert "visitLink.rel='noopener'" in compact
    assert "title.textContent=String(source.title" in compact
    assert "meta.textContent=metadata.join('·')" in compact


def test_source_card_contains_continue_and_visit_controls_without_being_a_link():
    html = INDEX_HTML.read_text(encoding="utf-8")

    card = re.search(
        r"const\s+(?P<name>\w+)\s*=\s*document\.createElement\(['\"]div['\"]\);"
        r"\s*(?P=name)\.className\s*=\s*['\"]message-source['\"]",
        html,
    )
    visit_link = re.search(
        r"const\s+(?P<name>\w+)\s*=\s*document\.createElement\(['\"]a['\"]\);"
        r"\s*(?P=name)\.className\s*=\s*['\"]message-source-button message-source-visit['\"]",
        html,
    )

    assert card is not None
    assert visit_link is not None
    assert card.group("name") != visit_link.group("name")

    compact = re.sub(r"\s+", "", html)
    assert ".message-source-controls{display:flex" in compact
    assert "continueButton.textContent='Continuer'" in compact
    assert "controls.appendChild(continueButton)" in compact
    assert "visitLink.textContent=isFile?'Ouvrirlefichier↗':'Visiterlelien↗'" in compact
    assert "controls.appendChild(visitLink)" in compact
    assert "card.appendChild(controls)" in compact
    assert "visitLink.target='_blank'" in compact
    assert "visitLink.rel='noopener'" in compact
    assert "visitLink.className='message-source'" not in compact


def test_docx_upload_is_hidden_in_developer_mode_with_its_own_module_selector():
    html = INDEX_HTML.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", html)
    composer = html[html.index('<div id="composerWrap"'):html.index("</main>")]
    developer = html[html.index('<div id="developerPanel"'):]

    assert 'id="uploadDocxInput"' not in composer
    assert 'id="uploadDocxInput"' in developer
    assert 'id="developerModuleSelect"' in developer
    assert 'id="uploadDocxInput"' in compact
    assert 'accept=".docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"' in compact
    assert "...[...moduleSelect.options].map(option=>option.cloneNode(true))" in compact
    assert "if(!developerModuleSelect.value)" in compact
    assert "constnamespace=developerModuleSelect.value" in compact
    assert "form.append('module',namespace)" in compact
    assert "form.append('file',file,file.name)" in compact
    assert "fetch(apiUrl('/knowledge/files'),{method:'POST',body:form})" in compact
    assert "/^\\/knowledge\\/files\\/" in compact


def test_http_server_uses_backend_port_for_http_and_websocket_requests():
    html = INDEX_HTML.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", html)

    assert "constBACKEND_PORT='8765'" in compact
    assert "constHTTP_SERVER_PORT='8080'" in compact
    assert "if(location.protocol!=='http:'&&location.protocol!=='https:')" in compact
    assert "return`http://localhost:${BACKEND_PORT}`" in compact
    assert "if(location.port!==HTTP_SERVER_PORT)returnlocation.origin" in compact
    assert "constbackendUrl=newURL(location.origin)" in compact
    assert "backendUrl.port=BACKEND_PORT" in compact
    assert "constapiOrigin=resolveBackendOrigin()" in compact
    assert "functionapiUrl(path){returnnewURL(path,`${apiOrigin}/`).toString();}" in compact
    assert "constwebsocketUrl=newURL('/ws',`${apiOrigin}/`)" in compact
    assert "websocketUrl.protocol=websocketUrl.protocol==='https:'?'wss:':'ws:'" in compact
    assert "constdefaultWs=websocketUrl.toString()" in compact


def test_select_module_action_is_matched_to_its_source_and_not_rendered_separately():
    html = re.sub(r"\s+", "", INDEX_HTML.read_text(encoding="utf-8"))
    template = re.sub(r"\s+", "", ACTION_TEMPLATE_JS.read_text(encoding="utf-8"))

    assert "source?.page_id" in html
    assert "action?.type!=='select_module'" in html
    assert "pageNamespace===actionModule" in html
    assert "normalizedModuleValue(source?.module)" not in html
    assert "embeddedActions.add(sourceAction)" in html
    assert "constremainingActions=actions.filter(action=>!embeddedActions.has(action))" in html
    assert "filter(action=>!embeddedActions.has(action))" in template
    assert "(d.actions||[]).forEach(action=>view.actions.appendChild" not in html


def test_production_action_template_handler_keeps_source_rendering():
    script = re.sub(r"\s+", "", ACTION_TEMPLATE_JS.read_text(encoding="utf-8"))

    assert "RESPONSE_HANDLERS.set('assistant.completed',completeAssistantWithTemplates)" in script
    assert "renderSources(view.sources,d.sources,actions)" in script
    assert "constremainingActions=actions.filter(action=>!embeddedActions.has(action))" in script
    assert "remainingActions.forEach(action=>" in script
    assert "d.status==='source_unavailable'" in script


def test_frontend_action_script_is_versioned_and_revalidated_after_changes():
    main = re.sub(r"\s+", "", MAIN_PY.read_text(encoding="utf-8"))

    assert "asset_version=ACTION_TEMPLATE_JS.stat().st_mtime_ns" in main
    assert 'src=\"/action-template.js?v={asset_version}\"' in main
    assert 'headers={\"Cache-Control\":\"no-store\"}' in main
    assert 'headers={\"Cache-Control\":\"no-cache\"}' in main


def test_failed_or_unavailable_flows_are_visually_terminal():
    html = re.sub(r"\s+", "", INDEX_HTML.read_text(encoding="utf-8"))

    assert "['failed','error','source_unavailable'].includes(status)" in html
    assert "if(!view.finished)finishFlow(id,true)" in html
