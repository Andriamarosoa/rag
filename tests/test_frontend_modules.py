from html.parser import HTMLParser
from pathlib import Path
import re


INDEX_HTML = Path(__file__).resolve().parents[1] / "index.html"
ACTION_TEMPLATE_JS = Path(__file__).resolve().parents[1] / "action-template.js"


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


def test_assistant_sources_use_safe_dom_links_without_rendering_model_html():
    html = INDEX_HTML.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", html)

    assert "view.content.textContent=text" in compact
    assert "renderSources(view.sources,d.sources)" in compact
    assert "safeHttpUrl(source.url)" in compact
    assert "url.protocol==='http:'||url.protocol==='https:'" in compact
    assert "url.hostname.toLowerCase()==='skb.uniconsults.mu'" in compact
    assert "link.target='_blank'" in compact
    assert "link.rel='noopener'" in compact
    assert "title.textContent=String(source.title" in compact
    assert "meta.textContent=metadata.join('·')" in compact


def test_production_action_template_handler_keeps_source_rendering():
    script = re.sub(r"\s+", "", ACTION_TEMPLATE_JS.read_text(encoding="utf-8"))

    assert "RESPONSE_HANDLERS.set('assistant.completed',completeAssistantWithTemplates)" in script
    assert "renderSources(view.sources,d.sources)" in script
    assert "d.status==='source_unavailable'" in script


def test_failed_or_unavailable_flows_are_visually_terminal():
    html = re.sub(r"\s+", "", INDEX_HTML.read_text(encoding="utf-8"))

    assert "['failed','error','source_unavailable'].includes(status)" in html
    assert "if(!view.finished)finishFlow(id,true)" in html
