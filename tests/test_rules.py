from pathlib import Path

from app.rules.models import RuleFile
import json


def test_rule_file_is_valid():
    payload = json.loads(Path("config/rules.json").read_text(encoding="utf-8"))
    rules = RuleFile.model_validate(payload)
    assert rules.version == 1
    assert any(rule.id == "password_reset" for rule in rules.rules)
    assert any(rule.id == "no_answer_suggest_email" for rule in rules.rules)
