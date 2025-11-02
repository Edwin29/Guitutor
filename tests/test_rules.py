from guitutor.eval.fingering import FingeringRules, FingeringEvaluator, Verdict
import os, tempfile, yaml

def test_rules_compare():
    data = {
        "items":[{"name":"C","expected":[{"finger":1,"string":2,"fret":1}]}]
    }
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as f:
        yaml.safe_dump(data,f); path=f.name
    rules = FingeringRules.from_yaml(path)
    ev = FingeringEvaluator(rules)
    v: Verdict = ev.compare({1:(2,1)})
    assert v.ok and v.details[1]=="ok"