"""
Integrity check for the labeling app, driven through Streamlit's AppTest.

Every step clicks or types through the real widgets, then verifies that the
workspace in st.session_state passes check_workspace, that the export
parses back into exactly the stored samples and labels, that the autosave
file matches the export, and that the labels equal an independently kept
expectation. Scripted scenarios cover each tab; a seeded random walk mixes
actions and tab switches.

Run: python app/labeling/check_labeling.py [steps] [seed]
"""

import json
import random
import re
import sys
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE)]

import decision_tab  # noqa: E402
import label_common as common  # noqa: E402
import sequence_tab  # noqa: E402
import token_tab  # noqa: E402

TABS = {
    "decision": ":material/rule: Decision",
    "token": ":material/sell: Token labeling",
    "sequence": ":material/segment: Sequence labeling",
}
MODULES = {"decision": decision_tab, "token": token_tab, "sequence": sequence_tab}


def allow_clearing() -> None:
    """AppTest reads set_value(None) on st.pills / st.segmented_control as "unchanged"; send an empty selection instead, as the browser does."""
    formatted = ButtonGroup.formatted_values.fget

    def formatted_values(self):
        return [] if getattr(self, "cleared", False) else formatted(self)

    ButtonGroup.formatted_values = property(formatted_values)


def choose(widget, value):
    widget.set_value(value)
    widget.cleared = value is None and isinstance(widget, ButtonGroup)
    return widget


class Driver:
    """Runs the app and checks every invariant after each interaction."""

    def __init__(self, folder: Path):
        self.at = AppTest.from_file(str(HERE / "label_app.py"), default_timeout=60)
        self.folder = folder
        self.tab = "decision"
        self.expected = {name: {} for name in TABS}
        self.steps = 0
        self.accepted = ""
        self.run()

    # ------------- driving
    def run(self, widget=None, allow_error: bool = False) -> None:
        self.at.session_state["task_tab"] = TABS[self.tab]
        (widget or self.at).run()
        self.steps += 1
        self.verify(allow_error)

    def switch(self, tab: str) -> None:
        self.tab = tab
        self.run()

    def find(self, kind: str, base: str):
        """The rendered widget of a kind whose key is base, or base plus a version suffix."""
        pattern = re.compile(rf"^{re.escape(base)}(_\d+)?$")
        matches = [w for w in getattr(self.at, kind) if w.key and pattern.match(w.key)]
        if not matches:
            raise AssertionError(f"no {kind} {base!r} on the page (step {self.steps})")
        return max(matches, key=lambda w: int(w.key.rsplit("_", 1)[-1]) if w.key[-1].isdigit() else -1)

    def has(self, kind: str, base: str) -> bool:
        try:
            self.find(kind, base)
            return True
        except AssertionError:
            return False

    def click(self, key: str, allow_error: bool = False) -> bool:
        button = self.find("button", key)
        if button.disabled:
            return False
        self.run(button.click(), allow_error)
        return True

    def space(self) -> dict:
        return self.at.session_state[self.tab]

    def notice(self) -> str:
        message = self.space().get("notice")
        return message["text"] if message else ""

    # ------------- invariants
    def verify(self, allow_error: bool) -> None:
        where = f"step {self.steps}, tab {self.tab}"
        exceptions = [e.value for e in self.at.exception]
        assert not exceptions, f"{where}: exception {exceptions[0][:500]} {list(self.at.exception[0].stack_trace)[-6:]}"
        current = self.notice() if self.tab in self.at.session_state else ""
        if allow_error:
            self.accepted = current
        errors = [e.value for e in self.at.error if not (self.accepted and e.value == self.accepted)]
        assert allow_error or not errors, f"{where}: error shown: {errors[0][:500] if errors else ''}"
        for name, module in MODULES.items():
            if name not in self.at.session_state:
                continue
            space = self.at.session_state[name]
            problems = common.check_workspace(module, space)
            assert not problems, f"{where}: {name} workspace invalid: {problems}"
            text = common.export_text(module, space)
            rows = [json.loads(line) for line in text.splitlines()]
            if rows:
                records, ids, labels = common.import_labeled(module, rows)
                order = sorted(space["labels"])
                assert records == [space["records"][i] for i in order], f"{where}: {name} samples changed in export"
                assert ids == [space["ids"][i] for i in order], f"{where}: {name} ids changed in export"
                assert list(labels.values()) == [space["labels"][i] for i in order], f"{where}: {name} labels changed in export"
            path = space["config"].get("autosave")
            if path and space["labels"]:
                saved = Path(path).read_text(encoding="utf-8")
                assert saved == text, f"{where}: {name} autosave file differs from the export"
            summary = {i: self.summary(name, label) for i, label in space["labels"].items()}
            assert summary == self.expected[name], (
                f"{where}: {name} labels {summary} != expected {self.expected[name]}"
            )

    @staticmethod
    def summary(name: str, label: dict):
        if name == "decision":
            return label["option"]
        if name == "token":
            return tuple(label["tags"])
        return (tuple((s["start"], s["end"], s["type"]) for s in label["spans"]), label["label"])

    # ------------- shared actions
    def load_example(self, example: str) -> None:
        self.run(self.find("selectbox", f"{self.tab}_example").select(example))
        self.expected[self.tab] = {}
        self.click(f"{self.tab}_load")
        assert self.space()["records"], "example did not load"

    def set_autosave(self) -> Path:
        path = self.folder / f"{self.tab}_autosave.jsonl"
        self.run(self.find("text_input", f"{self.tab}_cfg_autosave").input(str(path)))
        return path

    def go(self, button: str) -> None:
        self.click(f"{self.tab}_{button}")

    def pending_save(self) -> bool:
        return not self.find("button", f"{self.tab}_save").disabled

    # ------------- decision
    def answer(self, index: int | None) -> None:
        kind = "radio" if self.has("radio", "decision_answer") else "segmented_control"
        position = self.space()["position"]
        options = decision_tab.question(self.space()["config"])["options"]
        widget = self.find(kind, "decision_answer")
        if index is None:
            self.expected["decision"].pop(position, None)
            if kind == "radio" or widget.value is None:
                if position in self.space()["labels"]:
                    self.click("decision_unlabel")
                return
        else:
            self.expected["decision"][position] = options[index]
        self.run(choose(widget, index))

    # ------------- token
    def tag(self, tokens: list[int], tag_index: int) -> None:
        self.run(self.find("pills", "token_selection").set_value(tokens))
        self.click(f"token_tag_{tag_index}")

    def token_save(self) -> None:
        space = self.space()
        position = space["position"]
        draft = space["drafts"].get(position) or token_tab.draft(dict(space, drafts={}))
        if not self.pending_save():
            return
        self.expected["token"][position] = tuple(draft["tags"])
        self.click("token_save")

    # ------------- sequence
    def span(self, start: int, end: int, kind: str, allow_error: bool = False) -> None:
        if self.has("select_slider", "sequence_range"):
            self.run(self.find("select_slider", "sequence_range").set_range(start, end))
        self.run(self.find("segmented_control", "sequence_span_type").set_value(kind))
        self.click("sequence_add", allow_error)

    def sequence_label(self, name: str | None) -> None:
        kind = "radio" if self.has("radio", "sequence_label") else "segmented_control"
        self.run(choose(self.find(kind, "sequence_label"), name))

    def sequence_save(self) -> None:
        space = self.space()
        position = space["position"]
        if not self.pending_save():
            return
        draft = space["drafts"][position]
        self.expected["sequence"][position] = (tuple((s["start"], s["end"], s["type"]) for s in draft["spans"]), draft["label"])
        self.click("sequence_save")


# -------------    scripted scenarios    --------------------------
def decision_scenario(driver: Driver) -> None:
    driver.switch("decision")
    for example in decision_tab.EXAMPLES:
        driver.load_example(example)
        driver.set_autosave()
        options = decision_tab.question(driver.space()["config"])["options"]
        for row in range(len(driver.space()["records"])):
            driver.answer(row % len(options))
        assert len(driver.space()["labels"]) == len(driver.space()["records"]), f"{example}: not every row saved"
        driver.go("previous")
        driver.answer(None)
        driver.answer(0)
    driver.switch("token")
    driver.switch("decision")
    assert driver.space()["labels"], "decision labels lost after switching tabs"
    labels_before = dict(driver.space()["labels"])
    driver.run(driver.find("text_area", "decision_cfg_instructions").input("A changed question"))
    assert driver.space()["labels"] == labels_before, "editing the question changed saved labels"


def token_scenario(driver: Driver) -> None:
    driver.switch("token")
    for example in token_tab.EXAMPLES:
        driver.load_example(example)
        driver.set_autosave()
        tags = token_tab.tag_set(driver.space()["config"])
        for row in range(len(driver.space()["records"])):
            size = len(token_tab.draft(driver.space())["tokens"])
            driver.tag([0], 1)
            driver.tag(list(range(1, size, 2)), 2 % len(tags))
            driver.token_save()
        driver.go("previous")
        driver.tag([0], 0)
        driver.click("token_discard")
        driver.switch("sequence")
        driver.switch("token")
        driver.tag([0], 1)
        driver.go("previous")
        driver.go("next")
        assert token_tab.draft(driver.space())["tags"][0] == tags[1], "token draft lost on navigation"
        driver.token_save()
        driver.go("previous")
        driver.expected["token"].pop(driver.space()["position"])
        driver.click("token_unlabel")


def sequence_scenario(driver: Driver) -> None:
    driver.switch("sequence")
    for example in sequence_tab.EXAMPLES:
        driver.load_example(example)
        driver.set_autosave()
        types = sequence_tab.span_types(driver.space()["config"])
        names = sequence_tab.label_set(driver.space()["config"])
        for row in range(len(driver.space()["records"])):
            size = len(sequence_tab.draft(driver.space())["tokens"])
            driver.span(0, 1, types[0])
            before = json.dumps(driver.space()["drafts"])
            driver.span(1, 2, types[-1], allow_error=True)
            assert "overlaps" in driver.notice(), "overlapping span was not rejected"
            assert json.dumps(driver.space()["drafts"]) == before, "rejected span changed the draft"
            driver.span(size - 1, size - 1, types[-1])
            driver.run(driver.find("pills", "sequence_pick").set_value(1))
            assert driver.space()["view"]["range"] == (size - 1, size - 1), "picking a span did not load its range"
            driver.run(driver.find("select_slider", "sequence_range").set_range(size - 2, size - 1))
            driver.click("sequence_update")
            driver.run(driver.find("pills", "sequence_pick").set_value(0))
            driver.click("sequence_delete")
            driver.span(0, 0, types[0])
            driver.sequence_label(names[row % len(names)])
            driver.sequence_save()
        driver.go("previous")
        driver.sequence_label(None)
        driver.sequence_save()


def import_scenario(driver: Driver) -> None:
    """Round-trip each tab's export through the uploader, then feed it corrupted files."""
    for name in TABS:
        driver.switch(name)
        exported = common.export_text(MODULES[name], driver.space())
        labels = dict(driver.space()["labels"])
        expected_before = dict(driver.expected[name])
        expected = {i: e for i, e in zip(range(len(labels)), [driver.expected[name][k] for k in sorted(labels)])}
        driver.expected[name] = expected
        driver.run(driver.find("file_uploader", f"{name}_upload").set_value((f"{name}.jsonl", exported.encode(), "application/json")))
        assert [driver.space()["labels"][i] for i in sorted(driver.space()["labels"])] == [labels[i] for i in sorted(labels)], (
            f"{name}: labels changed after re-importing the export"
        )
        before = json.dumps(driver.space(), default=str)
        rows = [json.loads(line) for line in exported.splitlines()]
        corrupt = json.loads(json.dumps(rows))
        if name == "decision":
            corrupt[0]["answer"] = len(corrupt[0]["options"]) + 3
        elif name == "token":
            corrupt[0]["tags"] = corrupt[0]["tags"][:-1]
        else:
            corrupt[0]["bio"] = ["B-X"] * len(corrupt[0]["tokens"])
        data = "".join(json.dumps(row) + "\n" for row in corrupt).encode()
        driver.run(driver.find("file_uploader", f"{name}_upload").set_value((f"{name}_bad.jsonl", data, "application/json")), allow_error=True)
        assert "rejected" in driver.notice(), f"{name}: corrupted file was not rejected"
        after = json.loads(json.dumps(driver.space(), default=str))
        after["notice"] = json.loads(before)["notice"]
        assert json.dumps(after, default=str) == before, f"{name}: rejected file changed the workspace"
        broken = b'{"not json"\n'
        driver.run(driver.find("file_uploader", f"{name}_upload").set_value((f"{name}_broken.jsonl", broken, "application/json")), allow_error=True)
        assert "not valid JSON" in driver.notice(), f"{name}: broken JSON was not reported"
        driver.expected[name] = expected_before
        driver.click(f"{name}_restore")
        assert [driver.space()["labels"][i] for i in sorted(driver.space()["labels"])] == [labels[i] for i in sorted(labels)], (
            f"{name}: restore did not bring back the previous data"
        )


def csv_scenario(driver: Driver) -> None:
    """A CSV with blanks, numbers, unicode, quotes and newlines loads and labels cleanly."""
    driver.switch("sequence")
    data = (
        'id,text,score,note\n'
        '1,"Zoë visited Zürich, then \'Oslo\'.",3.5,\n'
        '2,"Line one\nline two with ""quotes""",,n/a\n'
        '3,NaN is a word here,inf,x\n'
    ).encode()
    driver.expected["sequence"] = {}
    driver.run(driver.find("file_uploader", "sequence_upload").set_value(("rows.csv", data, "text/csv")))
    assert len(driver.space()["records"]) == 3, "CSV rows did not load"
    driver.span(0, 0, sequence_tab.span_types(driver.space()["config"])[0])
    driver.sequence_save()
    driver.sequence_save()
    driver.sequence_save()


def random_walk(driver: Driver, steps: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    for _ in range(steps):
        tab = driver.tab
        space = driver.space()
        roll = rng.random()
        if roll < 0.08:
            driver.switch(rng.choice(list(TABS)))
        elif roll < 0.12 or not space["records"]:
            driver.load_example(rng.choice(list(MODULES[tab].EXAMPLES)))
            if rng.random() < 0.5:
                driver.set_autosave()
        elif roll < 0.25:
            driver.go(rng.choice(["previous", "next", "unlabeled"]))
        elif tab == "decision":
            options = decision_tab.question(space["config"])["options"]
            driver.answer(rng.choice([None, *range(len(options))]))
        elif tab == "token":
            size = len(token_tab.draft(space)["tokens"])
            choice = rng.random()
            if choice < 0.6:
                picked = sorted(rng.sample(range(size), rng.randint(1, size)))
                driver.tag(picked, rng.randrange(len(token_tab.tag_set(space["config"]))))
            elif choice < 0.85:
                driver.token_save()
            elif choice < 0.95:
                driver.click("token_discard")
            elif space["position"] in space["labels"]:
                driver.expected["token"].pop(space["position"])
                driver.click("token_unlabel")
        else:
            size = len(sequence_tab.draft(space)["tokens"])
            types = sequence_tab.span_types(space["config"])
            choice = rng.random()
            if choice < 0.5:
                start = rng.randrange(size)
                driver.span(start, min(size - 1, start + rng.randint(0, 2)), rng.choice(types), allow_error=True)
            elif choice < 0.65:
                driver.sequence_label(rng.choice([None, *sequence_tab.label_set(space["config"])]))
            elif choice < 0.85:
                driver.sequence_save()
            elif sequence_tab.draft(space)["spans"]:
                driver.run(driver.find("pills", "sequence_pick").set_value(0))
                driver.click("sequence_delete")


# -------------    unit checks    ---------------------------------
def unit_checks(folder: Path) -> None:
    import numpy as np

    assert common.to_json_safe({"a": np.int64(3), "b": float("nan"), "c": (1, 2), 4: np.array([1.5])}) == {
        "a": 3, "b": None, "c": [1, 2], "4": [1.5]
    }
    rows = common.read_records("x.csv", b"a,b\n1,\n,2\n")
    assert rows == [{"a": "1", "b": None}, {"a": None, "b": "2"}], rows
    for bad in (b"", b"[1, 2]", b'{"a": 1}\n[2]'):
        try:
            common.read_records("x.jsonl" if b"\n" in bad else "x.json", bad)
        except (common.IntegrityError, json.JSONDecodeError):
            continue
        raise AssertionError(f"bad file accepted: {bad!r}")
    target = folder / "atomic.jsonl"
    common.write_atomic(str(target), "one\n")
    common.write_atomic(str(target), "two\n")
    assert target.read_text() == "two\n" and target.with_name("atomic.jsonl.bak").read_text() == "one\n"
    for bad_path in (str(folder / "missing" / "x.jsonl"), str(folder / "x.txt")):
        try:
            common.write_atomic(bad_path, "x")
        except common.IntegrityError:
            continue
        raise AssertionError(f"bad autosave path accepted: {bad_path}")
    assert sequence_tab.normalize_range(3, 5) == (3, 3)
    assert sequence_tab.normalize_range([4, 1], 5) == (1, 4)
    assert sequence_tab.normalize_range((0, 99), 5) == (0, 4)
    assert sequence_tab.normalize_range(None, 5) == (0, 0)


def main(steps: int = 300, seed: int = 7) -> None:
    allow_clearing()
    with tempfile.TemporaryDirectory() as workdir:
        folder = Path(workdir)
        unit_checks(folder)
        driver = Driver(folder)
        for scenario in (decision_scenario, token_scenario, sequence_scenario, import_scenario, csv_scenario):
            scenario(driver)
            print(f"{scenario.__name__}: ok ({driver.steps} app runs so far)")
        random_walk(driver, steps, seed)
        print(f"random_walk: ok ({steps} actions, {driver.steps} app runs)")
    print("all labeling integrity checks passed")


if __name__ == "__main__":
    main(*(int(argument) for argument in sys.argv[1:3]))
