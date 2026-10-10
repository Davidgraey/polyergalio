"""
Drag-to-select span picker (Streamlit Components v2, inline JS, no build step).

The sample is drawn as one <span> per tokenizer token. Dragging (or clicking, or
shift-clicking) selects a token range; the component reports ``{"range": [first, last]}``
as token indices, so offsets always snap to the same tokenizer the validators use.
Existing spans are shown as coloured underlines. Python stays the source of truth:
the selection is passed back in through ``data`` on every run.

Text is inserted with textContent only, never innerHTML.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import streamlit as st

HEX = {"blue": "#1c83e1", "green": "#21c354", "orange": "#ff8700", "violet": "#803df5",
       "red": "#ff4b4b", "yellow": "#e8b900", "gray": "#808495"}

CSS = """
.hp-root { line-height: 2; font-size: 1rem; padding: 0.25rem 0.1rem; user-select: none;
  -webkit-user-select: none; touch-action: none; cursor: text; }
.tok { border-radius: 3px; padding: 1px 0; }
.tok:hover { background: rgba(217, 87, 58, 0.14); }
.tok.has { background: color-mix(in srgb, var(--c) 20%, transparent); }
.tok.sel, .tok.sel.has { background: #d9573a; color: #fff; }
.hint { font-size: 0.72rem; opacity: 0.6; margin-top: 0.2rem; }
"""

JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const old = parentElement.querySelector('.hp-wrap');
  if (old) old.remove();
  const wrap = document.createElement('div');
  wrap.className = 'hp-wrap';
  const root = document.createElement('div');
  root.className = 'hp-root';
  wrap.appendChild(root);
  const hint = document.createElement('div');
  hint.className = 'hint';
  hint.textContent = 'Drag over the text to select a span. Shift-click extends the selection.';
  wrap.appendChild(hint);
  parentElement.appendChild(wrap);

  const text = data.text, toks = data.tokens, els = [];
  let cur = 0;
  toks.forEach((tk, i) => {
    if (tk.s > cur) root.appendChild(document.createTextNode(text.slice(cur, tk.s)));
    const el = document.createElement('span');
    el.className = 'tok';
    el.textContent = text.slice(tk.s, tk.e);
    const own = data.owner[i];
    if (own && own.length) {
      el.classList.add('has');
      el.style.setProperty('--c', own[0].color);
      el.style.boxShadow = own.map((o, k) => `0 ${2 + 3 * k}px 0 0 ${o.color}`).join(', ');
      el.style.marginBottom = `${3 * (own.length - 1)}px`;
      el.title = own.map((o) => o.label).join(' + ');
    }
    root.appendChild(el);
    els.push(el);
    cur = tk.e;
  });
  if (cur < text.length) root.appendChild(document.createTextNode(text.slice(cur)));

  let sel = data.range.slice();
  let anchor = sel[0], dragging = false, cand = sel;
  const paint = (lo, hi) => els.forEach((el, i) => el.classList.toggle('sel', i >= lo && i <= hi));
  paint(sel[0], sel[1]);

  const indexAt = (x, y) => {
    const hit = root.getRootNode().elementFromPoint(x, y);
    const el = hit && hit.closest ? hit.closest('.tok') : null;
    return el && root.contains(el) ? els.indexOf(el) : -1;
  };
  const extend = (i) => {
    cand = [Math.min(anchor, i), Math.max(anchor, i)];
    paint(cand[0], cand[1]);
  };

  root.addEventListener('pointerdown', (e) => {
    const i = indexAt(e.clientX, e.clientY);
    if (i < 0) return;
    anchor = e.shiftKey ? sel[0] : i;
    dragging = true;
    extend(i);
    e.preventDefault();
  });
  root.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const i = indexAt(e.clientX, e.clientY);
    if (i >= 0) extend(i);
  });
  const finish = () => {
    if (!dragging) return;
    dragging = false;
    sel = cand;
    setTriggerValue('picked', sel);
  };
  root.addEventListener('pointerup', finish);
  root.addEventListener('pointercancel', finish);
  document.addEventListener('pointerup', finish);
  return () => document.removeEventListener('pointerup', finish);
}
"""


def available() -> bool:
    return hasattr(getattr(st, "components", None), "v2") and hasattr(st.components.v2, "component")


_component = st.components.v2.component("hitl_span_picker", css=CSS, js=JS) if available() else None


def span_picker(key: str, text: str, toks: Sequence[tuple], spans: List[dict], colors: Callable[[str], str],
                rng: tuple, height: int = 150) -> Optional[list]:
    """Render the picker and return ``[first, last]`` token indices on the run where the user just selected.

    ``toks`` are (text, start, end); ``colors`` maps a span label to a Streamlit color. Selections are
    *trigger* values, so choosing the same range twice still reports, and the component key must stay
    stable across edits (a new key remounts it and loses the interaction).
    """
    owner: List[List[dict]] = [[] for _ in toks]
    for sp in spans:
        for i, (_, s, e) in enumerate(toks):
            if s >= int(sp["start"]) and e <= int(sp["end"]):
                owner[i].append({"color": HEX.get(colors(sp.get("label")), HEX["gray"]),
                                 "label": str(sp.get("label", ""))})
    data = {"text": text, "tokens": [{"s": s, "e": e} for _, s, e in toks], "owner": owner,
            "range": [int(rng[0]), int(rng[1])]}
    result = _component(data=data, key=key, on_picked_change=lambda: None, height=height)
    picked = result.picked if result is not None else None
    return list(picked) if picked else None
