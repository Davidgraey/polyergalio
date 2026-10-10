"""Compact density and distinct sample and labeling panels, scoped to ``st-key-<key>`` classes."""

COMPACT_CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; padding-bottom: 2.5rem; }
[data-testid="stVerticalBlock"] { gap: 0.5rem; }
[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
.stButton button, .stDownloadButton button {
  min-height: 1.75rem; padding: 0.1rem 0.6rem; font-size: 0.8rem; line-height: 1.2;
}
[data-testid="stExpander"] summary { padding: 0.3rem 0.6rem; font-size: 0.82rem; }
[data-testid="stMarkdownContainer"] h5 { margin: 0; padding: 0; }
.st-key-sample_panel { background: #e6f1ef; border: 1px solid #b7d3cf !important;
  border-left: 6px solid #272736 !important; }
.st-key-sample_text { background: #ffffff; border: 1px solid #b7d3cf !important; border-radius: 6px;
  box-shadow: inset 0 1px 3px rgba(39,39,54,0.08); }
.st-key-sample_text [data-testid="stMarkdownContainer"] p { font-size: 1rem; line-height: 1.55; }
.st-key-label_panel { background: #ffffff; border: 1px solid #ecc3b8 !important;
  border-left: 6px solid #d9573a !important; margin-top: 0.75rem; }
.st-key-action_bar { background: #fbeee9; border-radius: 6px; padding: 0.3rem 0.5rem; }
</style>
"""
