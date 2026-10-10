"""Box-and-arrow diagrams for models that are not Network graphs."""

import textwrap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SOURCE_COLOR = "#ffd166"
STEP_COLOR = "#118ab2"
SINK_COLOR = "#06d6a0"


def plot_graph(
    edges: list,
    title: str = "",
    x_spacing: float = 3.2,
    y_spacing: float = 1.8,
    box_width: float = 2.8,
    box_height: float = 1.0,
) -> Figure:
    """
    Layered top-to-bottom diagram of labelled boxes joined by arrows.

    Parameters
    ----------
    edges : (source, target) label pairs forming an acyclic graph
    title : figure title
    x_spacing, y_spacing : distance between boxes in a row and between rows
    box_width, box_height : box size

    Returns
    -------
    Figure
    """
    nodes = list(dict.fromkeys(name for edge in edges for name in edge))
    parents = {node: [s for s, t in edges if t == node] for node in nodes}
    children = {node: [t for s, t in edges if s == node] for node in nodes}
    depth = {}

    def level(node):
        if node not in depth:
            depth[node] = 1 + max((level(p) for p in parents[node]), default=-1)
        return depth[node]

    rows = {}
    for node in nodes:
        rows.setdefault(level(node), []).append(node)

    positions = {}
    for row, members in rows.items():
        for index, node in enumerate(members):
            positions[node] = ((index - (len(members) - 1) / 2) * x_spacing, -row * y_spacing)

    columns = max(len(members) for members in rows.values())
    width = (columns - 1) * x_spacing + box_width + 1
    height = (len(rows) - 1) * y_spacing + box_height + 1
    fig, ax = plt.subplots(figsize=(width * 0.55, height * 0.55))

    for node, (x, y) in positions.items():
        color = SOURCE_COLOR if not parents[node] else SINK_COLOR if not children[node] else STEP_COLOR
        ax.add_patch(
            FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="black",
                zorder=2,
            )
        )
        ax.text(x, y, textwrap.fill(node, width=22), ha="center", va="center", fontsize=8, zorder=3)

    for source, target in edges:
        x_from, y_from = positions[source]
        x_to, y_to = positions[target]
        skips = depth[target] - depth[source] > 1
        ax.add_patch(
            FancyArrowPatch(
                (x_from, y_from - box_height / 2),
                (x_to, y_to + box_height / 2),
                arrowstyle="-|>",
                mutation_scale=12,
                connectionstyle="arc3,rad=0.35" if skips else "arc3,rad=0",
                linestyle="--" if skips else "-",
                color="#ef476f" if skips else "gray",
                zorder=1,
            )
        )

    xs = [x for x, _ in positions.values()]
    ax.set_xlim(min(xs) - box_width / 2 - 0.5, max(xs) + box_width / 2 + 0.5)
    ax.set_ylim(-(len(rows) - 1) * y_spacing - box_height / 2 - 0.5, box_height / 2 + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    return fig


PHASE_COLORS = {
    "discover": "#f4a261",
    "connect": "#ffd166",
    "initialize": "#118ab2",
    "step": "#06d6a0",
    "aggregate": "#8338ec",
    "apply": "#ef476f",
    "collect": "#90be6d",
}
STATUS_COLORS = {
    "listening": "#e0e0e0",
    "connected": "#ffd166",
    "rejected": "#ef476f",
    "discovered": "#fbe3c8",
    "skipped": "#f1f1f1",
    "dropped": "#ef9a9a",
    "restarted": "#ffcc80",
    "reconnected": "#b2dfdb",
    "initialized": "#a8dadc",
    "stepped": "#b7efc5",
    "updated": "#cdb4db",
    "collected": "#c7e9b4",
}
PHASE_MESSAGES = {
    "discover": ("who is there? (UDP)", "node id + port"),
    "connect": ("challenge reply (HMAC)", "accepted + node id"),
    "initialize": ("model + hyperparameters", "completed"),
    "step": ("data shard", "gradients + loss"),
    "aggregate": (None, None),
    "apply": ("pooled gradients", "completed"),
    "collect": ("export model", "serialized weights"),
}


def plot_cluster(nodes: list, phases: tuple, phase, round_index: int, pooled_norm) -> Figure:
    """
    Orchestrator, nodes and the messages of the current phase, under a strip of the phases.

    Parameters
    ----------
    nodes : dicts with id, status and, once known, rows, loss, gradient_norm, sent_kb, received_kb
    phases : phase names in order, drawn as the strip along the top
    phase : phase that just ran, or None
    round_index : completed rounds
    pooled_norm : norm of the last pooled gradients, or None

    Returns
    -------
    Figure
    """
    slot = 1.7
    bottom = -1.2 - slot * (len(nodes) - 1) - 0.9
    fig, ax = plt.subplots(figsize=(9, (1.2 - bottom) * 0.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom, 1.2)
    ax.axis("off")
    current = phases.index(phase) if phase in phases else -1
    step = 9.4 / len(phases)

    for index, name in enumerate(phases):
        active = index == current
        ax.add_patch(
            FancyBboxPatch(
                (0.3 + index * step, 0.15),
                step - 0.15,
                0.6,
                boxstyle="round,pad=0.02",
                facecolor=PHASE_COLORS[name] if index <= current else "#f1f1f1",
                edgecolor="black",
                linewidth=2.5 if active else 0.8,
            )
        )
        ax.text(0.3 + index * step + (step - 0.15) / 2, 0.45, name, ha="center", va="center", fontsize=8, fontweight="bold" if active else "normal")

    middle = -1.2 - slot * (len(nodes) - 1) / 2
    ax.add_patch(
        FancyBboxPatch(
            (0.3, middle - 0.8),
            2.6,
            1.6,
            boxstyle="round,pad=0.03",
            facecolor=SOURCE_COLOR,
            edgecolor="black",
            linewidth=2.5 if phase == "aggregate" else 1.0,
        )
    )
    orchestrator_text = f"Orchestrator\nround {round_index}"
    if pooled_norm is not None:
        orchestrator_text += f"\npooled |g| {pooled_norm:.3f}"
    if phase == "aggregate":
        orchestrator_text += "\naveraging gradients"
    ax.text(1.6, middle, orchestrator_text, ha="center", va="center", fontsize=9)

    outbound, inbound = PHASE_MESSAGES.get(phase, (None, None))
    for index, node in enumerate(nodes):
        y = -1.2 - index * slot
        rejected = node["status"] == "rejected"
        ax.add_patch(
            FancyBboxPatch(
                (6.3, y - 0.65),
                3.4,
                1.3,
                boxstyle="round,pad=0.03",
                facecolor=STATUS_COLORS.get(node["status"], "#e0e0e0"),
                edgecolor="black",
            )
        )
        lines = [node["id"], node["status"]]
        if "rows" in node:
            lines.append(f"{node['rows']} rows  loss {node['loss']:.3f}  |g| {node['gradient_norm']:.3f}")
        ax.text(8.0, y, "\n".join(lines), ha="center", va="center", fontsize=8)

        offline = node["status"] in ("listening", "dropped", "skipped", "restarted") or (
            node["status"] == "discovered" and phase != "discover"
        )
        active = phase in PHASE_COLORS and phase != "aggregate" and not offline
        color = "#ef476f" if rejected else PHASE_COLORS[phase] if active else "#cccccc"
        style = "--" if rejected or not active else "-"
        for offset, message, direction, size in ((0.22, outbound, 1, node.get("sent_kb")), (-0.22, inbound, -1, node.get("received_kb"))):
            start, end = (2.95, 6.25) if direction == 1 else (6.25, 2.95)
            ax.add_patch(
                FancyArrowPatch(
                    (start, y + offset),
                    (end, y + offset),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=color,
                    linestyle=style,
                    linewidth=1.8 if active else 1.0,
                )
            )
            if active and message and not (rejected and direction == -1):
                label = "rejected" if rejected else message + (f" ({size:.1f} KB)" if size else "")
                ax.text(4.6, y + offset + 0.12, label, ha="center", va="bottom", fontsize=7, color="black")
    return fig
