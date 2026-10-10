"""Task editors and shared UI pieces for the labeling page.

The persisted draft (keyed ``task::sample``) is the single source of truth for in-progress work.
Widgets are seeded from it and write back through ``ctx.edit``. Widget keys carry a per-task
version; bumping it re-creates the widgets from the draft instead of stale widget state.
"""

from task_defs import TaskType

from .base import CHART_HEIGHT, CONTEXT_HEIGHT, STATUS_BADGE, TYPE_ICONS, Ctx, color_for, plain
from .cards import guidance_card
from .classification import classification_body
from .contrastive import contrastive_body
from .span import span_body
from .token import token_body

BODIES = {
    TaskType.CLASSIFICATION: classification_body,
    TaskType.TOKEN: token_body,
    TaskType.SPAN: span_body,
    TaskType.CONTRASTIVE: contrastive_body,
}

__all__ = ["BODIES", "CHART_HEIGHT", "CONTEXT_HEIGHT", "STATUS_BADGE", "TYPE_ICONS", "Ctx", "color_for",
           "guidance_card", "plain"]
