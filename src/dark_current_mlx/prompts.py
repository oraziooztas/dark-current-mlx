"""Prompt templates for pairwise LLM-as-a-judge calls."""

from __future__ import annotations

from .models import JudgeCall, PairwiseItem

PAIRWISE_JUDGE_TEMPLATE_ID = "pairwise_judge_v1"

PAIRWISE_JUDGE_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question below.
You should choose the response that follows the user's instructions better and provides more helpful, harmless, and honest information.

[User Question]
{instruction}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]

Provide your evaluation reasoning first, and then output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, or "[[Tie]]" for a tie.
"""


def render_pairwise_prompt(item: PairwiseItem, order: str) -> JudgeCall:
    """Render the normal or swapped prompt for an item."""

    if order not in {"ab", "ba"}:
        raise ValueError("order must be 'ab' or 'ba'")

    response_a = item.response_a if order == "ab" else item.response_b
    response_b = item.response_b if order == "ab" else item.response_a
    prompt = PAIRWISE_JUDGE_TEMPLATE.format(
        instruction=item.instruction,
        response_a=response_a,
        response_b=response_b,
    )
    return JudgeCall(item=item, order=order, prompt=prompt)
