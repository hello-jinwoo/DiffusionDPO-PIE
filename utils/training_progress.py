from __future__ import annotations

from typing import Optional

from tqdm.auto import tqdm


def create_progress_bar(
    current_step: int,
    max_steps: int,
    *,
    disable: bool = False,
    description: Optional[str] = None,
):
    """Create a tqdm progress bar seeded with a prior step count."""
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if current_step < 0:
        raise ValueError("current_step must be non-negative")
    if current_step > max_steps:
        raise ValueError("current_step cannot exceed max_steps")

    progress_bar = tqdm(total=max_steps, initial=current_step, disable=disable)
    if description is not None:
        progress_bar.set_description(description)
    return progress_bar
