"""
Roast Engine for Dungeon Nutrition Web App.

This module defines a RoastEngine class that generates snarky, Dungeon Crawler Carl‑style
comments about food items based on their nutritional values. Rules determine which
"tags" apply to a given nutrition profile, and each tag has a list of roast lines
from which a random one is chosen. The engine combines an opener, all applicable
roasts, and a closer into a single string.
"""

import random
from typing import Dict, List, Tuple


class RoastEngine:
    """Generate dungeon‑themed roasts from nutrition information."""

    # Each rule is a tuple of (tag_name, boolean expression as a string). The expression
    # is evaluated with a dict 'n' containing nutrition values.
    RULES: List[Tuple[str, str]] = [
        ("sugar_bomb", "n.get('sugar_g', 0) >= 20"),
        ("salt_lick", "n.get('sodium_mg', 0) >= 600"),
        ("sad_snack", "n.get('protein_g', 0) < 5 and n.get('calories', 0) >= 250"),
        ("virtuous", "n.get('fiber_g', 0) >= 5"),
        ("frankenfood", "n.get('ingredients_count', 0) > 20"),
    ]

    # Predefined lines for each tag. Feel free to expand or tweak these for your own
    # brand of sarcasm. They'll be combined with an opener and a closer.
    ROASTS: Dict[str, List[str]] = {
        "sugar_bomb": [
            "Behold: liquid regret. Your pancreas is filing for PTO.",
            "That’s not a beverage; that’s a side quest for insulin.",
            "Ah, vintage corn syrup – bold nose of citrus, lingering finish of regret.",
        ],
        "salt_lick": [
            "Ah yes, artisanal ocean dust. Hydrate, champion.",
            "Your blood pressure just unlocked New Game+.",
            "Sodium so high the ocean wrote to say 'tone it down.'",
        ],
        "sad_snack": [
            "High calories, low protein – bold choice. Gains remain theoretical.",
            "It’s like air, but it costs macros.",
            "Zero sugar, zero joy. Net carbs: emotional.",
        ],
        "virtuous": [
            "Fiber? In this economy? Certified buff to future‑you.",
            "Protein bomb acquired. Your biceps send a polite nod.",
            "A wholesome pick? Who are you and what did you do with Ben?",
        ],
        "frankenfood": [
            "Ingredients list reads like a spellbook. Try not to summon E‑numbers.",
            "If you can’t pronounce it, maybe don’t lunch it.",
            "Alchemy for breakfast? Bold move.",
        ],
    }

    def tags_from_nutrition(self, nutrition: Dict[str, float]) -> List[str]:
        """Return a list of tags whose rules match the given nutrition data."""
        tags: List[str] = []
        for name, expr in self.RULES:
            try:
                if eval(expr, {}, {"n": nutrition}):
                    tags.append(name)
            except Exception:
                # Ignore malformed expressions or missing values
                continue
        return tags

    def roast(self, nutrition: Dict[str, float]) -> str:
        """Generate a combined roast string for the given nutrition profile."""
        tags = self.tags_from_nutrition(nutrition)
        # If no tag applies, default to virtuous; otherwise leave tags unchanged
        if not tags:
            tags = ["virtuous"]

        opener = random.choice([
            "Scanning…",
            "Ah, a new offering. Let’s see…",
            "Analyzing your loot…",
        ])
        lines = [opener]
        for tag in tags:
            lines.append(random.choice(self.ROASTS.get(
                tag, ["Mysterious morsel detected."])))
        closer = random.choice([
            "Cart debuff applied.",
            "Proceed, brave shopper.",
            "Confess your crimes at self‑checkout.",
        ])
        lines.append(closer)
        return " " .join(lines)
