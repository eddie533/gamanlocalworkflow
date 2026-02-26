#!/usr/bin/env python3
"""
Role Classifier - Map raw role strings to standardised role categories using GPT-5-nano.
"""

import json
from openai import OpenAI
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

ALLOWED_ROLES = [
    "Founder Still In Business",
    "Founder NO Longer in Business",
    "Shareholder",
    "Family of Founder",
    "Director/Chairman that is NOT Shareholder",
    "Management NOT Founder/Shareholder",
]


class RoleClassifier:
    """Classify a raw role string into one of the allowed categories using gpt-5-nano."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = f"""You are a role classifier for a B2B contact database.

Given a person's role/title at a company, classify it into EXACTLY ONE of these categories:
{chr(10).join(f'- {r}' for r in ALLOWED_ROLES)}

CLASSIFICATION RULES:
- "Founder Still In Business": founders, co-founders, or owners who are still active at the company
- "Founder NO Longer in Business": founders who have left or the company is defunct
- "Shareholder": investors, shareholders, or equity holders who are not founders
- "Family of Founder": relatives of the founder (spouse, child, sibling, etc.)
- "Director/Chairman that is NOT Shareholder": board directors, chairmen, non-executive directors who are not known shareholders
- "Management NOT Founder/Shareholder": CEOs, CTOs, CFOs, VPs, managers, and other executives who are not founders or shareholders

When in doubt between categories, prefer "Management NOT Founder/Shareholder" as the default for executive titles (CEO, CTO, COO, etc.) unless the role explicitly mentions founder, owner, shareholder, or director/chairman.

---
**Output requirements**
• Respond **only** with a valid JSON object.
• Do **not** wrap it in backticks or code fences.
• Use exactly this key:
  - "role" (one of the allowed categories above, exact spelling)"""

    def classify(self, role: str) -> str:
        """Return one of the allowed role categories for the given raw role string."""
        if not role or not role.strip():
            return "Management NOT Founder/Shareholder"

        user_prompt = f"Role/title: {role.strip()}"

        try:
            combined_input = f"{self.system_prompt}\n\n{user_prompt}"

            response = self.client.responses.create(
                model="gpt-5-nano",
                input=combined_input,
            )

            result = json.loads(response.output_text)
            classified = result.get("role", "Management NOT Founder/Shareholder")

            if classified not in ALLOWED_ROLES:
                return "Management NOT Founder/Shareholder"

            return classified

        except Exception as e:
            print(f"  ⚠ Role classification error for '{role}': {e}")
            return "Management NOT Founder/Shareholder"
