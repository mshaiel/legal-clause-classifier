"""
Data pipeline for LexGLUE/LEDGAR legal contract provision classification.

Handles:
- Dataset loading from Hugging Face Hub (coastalcph/lex_glue, ledgar)
- Class ID to Provision Name bidirectional mappings
- Phi-3-mini conversational prompt formatting for SFT
- Inference prompt generation
"""

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

# LEDGAR 100 Provision Categories (canonical order matching coastalcph/lex_glue)
LEDGAR_PROVISION_CLASSES = [
    "Adjustments",
    "Affiliate Transactions",
    "Affiliates",
    "Amendments",
    "Anti-Corruption Laws",
    "Applicable Laws",
    "Approvals",
    "Arbitration",
    "Assignment",
    "Assignments",
    "Assigns",
    "Audits",
    "Authorizations",
    "Books",
    "Brokers",
    "Capitalization",
    "Change In Control",
    "Closings",
    "Compliance With Laws",
    "Conditions Precedent",
    "Confidentiality",
    "Consent To Jurisdiction",
    "Consents",
    "Construction",
    "Cooperation",
    "Costs",
    "Counterparts",
    "Death",
    "Defined Terms",
    "Disability",
    "Disclosure",
    "Effect Of Termination",
    "Effectiveness",
    "Entire Agreements",
    "Environmental Matters",
    "Enforceability",
    "Escrow",
    "Exclusivity",
    "Expenses",
    "Fees",
    "Financial Statements",
    "Force Majeure",
    "Further Assurances",
    "Governing Laws",
    "Headings",
    "Indemnifications",
    "Insurance",
    "Intellectual Property",
    "Interest",
    "Interim Operating Covenants",
    "Internal Controls",
    "Investments",
    "Jurisdictions",
    "Labels",
    "Liabilities",
    "Liens",
    "Litigations",
    "Miscellaneous",
    "Modification",
    "No Conflicts",
    "No Defaults",
    "No Third Party Beneficiaries",
    "Notices",
    "Other Agreements",
    "Participations",
    "Payments",
    "Powers",
    "Public Announcements",
    "Purchases",
    "Qualifications",
    "Real Properties",
    "Record Dates",
    "Records",
    "Releases",
    "Remedies",
    "Representations",
    "Resignations",
    "Sanctions",
    "Severability",
    "Solvency",
    "Specific Performance",
    "Submission To Jurisdiction",
    "Subsidiaries",
    "Successors",
    "Survival",
    "Tax",
    "Taxes",
    "Terminations",
    "Terms",
    "Titles",
    "Transactions With Affiliates",
    "Transfer Taxes",
    "Transfers",
    "Trustees",
    "Use Of Proceeds",
    "Vacancies",
    "Venues",
    "Vesting",
    "Waiver Of Jury Trials",
    "Waivers",
]

ID2LABEL = {idx: name for idx, name in enumerate(LEDGAR_PROVISION_CLASSES)}
LABEL2ID = {name: idx for idx, name in enumerate(LEDGAR_PROVISION_CLASSES)}

SYSTEM_PROMPT = (
    "You are an expert legal AI assistant. Classify the following contract provision into "
    "exactly one of the 100 standard LEDGAR provision categories. Respond with only the exact category name."
)


def format_instruction_prompt(provision_text: str, category_name: str | None = None) -> str:
    """
    Format a legal contract provision into Phi-3 chat template syntax.

    If category_name is provided, formats complete conversation for training.
    If category_name is None, formats prompt up to the assistant response prefix for inference.
    """
    cleaned_text = provision_text.strip()

    prompt = (
        f"<|user|>\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"Contract Provision:\n"
        f'"{cleaned_text}"\n<|end|>\n'
        f"<|assistant|>\n"
    )

    if category_name is not None:
        prompt += f"{category_name.strip()}<|end|>"

    return prompt


def prepare_ledgar_dataset(
    split: str | None = None,
    dataset_name: str = "coastalcph/lex_glue",
    dataset_subset: str = "ledgar",
) -> DatasetDict | Dataset:
    """
    Load and preprocess the LexGLUE LEDGAR dataset with instruction formatting.

    Adds:
    - 'formatted_text': Full instruction prompt with assistant target label
    - 'prompt_only': Prompt without the assistant response (for evaluation)
    - 'category_name': String label name
    - 'class_id': Integer label ID (0-99)
    """
    if split is not None:
        ds = load_dataset(dataset_name, dataset_subset, split=split)
        return _apply_formatting(ds)

    raw_datasets = load_dataset(dataset_name, dataset_subset)
    formatted_dict = {}
    for s in raw_datasets:
        formatted_dict[s] = _apply_formatting(raw_datasets[s])
    return DatasetDict(formatted_dict)


def _apply_formatting(dataset: Dataset) -> Dataset:
    """Map formatting onto dataset partition."""
    label_names = (
        dataset.features["label"].names if "label" in dataset.features else LEDGAR_PROVISION_CLASSES
    )

    def _map_fn(example: dict[str, Any]) -> dict[str, Any]:
        label_id = example["label"]
        category_name = label_names[label_id]
        return {
            "formatted_text": format_instruction_prompt(example["text"], category_name),
            "prompt_only": format_instruction_prompt(example["text"], category_name=None),
            "category_name": category_name,
            "class_id": label_id,
        }

    return dataset.map(_map_fn, desc="Formatting instruction prompts")
