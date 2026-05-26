"""Clinical-domain rule table — the ONE file with ICH-M11 / USDM knowledge.

This is the only file in the pipeline that knows about ICH M11 section
numbering or USDM class names. Everything else reads from these mappings.

To swap to another vertical (legal/industrial/financial), replace this
file with the equivalent domain mapping — the rest of the pipeline doesn't
change.

Maintained against:
  - ICH M11 final guideline (effective May 2026)
  - USDM v4 (cdisc-org/usdm 0.67.0)
"""

from __future__ import annotations

# === ICH M11 section → functional label
# These are universal across any ICH-M11-compliant clinical protocol.
# Format: section_id_prefix -> functional_label
# Use longest-prefix match (so "5.1" wins over "5").
ICH_M11_TO_FUNCTIONAL_LABEL: dict[str, str] = {
    # §1 — Synopsis / general info
    "1": "synopsis",
    "1.1": "synopsis",
    "1.2": "synopsis",
    "1.3": "schedule_of_activities",
    # §2 — Introduction (background)
    "2": "background",
    # §3 — Trial design
    "3": "study_design",
    "3.1": "synopsis_design",
    "3.2": "objectives_endpoints",
    "3.3": "design_arms",
    "3.4": "design_arms",  # number of subjects + arm allocation
    "3.5": "design_arms",  # masking/blinding
    "3.6": "design_timing",
    # §4 — Trial population
    "4": "study_population",
    "4.1": "study_population",
    "4.2": "population_rationale",
    # §5 — Eligibility
    "5": "eligibility",
    "5.1": "eligibility_inclusion",
    "5.2": "eligibility_exclusion",
    "5.3": "withdrawal_criteria",
    # §6 — Trial intervention
    "6": "intervention",
    "6.1": "intervention",
    "6.2": "dosing",
    "6.3": "dose_modification",
    "6.4": "concomitant_therapy",
    # §7 — Statistical analysis
    "7": "statistical_analysis",
    # §8 — Trial assessments and procedures
    "8": "activities_procedures",
    "8.1": "activities_procedures",
    "8.2": "activities_procedures",
    "8.3": "biomarkers",
    # §9 — Safety
    "9": "safety_assessments",
    "9.1": "adverse_event_reporting",
    "9.2": "safety_assessments",
    # §10 — General considerations (ethics, governance)
    "10": "governance",
    # §11 — Quality assurance / monitoring
    "11": "quality_monitoring",
}


# === Functional label → list of likely USDM target classes
# These are HINTS, not bindings. The extractor uses these to know which
# USDM Pydantic class to target on a given chunk. Multiple classes allowed
# because one chunk often produces several record types.
FUNCTIONAL_LABEL_TO_USDM_CLASSES: dict[str, list[str]] = {
    "synopsis": ["Study", "StudyVersion", "StudyTitle", "StudyIdentifier", "Indication"],
    "synopsis_design": ["InterventionalStudyDesign", "StudyArm", "StudyDesignPopulation"],
    "schedule_of_activities": [
        "ScheduleTimeline",
        "Encounter",
        "ScheduledActivityInstance",
        "Activity",
    ],
    "background": [],  # narrative, no extraction
    "study_design": [
        "InterventionalStudyDesign",
        "StudyArm",
        "StudyCell",
        "StudyEpoch",
        "StudyElement",
        "Masking",
    ],
    "objectives_endpoints": ["Objective", "Endpoint", "Estimand"],
    "design_arms": ["StudyArm", "StudyCell", "StudyEpoch", "Masking"],
    "design_timing": ["StudyEpoch", "Duration"],
    "study_population": ["StudyDesignPopulation", "StudyCohort", "Characteristic"],
    "population_rationale": ["StudyDesignPopulation"],
    "eligibility": ["EligibilityCriterion", "EligibilityCriterionItem"],
    "eligibility_inclusion": ["EligibilityCriterion", "EligibilityCriterionItem"],
    "eligibility_exclusion": ["EligibilityCriterion", "EligibilityCriterionItem"],
    "withdrawal_criteria": ["EligibilityCriterion", "EligibilityCriterionItem"],
    "intervention": [
        "StudyIntervention",
        "AdministrableProduct",
        "Substance",
        "Ingredient",
        "MedicalDevice",
    ],
    "dosing": ["StudyIntervention", "Administration", "Strength"],
    "dose_modification": ["StudyIntervention", "Administration"],
    "concomitant_therapy": ["StudyIntervention"],
    "statistical_analysis": [
        "AnalysisPopulation",
        "Estimand",
    ],
    "activities_procedures": ["Activity", "Procedure", "Encounter", "BiomedicalConcept"],
    "biomarkers": ["BiomedicalConcept", "BiomedicalConceptCategory", "BiospecimenRetention"],
    "safety_assessments": ["Activity", "Procedure", "Condition"],
    "adverse_event_reporting": ["Condition"],  # USDM AE definitions live as Conditions
    "governance": ["StudyRole", "Organization", "StudySite"],
    "quality_monitoring": [],  # process; not typically extracted
}


# === Functional label → role hints (free-form, for prompt priming)
FUNCTIONAL_LABEL_TO_ROLE_HINTS: dict[str, list[str]] = {
    "synopsis": ["study_metadata", "title", "identifier"],
    "synopsis_design": ["design_overview"],
    "schedule_of_activities": ["soa_row", "visit_schedule"],
    "objectives_endpoints": ["primary_endpoint", "secondary_endpoint", "objective"],
    "eligibility_inclusion": ["inclusion_criterion"],
    "eligibility_exclusion": ["exclusion_criterion"],
    "withdrawal_criteria": ["withdrawal_criterion"],
    "intervention": ["investigational_product"],
    "dosing": ["dose_regimen", "administration"],
    "dose_modification": ["dose_modification"],
    "concomitant_therapy": ["concomitant_medication"],
    "study_population": ["target_population", "demographics"],
    "design_arms": ["arm_definition"],
    "activities_procedures": ["procedure", "assessment"],
    "biomarkers": ["biomarker", "biospecimen"],
    "safety_assessments": ["safety_assessment"],
    "adverse_event_reporting": ["ae_reporting_rule", "sae_definition"],
    "statistical_analysis": ["statistical_method", "estimand"],
}


def lookup_functional_label(m11_section: str | None) -> str | None:
    """Look up functional label for a chunk by its m11_section.

    Uses longest-prefix match: '5.1' wins over '5'; '5.1.3' tries '5.1.3'
    first, then '5.1', then '5'. Returns None if no match.
    """
    if not m11_section:
        return None
    # Try the exact id and all parent prefixes, longest first
    parts = m11_section.split(".")
    while parts:
        prefix = ".".join(parts)
        if prefix in ICH_M11_TO_FUNCTIONAL_LABEL:
            return ICH_M11_TO_FUNCTIONAL_LABEL[prefix]
        parts.pop()
    return None


def usdm_class_hints(functional_label: str | None) -> list[str]:
    """USDM class names this functional label is likely to produce."""
    if not functional_label:
        return []
    return FUNCTIONAL_LABEL_TO_USDM_CLASSES.get(functional_label, [])


def role_hints(functional_label: str | None) -> list[str]:
    """Free-form role hints for prompt priming."""
    if not functional_label:
        return []
    return FUNCTIONAL_LABEL_TO_ROLE_HINTS.get(functional_label, [])


__all__ = [
    "ICH_M11_TO_FUNCTIONAL_LABEL",
    "FUNCTIONAL_LABEL_TO_USDM_CLASSES",
    "FUNCTIONAL_LABEL_TO_ROLE_HINTS",
    "lookup_functional_label",
    "usdm_class_hints",
    "role_hints",
]
