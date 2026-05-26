"""FMLS extraction layer — USDM v4 meta-model + Provenance wrapper.

Meta-model source: `pip install usdm` (currently 0.67.0). Pydantic models
live under `usdm_model.*`. We do NOT reimplement them.

This module:
  1. Re-exports the USDM classes you'll touch most often (convenience).
  2. Defines `Provenance` and the generic `Extracted[T]` wrapper.
  3. Defines a small helper for the discriminated-union output JSON.

For the full class inventory and architecture notes, see
`RESEARCH/usdm_meta_model.md`.

Adding new extraction targets:
  - No new file. Just import the USDM class and wrap with `Extracted[...]`.
  - e.g. `Extracted[MedicalDevice]`, `Extracted[BiomedicalConcept]`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# === Re-export the USDM classes we'll touch most often.
# Source of truth: pip install usdm. Full list in RESEARCH/usdm_meta_model.md.
from usdm_model.wrapper import Wrapper
from usdm_model.study import Study
from usdm_model.study_version import StudyVersion
from usdm_model.study_design import (
    StudyDesign,
    InterventionalStudyDesign,
    ObservationalStudyDesign,
)
from usdm_model.study_arm import StudyArm
from usdm_model.study_cell import StudyCell
from usdm_model.study_epoch import StudyEpoch
from usdm_model.study_element import StudyElement
from usdm_model.study_title import StudyTitle
from usdm_model.identifier import (
    Identifier,
    StudyIdentifier,
    ReferenceIdentifier,
    AdministrableProductIdentifier,
    MedicalDeviceIdentifier,
)
from usdm_model.eligibility_criterion import (
    EligibilityCriterion,
    EligibilityCriterionItem,
)
from usdm_model.population_definition import (
    PopulationDefinition,
    StudyDesignPopulation,
    StudyCohort,
)
from usdm_model.characteristic import Characteristic
from usdm_model.objective import Objective
from usdm_model.endpoint import Endpoint
from usdm_model.estimand import Estimand
from usdm_model.intercurrent_event import IntercurrentEvent
from usdm_model.study_intervention import StudyIntervention
from usdm_model.administrable_product import AdministrableProduct
from usdm_model.administration import Administration
from usdm_model.substance import Substance
from usdm_model.ingredient import Ingredient
from usdm_model.strength import Strength
from usdm_model.medical_device import MedicalDevice
from usdm_model.activity import Activity
from usdm_model.procedure import Procedure
from usdm_model.encounter import Encounter
from usdm_model.schedule_timeline import ScheduleTimeline
from usdm_model.schedule_timeline_exit import ScheduleTimelineExit
from usdm_model.scheduled_instance import (
    ScheduledInstance,
    ScheduledActivityInstance,
    ScheduledDecisionInstance,
    ConditionAssignment,
)
from usdm_model.timing import Timing
from usdm_model.duration import Duration
from usdm_model.biomedical_concept import BiomedicalConcept
from usdm_model.biomedical_concept_category import BiomedicalConceptCategory
from usdm_model.biomedical_concept_property import BiomedicalConceptProperty
from usdm_model.biomedical_concept_surrogate import BiomedicalConceptSurrogate
from usdm_model.biospecimen_retention import BiospecimenRetention
from usdm_model.condition import Condition
from usdm_model.indication import Indication
from usdm_model.analysis_population import AnalysisPopulation
from usdm_model.organization import Organization
from usdm_model.study_site import StudySite
from usdm_model.address import Address
from usdm_model.geographic_scope import GeographicScope
from usdm_model.study_role import StudyRole
from usdm_model.assigned_person import AssignedPerson
from usdm_model.person_name import PersonName
from usdm_model.study_amendment import StudyAmendment
from usdm_model.study_amendment_impact import StudyAmendmentImpact
from usdm_model.study_amendment_reason import StudyAmendmentReason
from usdm_model.study_change import StudyChange
from usdm_model.study_definition_document import StudyDefinitionDocument
from usdm_model.study_definition_document_version import StudyDefinitionDocumentVersion
from usdm_model.narrative_content import NarrativeContent, NarrativeContentItem
from usdm_model.comment_annotation import CommentAnnotation
from usdm_model.abbreviation import Abbreviation
from usdm_model.masking import Masking
from usdm_model.subject_enrollment import SubjectEnrollment
from usdm_model.product_organization_role import ProductOrganizationRole
from usdm_model.transition_rule import TransitionRule
from usdm_model.code import Code
from usdm_model.alias_code import AliasCode
from usdm_model.response_code import ResponseCode
from usdm_model.quantity_range import Quantity, Range, QuantityRange
from usdm_model.governance_date import GovernanceDate
from usdm_model.document_content_reference import DocumentContentReference
from usdm_model.syntax_template import SyntaxTemplate
from usdm_model.syntax_template_dictionary import (
    SyntaxTemplateDictionary,
    ParameterMap,
)


# === USDM package version this code was last validated against.
USDM_SCHEMA_VERSION = "usdm-0.67.0"


# === Provenance — our addition. Not in USDM. Mandatory on every extraction.

class Provenance(BaseModel):
    """Audit trail for every extracted USDM instance.

    Carries: which source bytes the extraction came from, which model
    produced it, which prompt, when, with what confidence. Non-nullable
    in production; missing provenance = rejected at the validation gate.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_chunk_id: str
    """The chunk_id from chunks.json that this extraction was drawn from."""

    evidence_quote: str
    """Verbatim substring of the source chunk text. Must pass substring
    validation against the chunk's text."""

    evidence_page: int = Field(ge=0)
    evidence_bbox: Optional[tuple[float, float, float, float]] = None
    """(x1, y1, x2, y2) on the source PDF page; for clickable citation."""

    extractor: str
    """Logical extractor name, e.g. 'o4-mini', 'gliner-bio', 'regex-xref-v1'."""

    extractor_version: str
    """Specific version, e.g. 'o4-mini-2025-04-16', 'gliner-large-bio-v0.1'."""

    prompt_hash: Optional[str] = None
    """SHA-256 of the prompt template (LLM extractors only)."""

    ruleset_version: Optional[str] = None
    """Ruleset version (rule-based extractors only, e.g. NegEx)."""

    schema_version: str = USDM_SCHEMA_VERSION
    """USDM package version this extraction was validated against."""

    extraction_confidence: float = Field(ge=0.0, le=1.0)
    extracted_at: datetime
    reviewer: Optional[str] = None
    """Email/id of human reviewer if this extraction was reviewed."""

    validation_status: str = "pending"
    """One of: pending | passed | failed | reviewed."""


# === Extracted[T] — the universal extraction wrapper.

T = TypeVar("T", bound=BaseModel)


class Extracted(BaseModel, Generic[T]):
    """Universal wrapper: any USDM class + mandatory provenance.

    Usage:
        Extracted[EligibilityCriterion](
            value=EligibilityCriterion(...),
            provenance=Provenance(...),
        )

    Same shape for all 100 USDM classes. No per-class wrapper code.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    value: T
    provenance: Provenance


# === Convenience: type aliases for the v1 Tier-1 extraction targets.
# These are NOT new types — just naming the common Extracted[T]
# specializations for clarity in extraction code.

ExtractedStudy = Extracted[Study]
ExtractedStudyVersion = Extracted[StudyVersion]
ExtractedStudyDesign = Extracted[InterventionalStudyDesign]
ExtractedStudyArm = Extracted[StudyArm]
ExtractedStudyDesignPopulation = Extracted[StudyDesignPopulation]
ExtractedEligibilityCriterionItem = Extracted[EligibilityCriterionItem]
ExtractedEligibilityCriterion = Extracted[EligibilityCriterion]
ExtractedObjective = Extracted[Objective]
ExtractedEndpoint = Extracted[Endpoint]
ExtractedStudyIntervention = Extracted[StudyIntervention]
ExtractedAdministrableProduct = Extracted[AdministrableProduct]
ExtractedActivity = Extracted[Activity]
ExtractedEncounter = Extracted[Encounter]
ExtractedScheduleTimeline = Extracted[ScheduleTimeline]
ExtractedScheduledActivityInstance = Extracted[ScheduledActivityInstance]
ExtractedStudyIdentifier = Extracted[StudyIdentifier]
ExtractedStudyTitle = Extracted[StudyTitle]
ExtractedCondition = Extracted[Condition]
ExtractedIndication = Extracted[Indication]


__all__ = [
    "USDM_SCHEMA_VERSION",
    "Provenance",
    "Extracted",
    # USDM classes re-exported (full list — see RESEARCH/usdm_meta_model.md)
    "Wrapper", "Study", "StudyVersion",
    "StudyDesign", "InterventionalStudyDesign", "ObservationalStudyDesign",
    "StudyArm", "StudyCell", "StudyEpoch", "StudyElement", "StudyTitle",
    "Identifier", "StudyIdentifier", "ReferenceIdentifier",
    "AdministrableProductIdentifier", "MedicalDeviceIdentifier",
    "EligibilityCriterion", "EligibilityCriterionItem",
    "PopulationDefinition", "StudyDesignPopulation", "StudyCohort",
    "Characteristic",
    "Objective", "Endpoint", "Estimand", "IntercurrentEvent",
    "StudyIntervention", "AdministrableProduct", "Administration",
    "Substance", "Ingredient", "Strength", "MedicalDevice",
    "Activity", "Procedure", "Encounter",
    "ScheduleTimeline", "ScheduleTimelineExit", "Timing", "Duration",
    "ScheduledInstance", "ScheduledActivityInstance",
    "ScheduledDecisionInstance", "ConditionAssignment",
    "BiomedicalConcept", "BiomedicalConceptCategory",
    "BiomedicalConceptProperty", "BiomedicalConceptSurrogate",
    "BiospecimenRetention",
    "Condition", "Indication",
    "AnalysisPopulation",
    "Organization", "StudySite", "Address", "GeographicScope",
    "StudyRole", "AssignedPerson", "PersonName",
    "StudyAmendment", "StudyAmendmentImpact", "StudyAmendmentReason",
    "StudyChange",
    "StudyDefinitionDocument", "StudyDefinitionDocumentVersion",
    "NarrativeContent", "NarrativeContentItem", "CommentAnnotation",
    "Abbreviation", "Masking", "SubjectEnrollment",
    "ProductOrganizationRole", "TransitionRule",
    "Code", "AliasCode", "ResponseCode",
    "Quantity", "Range", "QuantityRange",
    "GovernanceDate", "DocumentContentReference",
    "SyntaxTemplate", "SyntaxTemplateDictionary", "ParameterMap",
    # Convenience aliases
    "ExtractedStudy", "ExtractedStudyVersion", "ExtractedStudyDesign",
    "ExtractedStudyArm", "ExtractedStudyDesignPopulation",
    "ExtractedEligibilityCriterionItem", "ExtractedEligibilityCriterion",
    "ExtractedObjective", "ExtractedEndpoint",
    "ExtractedStudyIntervention", "ExtractedAdministrableProduct",
    "ExtractedActivity", "ExtractedEncounter",
    "ExtractedScheduleTimeline", "ExtractedScheduledActivityInstance",
    "ExtractedStudyIdentifier", "ExtractedStudyTitle",
    "ExtractedCondition", "ExtractedIndication",
]
