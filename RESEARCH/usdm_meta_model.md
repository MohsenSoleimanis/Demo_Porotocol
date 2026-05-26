# USDM v4 Meta-Model — Canonical Spec for the FMLS KG

**Source of truth:** `pip install usdm` (currently `usdm-0.67.0`). All Pydantic models live under `usdm_model.*`. We do not reimplement them.

**Verified:** 2026-05-22 — inspected the installed package directly. **100 classes total** = 88 domain classes + 7 abstract `ApiBaseModel*` bases + 5 `Extension*` / `Base*` extension-framework types.

This document is the canonical reference. Whenever the spec is ambiguous, **inspect the installed `usdm_model.<class>.model_fields` first** — that's ground truth.

---

## Architectural facts that matter (caught from field inspection)

These shape every extraction decision downstream:

1. **Top-down containment hierarchy:**
   ```
   Wrapper
     └── Study
          └── versions: List[StudyVersion]
               └── studyDesigns: List[InterventionalStudyDesign | ObservationalStudyDesign]
                    └── arms, activities, encounters, objectives, eligibilityCriteria,
                        scheduleTimelines, estimands, indications, studyCells, epochs,
                        elements, population (single), analysisPopulations,
                        biospecimenRetentions
   ```

2. **StudyVersion is the heavyweight container.** These collections live at StudyVersion level (NOT StudyDesign), and are referenced by ID from StudyDesign:
   - `eligibilityCriterionItems` (the actual criterion *texts*)
   - `studyInterventions`
   - `administrableProducts`
   - `medicalDevices`
   - `biomedicalConcepts`, `bcCategories`, `bcSurrogates`
   - `conditions`, `abbreviations`, `dictionaries`
   - `organizations`, `roles`
   - `studyIdentifiers`, `referenceIdentifiers`
   - `titles`, `narrativeContentItems`
   - `amendments`, `dateValues`, `notes`

3. **Normalization pattern — `EligibilityCriterion` vs `EligibilityCriterionItem`:**
   - `EligibilityCriterionItem` holds the *actual text* (e.g., "Patients must have ECOG ≤ 2"). Lives on `StudyVersion.eligibilityCriterionItems[]`. Has `id`, `text`, `dictionaryId`.
   - `EligibilityCriterion` is a *reference* (e.g., "this is exclusion criterion #7 in this design"). Lives on `StudyDesign.eligibilityCriteria[]`. Has `id`, `category` (inclusion/exclusion via Code), `criterionItemId` (pointer to the Item), `nextId`/`previousId` (ordering).
   - **Why:** the same criterion text can be reused across designs without duplicating the text.

4. **Objective → Endpoint is containment, not reference.** `Objective.endpoints: List[Endpoint]` — endpoints live INSIDE their objective. There's no separate flat endpoint list at design level.

5. **Arm → Intervention is INDIRECT.** `StudyArm` does NOT have an `interventionId`. The arm-intervention mapping is implicit through `StudyCell` (arm × epoch × element) — the Cell ties an Arm to a StudyElement which references a StudyIntervention. This is the CDISC "Element ↔ Arm" matrix model.

6. **Dosing lives inside `StudyIntervention.administrations: List[Administration]`.** Not its own top-level entity.

7. **Procedures live inside `Activity.definedProcedures: List[Procedure]`.** Not at design level.

8. **SoA structure:** `ScheduleTimeline.instances: List[ScheduledActivityInstance | ScheduledDecisionInstance]`. Each `ScheduledActivityInstance` has `activityIds[]`, `epochId`, `encounterId`, `timelineExitId`, etc. The SoA matrix is built by traversing these.

9. **Every class has** `extensionAttributes: List[ExtensionAttribute]` (custom fields), `notes: List[CommentAnnotation]` (annotations on any class), `instanceType: Literal[ClassName]` (discriminator for polymorphic JSON).

10. **IDs are strings throughout** (except `Study.id` which is `Optional[uuid.UUID]`). Reference fields end in `Id` (singular) or `Ids` (list).

---

## The 100 classes, grouped by domain

### Envelope / Root (5)
- `Wrapper` — outermost JSON envelope
- `Study` — study root; contains `versions[]` + `documentedBy[]`
- `StudyVersion` — one protocol version; heavyweight container of all study-level entities
- `StudyDefinitionDocument` — the protocol document object
- `StudyDefinitionDocumentVersion` — versioned document

### Design (8)
- `StudyDesign` (abstract — never instantiated directly)
- `InterventionalStudyDesign` ✓ — interventional studies
- `ObservationalStudyDesign` ✓ — observational studies
- `StudyArm`
- `StudyCell` — arm × epoch matrix cell
- `StudyEpoch`
- `StudyElement` — what happens in a cell
- `TransitionRule` — Encounter/Element entry/exit rules

### Population (5)
- `StudyDesignPopulation` (single per design; has `cohorts[]`)
- `StudyCohort`
- `PopulationDefinition` (base type used by both)
- `Characteristic` — demographic/clinical population descriptors
- `SubjectEnrollment`

### Eligibility (2)
- `EligibilityCriterion` (reference: id + category + `criterionItemId` + ordering)
- `EligibilityCriterionItem` (text + dictionaryId; lives at StudyVersion level)

### Objectives / Endpoints / Estimands (3)
- `Objective` (has `endpoints[]` inside)
- `Endpoint` (purpose + level)
- `Estimand` — statistical estimand
- `IntercurrentEvent`

### Interventions / Products / Substances (8)
- `StudyIntervention` (role + type + administrations[])
- `AdministrableProduct` (dose form + sourcing + ingredients[])
- `AdministrableProductProperty`
- `Administration` (dose + route + frequency inside StudyIntervention)
- `Substance`
- `Ingredient`
- `Strength`
- `ProductOrganizationRole`
- `MedicalDevice`

### Activities / Procedures (5)
- `Activity` (has `definedProcedures[]`, `biomedicalConceptIds[]`, `timelineId`)
- `Procedure`
- `ScheduledInstance` (abstract)
- `ScheduledActivityInstance` ✓
- `ScheduledDecisionInstance` ✓
- `ConditionAssignment` (for ScheduledDecisionInstance branches)

### Schedule / Timing (4)
- `ScheduleTimeline` (instances[] + timings[] + entry + exits[])
- `ScheduleTimelineExit`
- `Timing`
- `Duration`

### Encounters (1)
- `Encounter` (visit; tied to StudyDesign via `encounters[]`)

### Biomedical Concepts (5)
- `BiomedicalConcept` (measurable concept linked to controlled term)
- `BiomedicalConceptCategory`
- `BiomedicalConceptProperty`
- `BiomedicalConceptSurrogate` (alternative form)
- `BiospecimenRetention`

### Conditions (1)
- `Condition` — clinical condition (referenced by ID from activities, eligibility, etc.)

### Statistical Analysis (1)
- `AnalysisPopulation`

### Organizations / People / Sites (6)
- `Organization`
- `StudySite`
- `GeographicScope`
- `Address`
- `StudyRole`
- `AssignedPerson`
- `PersonName`

### Amendments (4)
- `StudyAmendment` (at StudyVersion level)
- `StudyAmendmentImpact`
- `StudyAmendmentReason`
- `StudyChange`

### Documents / Narrative / Comments (4)
- `DocumentContentReference`
- `NarrativeContent`
- `NarrativeContentItem` (lives at StudyVersion level)
- `CommentAnnotation`

### Syntax templates (3)
- `SyntaxTemplate`
- `SyntaxTemplateDictionary`
- `ParameterMap`

### Value types (used inside other classes) (10)
- `Code` (code + codeSystem + decode)
- `AliasCode` (Code with alternative aliases)
- `ResponseCode`
- `Quantity` (value + unit)
- `Range` (min + max + unit)
- `QuantityRange` (union holder)
- `Identifier` (base)
- `StudyIdentifier`
- `ReferenceIdentifier`
- `AdministrableProductIdentifier`
- `MedicalDeviceIdentifier`
- `GovernanceDate`
- `Abbreviation`
- `Indication`
- `Masking`

### Extension framework (5)
- `Extension`
- `ExtensionClass`
- `ExtensionAttribute`
- `BaseCode`, `BaseAliasCode`, `BaseQuantity`, `BaseRange` — extension base types

### Abstract base classes (skip — not instantiated) (8)
- `ApiBaseModel`
- `ApiBaseModelWithId`
- `ApiBaseModelWithIdAndDesc`
- `ApiBaseModelWithIdAndName`
- `ApiBaseModelWithIdNameAndDesc`
- `ApiBaseModelWithIdNameAndLabel`
- `ApiBaseModelWithIdNameLabelAndDesc`
- `ApiBaseModelWithIdOnly`

---

## The relationship graph (where extracted entities point to each other)

```
Wrapper
  study: Study
            ├── versions[]: StudyVersion
            │              ├── studyDesigns[]: InterventionalStudyDesign | ObservationalStudyDesign
            │              │                  ├── arms[]: StudyArm
            │              │                  │         └── populationIds[]: ref → StudyDesignPopulation.id
            │              │                  ├── studyCells[]: StudyCell (arm × epoch × element)
            │              │                  ├── epochs[]: StudyEpoch
            │              │                  ├── elements[]: StudyElement
            │              │                  │             └── interventionId: ref → StudyIntervention.id
            │              │                  ├── studyInterventionIds[]: refs → StudyIntervention.id
            │              │                  ├── encounters[]: Encounter
            │              │                  ├── activities[]: Activity
            │              │                  │             ├── definedProcedures[]: Procedure
            │              │                  │             ├── biomedicalConceptIds[]: refs → BiomedicalConcept.id
            │              │                  │             ├── bcCategoryIds[]: refs → BiomedicalConceptCategory.id
            │              │                  │             ├── bcSurrogateIds[]: refs → BiomedicalConceptSurrogate.id
            │              │                  │             └── timelineId: ref → ScheduleTimeline.id
            │              │                  ├── scheduleTimelines[]: ScheduleTimeline
            │              │                  │                       ├── instances[]: ScheduledActivityInstance | ScheduledDecisionInstance
            │              │                  │                       │   (activityIds[], encounterId, epochId, timelineExitId)
            │              │                  │                       ├── timings[]: Timing
            │              │                  │                       └── exits[]: ScheduleTimelineExit
            │              │                  ├── objectives[]: Objective
            │              │                  │              └── endpoints[]: Endpoint  (inline, not refs)
            │              │                  ├── estimands[]: Estimand
            │              │                  ├── indications[]: Indication
            │              │                  ├── eligibilityCriteria[]: EligibilityCriterion
            │              │                  │                       └── criterionItemId: ref → EligibilityCriterionItem.id
            │              │                  ├── population: StudyDesignPopulation
            │              │                  │             ├── cohorts[]: StudyCohort
            │              │                  │             └── criterionIds[]: refs → EligibilityCriterion.id
            │              │                  ├── analysisPopulations[]: AnalysisPopulation
            │              │                  └── biospecimenRetentions[]: BiospecimenRetention
            │              │
            │              ├── eligibilityCriterionItems[]: EligibilityCriterionItem  (texts)
            │              ├── studyInterventions[]: StudyIntervention
            │              │                       └── administrations[]: Administration (dose, route, freq, duration)
            │              ├── administrableProducts[]: AdministrableProduct
            │              │                         ├── ingredients[]: Ingredient
            │              │                         │              └── substanceIds[]: refs → Substance
            │              │                         └── identifiers[]: AdministrableProductIdentifier
            │              ├── medicalDevices[]: MedicalDevice
            │              ├── biomedicalConcepts[]: BiomedicalConcept
            │              │                       └── properties[]: BiomedicalConceptProperty
            │              ├── bcCategories[]: BiomedicalConceptCategory
            │              ├── bcSurrogates[]: BiomedicalConceptSurrogate
            │              ├── conditions[]: Condition
            │              ├── abbreviations[]: Abbreviation
            │              ├── narrativeContentItems[]: NarrativeContentItem
            │              ├── organizations[]: Organization
            │              ├── productOrganizationRoles[]: ProductOrganizationRole
            │              ├── roles[]: StudyRole
            │              │         └── assignedPersons[]: AssignedPerson
            │              ├── studyIdentifiers[]: StudyIdentifier
            │              ├── referenceIdentifiers[]: ReferenceIdentifier
            │              ├── titles[]: StudyTitle
            │              ├── amendments[]: StudyAmendment
            │              │                ├── reasons[]: StudyAmendmentReason
            │              │                ├── impacts[]: StudyAmendmentImpact
            │              │                └── changes[]: StudyChange
            │              ├── dateValues[]: GovernanceDate
            │              ├── dictionaries[]: SyntaxTemplateDictionary
            │              └── notes[]: CommentAnnotation
            └── documentedBy[]: StudyDefinitionDocument
                              └── versions[]: StudyDefinitionDocumentVersion
                                            └── contents[]: NarrativeContent (refs NarrativeContentItem)
```

---

## Our extension: Provenance for every extracted instance

USDM has no provenance fields. We add them via a generic wrapper. Every extracted record (regardless of which of the 100 classes it is) is wrapped in:

```python
class Provenance(BaseModel):
    evidence_chunk_id: str
    evidence_quote: str                  # verbatim substring of source chunk text
    evidence_page: int
    evidence_bbox: tuple[float, float, float, float] | None
    extractor: str                       # e.g. "o4-mini"
    extractor_version: str               # e.g. "o4-mini-2025-04-16"
    prompt_hash: str                     # sha256 of the prompt template
    schema_version: str                  # e.g. "usdm-0.67.0"
    extraction_confidence: float
    extracted_at: datetime

class Extracted(BaseModel, Generic[T]):
    value: T               # any usdm_model.* class
    provenance: Provenance
```

`Extracted[EligibilityCriterionItem]`, `Extracted[Objective]`, `Extracted[StudyIntervention]`, etc. — same wrapper for all 100 classes.

---

## Extraction-target priority (incremental, not part of meta-model scope)

Meta-model commitment: **all 100 classes are part of the schema.** What follows is the *extraction* roadmap — which classes we actually populate from PDF text first.

**Tier 1 — extract first (gives you a queryable skeleton KG):**
- `Study`, `StudyVersion` (one of each per protocol)
- `StudyIdentifier`, `StudyTitle`
- `InterventionalStudyDesign` (or `ObservationalStudyDesign`)
- `StudyArm`
- `StudyDesignPopulation`
- `EligibilityCriterionItem` (text) + `EligibilityCriterion` (reference)
- `Objective` + nested `Endpoint`
- `StudyIntervention` + nested `Administration`
- `AdministrableProduct`
- `Activity` (with nested `Procedure`)
- `Encounter`
- `ScheduleTimeline` + `ScheduledActivityInstance`

**Tier 2 — extract next:**
- `Condition`, `Indication`
- `Estimand`, `IntercurrentEvent`
- `StudyCell`, `StudyEpoch`, `StudyElement` (the arm × epoch × element matrix)
- `BiomedicalConcept`, `BiomedicalConceptCategory`, `BiomedicalConceptProperty`
- `Substance`, `Ingredient`, `Strength`
- `MedicalDevice`
- `AnalysisPopulation`
- `StudyCohort`

**Tier 3 — extract later:**
- `Organization`, `StudySite`, `Address`, `StudyRole`, `AssignedPerson`, `PersonName`, `GeographicScope`
- `StudyAmendment` family (when amendments exist)
- `Abbreviation`, `NarrativeContentItem`, `CommentAnnotation`
- `Masking`, `BiospecimenRetention`, `SubjectEnrollment`
- `Characteristic`
- All `Identifier` subtypes beyond `StudyIdentifier`

**Tier 4 — derived or always-present (don't extract, construct):**
- `Wrapper` (constructed at output time)
- `Code`, `AliasCode`, `ResponseCode`, `Quantity`, `Range`, `QuantityRange`, `Duration`, `Timing`, `GovernanceDate` (value types embedded in other extractions)
- `Extension`, `ExtensionAttribute`, `ExtensionClass`, `Base*` (extension framework)
- All `ApiBaseModel*` abstract bases
- `SyntaxTemplate`, `SyntaxTemplateDictionary`, `ParameterMap` (template framework)
- `DocumentContentReference`, `NarrativeContent` (document structure)
- `TransitionRule`, `ConditionAssignment` (rule structures)
- `ScheduleTimelineExit` (auto-derived from timeline parsing)
- `ScheduledDecisionInstance` (branching decisions — rare in protocols)

---

## Conformance validation

CDISC ships `cdisc-rules-engine` for USDM JSON validation. We run it after assembly:

```bash
pip install cdisc-rules-engine
```

Produces a conformance report. Our pipeline must pass this report for any USDM JSON we emit.

---

## What this commits us to

1. **Source of truth = `usdm_model.*` Pydantic classes** (currently `usdm-0.67.0`).
2. **Wrapper for provenance = `Extracted[T]` generic** — one mixin, applies to all 100 classes.
3. **Output JSON = standard USDM v4 Wrapper(study=Study(...))** with provenance attached out-of-band (in `extensionAttributes` or a side-car file — TBD; doesn't affect USDM conformance).
4. **Conformance = pass `cdisc-rules-engine` validation**.
5. **Extraction = incremental** per the tier roadmap above. Meta-model coverage = full v4. Extraction coverage = grows over time.

When a new USDM version ships (v4.1, v5), bump the `usdm` package version, re-inspect, regenerate this doc. No reimplementation of models.
