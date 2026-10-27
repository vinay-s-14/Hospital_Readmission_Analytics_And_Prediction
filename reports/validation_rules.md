# Validation Rules

| Rule | Action |
|---|---|
| encounter_id must exist | Reject |
| patient_nbr must exist | Reject |
| readmitted must exist | Reject |
| time_in_hospital must be present | Reject |
| Exact duplicate rows | Remove |
| `?` special missing code | Convert to missing |
| time_in_hospital outside 1–14 | Flag for review |
| readmitted not in {NO, >30, <30} | Flag/reject |
| numeric count fields non-numeric | Convert to missing and log |
| private/PII fields added | Prohibited |
