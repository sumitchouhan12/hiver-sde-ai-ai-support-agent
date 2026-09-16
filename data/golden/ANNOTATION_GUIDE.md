# Golden Set Human Review Guide

## Required
Review all 200 rows and replace the proposed fields with human labels.

Use exactly one intent:
- software_update_issue
- battery_power_charging
- connectivity_network
- account_access_security
- data_storage_backup
- apps_media_services
- device_hardware_issue
- billing_purchase_payment
- product_howto_information
- general_complaint_unclear

Use exactly one decision:
- AUTO_HANDLE
- ESCALATE

### Labeling principles
- Label the customer's primary issue, not the emotion.
- If multiple issues exist, choose the issue that blocks resolution; if genuinely ambiguous, use `general_complaint_unclear`.
- Account-specific and financial issues should normally be escalated.
- Safety-sensitive hardware issues should be escalated.
- Auto-handle only when the issue is clear, common, and can be answered safely from historical evidence.

### Sampling note
The 200 examples were sampled from AppleSupport-linked customer messages after URL/emoji/link cleanup and deduplication. The current labels are AI-assisted proposals for review, not gold labels.

### Adjudication
Have one primary annotator label all rows. A second annotator independently labels at least 40 rows. Resolve disagreements with a short adjudication note and report inter-annotator agreement.
