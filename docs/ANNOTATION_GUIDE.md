# Annotation guide (CVAT point tracks)

## Setup

- Task type: video, "CVAT for video 1.1" format, one **points** track per object.
- Labels (exactly these names): `unknown`, `near_field_particle`, `payload_candidate`, `vehicle_feature`, `background_feature`, `artifact`.
- Attributes on each point: `visibility` = clear | faint | ambiguous. Track-level text attributes `identity` and `identity_source` are optional.
- Start from pseudo-labels (`rso export-cvat`) and correct them; do not trust interpolated frames without looking.

## What to mark

- Every visible moving or hovering point or small blob in the unmasked area, on every frame it is visible (dense on evaluation clips).
- The centre of brightness, not the edge. For a defocused disc, the disc centre.
- End the track (outside) when the object leaves, is hidden by the ship or HUD, or becomes indistinguishable.

## Category rules (when unsure, use `unknown`)

| Label | Use only when |
|---|---|
| `near_field_particle` | clearly defocused disc, or very fast, or tumbling/flickering, or obviously shed from the ship |
| `payload_candidate` | a resolved object emerging from the payload door during the deployment window and drifting away steadily |
| `vehicle_feature` | fixed relative to the camera (glint, tile edge, antenna) |
| `background_feature` | moves exactly with the Earth (cloud puff, sun glint on water) |
| `artifact` | compression blocks, HUD remnants, interlacing |
| `unknown` | anything else, including point-like objects you cannot explain |

Never label a dot as a named satellite from appearance. `identity` is filled only from independent evidence (for example an operator release list), and `identity_source` says what that evidence is.

## Visibility

- `clear`: you would bet on the position within 2 px.
- `faint`: present but position uncertain (counts as a positive with wider tolerance).
- `ambiguous`: might not be an object at all; becomes an *ignore* region in training and is excluded from scoring.

## Splits

- Annotate whole clips; the dataset tool splits by time blocks with gaps, so do not mix frames of one physical track across clips.
- Keep at least one clip (different camera view if possible) as the locked test set. Do not tune thresholds on it.
