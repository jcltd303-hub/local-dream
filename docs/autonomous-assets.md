# Autonomous asset generation

Local Dream can run a resumable batch around the existing
`BackgroundGenerationService`. The autonomous layer does not replace or fork
the DreamLite/QNN inference path; it serializes normal generation requests,
persists progress and writes each result plus metadata.

## Plan contract

```json
{
  "run_id": "campaign-001",
  "model_id": "dreamlite-mobile",
  "backend_type": "dreamlite_litert",
  "max_retries": 2,
  "jobs": [
    {
      "id": "hero-01",
      "prompt": "editorial portrait, rooftop at blue hour",
      "negative_prompt": "",
      "seed": 12345,
      "width": 1024,
      "height": 1024,
      "steps": 4,
      "cfg": 1.0,
      "scheduler": "dpm",
      "reference_images": []
    }
  ]
}
```

`reference_images` uses the same base64 payload format already consumed by
the generation service. DreamLite Mobile still enforces its frozen 1024x1024
graph contract.

## Starting a run from app code

```kotlin
val plan = AutonomousAssetPlan.parse(planJson)
AutonomousAssetService.start(context, plan)
```

Call `AutonomousAssetService.resume(context)` to continue the most recently
persisted run after a process restart, or `stop(context)` to halt it.

## Persistence and outputs

State is checkpointed after every attempt under the app's private files
directory:

```
files/autonomous-assets/<run-id>/state.json
```

Successful images and per-asset metadata are written under:

```
Android/data/<package>/files/Pictures/LocalDreamAutonomous/<run-id>/
  <job-id>.png
  <job-id>.json
```

A completed job is never regenerated on resume. Failed jobs retry up to
`max_retries`, with short linear backoff, then remain recorded in the state
manifest for inspection.
