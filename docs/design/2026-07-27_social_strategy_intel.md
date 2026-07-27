# Design — Social Strategy Intel (X + YouTube + CN best-effort)

**Date:** 2026-07-27  
**Module:** research intel (not paper freeze / not live capital)  
**Status:** implemented v0 (`trad_research/social_intel`)

## Problem

Creators on X/YouTube claim “winning strategies”. We need a **bounded**, reproducible way to sample those claims, pull transcripts, and score them with **TRAD honesty gates** — without pretending to scrape all of X/Weibo.

## Solution

- Package `trad_research/social_intel` with:
  - date window (default 2026-04-27 → 2026-07-27)
  - YouTube search/meta via **yt-dlp**
  - transcripts via **youtube-transcript-api** + yt-dlp subs fallback
  - heuristic rule/claim extraction (LLM optional later)
  - deterministic G1–G6 rubric
  - batch artifacts under `reports/social_intel/BATCH_*/`
- LangGraph-style sequential pipeline in `pipeline.py` (no hard CrewAI dependency at runtime).
- China/Weibo: attach external best-effort JSON; log coverage gaps.

## Non-goals

- Full firehose scrape
- Frame-by-frame video vision
- Promoting strategies as alpha
- Changing `turbo_highvol_minalloc` freeze

## CLI

```powershell
python -m trad_research.social_intel run --max-videos 20 --batch-id 20260727
```

## Verification

```powershell
python -m pytest tests/test_social_intel_unit.py -q
```
