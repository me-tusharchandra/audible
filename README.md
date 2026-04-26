# Audible

A self-improving environment for training ambient-listening gating classifiers.

**Hackathon:** OpenEnv Hackathon (India 2026), Theme #4 — Self-Improvement.

## The problem

Always-on listening agents need to decide *when to act* on what they hear, not just respond when summoned by a wake word. The hard cases aren't "Hey Siri" — they're ambient utterances like "I wonder what the weather's like" or "It's cold in here", multi-speaker chatter, and rhetorical questions. Existing wake-word + intent-classification stacks are brittle on these.

## What this is

An OpenEnv environment where an adversarial generator and a gating classifier co-evolve. The generator produces increasingly subtle ambient utterances; the gate (a small edge-deployable model) learns to classify them into `IGNORE / UPDATE_CONTEXT / ACT(<tool>)`. Personalization layer adapts the gate to per-user preference profiles.

## Status

In development. Detailed plan, results, and demo links coming soon.
