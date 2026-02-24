# SafeStan Integration Provenance

This document records the upstream provenance of the staged Stan reward
integration path.

## Source branch

- `origin/safestan-integration`

## Upstream commits integrated

- `f4fdcdc437096b7afdd11b5256d0cb403f65045b`
- `d76eb55cf7f9746c23f5810e668901cdef8a8879`
- `0d8c1f53e6ce762a6cde860f25b19ceaa79ad726`

## Repository-local compatibility edits

The integration was adjusted locally for this repository to:

- isolate the path as opt-in staged runtime,
- align imports and config flattening with existing Hydra groups,
- preserve fail-fast runtime validation rules,
- keep paper-track defaults unchanged.

## Attribution policy

Human author attribution is tracked in git metadata (commit author fields,
`cherry-pick -x` references, and `Co-authored-by` trailers when applicable),
not in runtime code comments.

