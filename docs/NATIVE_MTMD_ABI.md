# Native mtmd ABI Boundary

Orbit isolates the unstable upstream mtmd C ABI behind a revision-bound native
bridge:

```text
Python runtime
  -> liborbit-mtmd-bridge (primitive values and opaque handles)
  -> matching libmtmd/libllama build
```

Python does not declare `mtmd_context_params`, `mtmd_input_text`, bitmap return
wrappers, or capability structures. The bridge constructs these values using
the headers from the native revision being built and explicitly initializes
every field used by Orbit. The bridge currently recognizes the reviewed mtmd
context profiles `mtmd-context-v1` (b9551, 56 bytes), `mtmd-context-v2` (the
earlier upstream candidate with batching/progress fields, 80 bytes) and
`mtmd-context-v3` (the bundled 46f4da2 vendor: an explicit `device` handle
follows `use_gpu`, 96 bytes), the input-text profiles v1/v2 and both bitmap
result shapes (`pointer-v1`, `wrapper-v1`). Unknown layouts fail before mmproj
initialization.

## Build Identity

The bridge is built into the same directory as its native libraries. Its
sidecar identity records:

- compiler and compiler version;
- bridge flags and relevant CMake configuration;
- hashes of the bridge source and relevant llama/mtmd headers;
- hashes of every co-located runtime library;
- upstream tag and commit;
- canonical source-tree and Orbit patchset hashes.

The bridge artifact and every required runtime library are hashed again before
the bridge is loaded. A missing sidecar, changed artifact, changed library, or
provenance mismatch is a controlled error. Build reuse is forbidden when any
identity input changes.

The bundled source provenance is stored in
`src/orbit/native_llama/vendor/LLAMA_PROVENANCE.json`. CMake receives the
manifest commit and build number explicitly; it must not derive
`LLAMA_COMMIT` from the enclosing Orbit repository.

## Current Vendor

- Upstream tag: none (`untagged`); upstream build number `12657` (the commit
  count, the same rule that gave `10968` for `41abbfd`), recorded as
  `upstream_build_number`
- Upstream commit: `46f4da217419ecac95708600e0f6ebc4bdb91cfd`
- Canonical source-tree SHA-256:
  `73cfdc82c1616470510c6cdc3c19efe3c72a646d401b79851546baa0429c4917`
- Orbit patchset SHA-256 (`orbit-unified-diff-v1`: SHA-256 over the
  concatenated `diff -u --label a/<path> --label b/<path>` outputs of the
  declared patched paths in sorted order, upstream vs vendored):
  `0b04f39c0dc57b1685822f62b322213d7c4a565da349f1040032813481a93f81` (this value is NOT reproducible against the
  previous pin's recorded one; the algorithm above is what generated it)
- Reproducible patchset attestation (`orbit-patchset-v2`, see
  `scripts/llama_provenance_v2.py`):
  `d2464f8a71b8416207663d4804fceb8f74ab21a35e0da174309b016a5d1a91a6`

Previous vendor: `41abbfd599fbdd3470fcae0a1fb6530ad8403cd7` (build 10968); before that `b9551` / `379ac6673b5cd75c7b4e07d1521c50f1e093878c`. A
vendor upgrade must rebuild the complete native runtime and bridge together,
then repeat tokenizer, renderer, mmproj, MTP, lifecycle, final-prefix, and
process-isolated conformance gates.
