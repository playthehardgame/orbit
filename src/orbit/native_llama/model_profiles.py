from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Mapping

from .model_contract import ReasoningBehavior, TEXT_CAPABILITIES, reasoning_enabled


GEMMA4_PROFILE_ID = "orbit-gemma4-native-v1"
GEMMA4_VERIFIED_MODEL_NAME = "gemma-4-26B-A4B-it"
# Gemma 4 E4B "ultra uncensored heretic" finetune: one exact Q4_K_M artifact
# (general.file_type 15), isolated from the 26B-A4B profile. MTP, final-prefix
# reuse and route-prefix reuse stay unqualified; image/audio input uses the
# declared BF16 mmproj (clip gemma4v/gemma4a).
GEMMA4_E4B_HERETIC_PROFILE_ID = "orbit-gemma4-e4b-heretic-native-v1"
GEMMA4_E4B_HERETIC_VERIFIED_MODEL_NAME = "Gemma 4 E4B It Ultra Uncensored Heretic"
GEMMA4_E4B_HERETIC_TEMPLATE_SHA256 = "781d10940fbc44be40064b5d43a056fc486c84ceaa55538226368b57314132bf"
GEMMA4_E4B_HERETIC_VERIFIED_FILE_TYPE = "15"
GEMMA4_E4B_HERETIC_VERIFIED_QUANTIZATION = "Q4_K_M"
# Thinking Machines Inkling 2 Mini (ggml-org GGUF): `inkling2_loop`, a looped MoE
# (24 blocks, 128 experts / 8 used, blocks 6-17 run 3x per token). Exactly one
# artifact is identity-pinned: the official ggml-org IQ4_NL (general.file_type
# 25), read from the real GGUF header. The embedded template renders tools as a
# JSON `tool_declare` block and reasoning as `<|content_thinking|>` blocks, and
# takes `reasoning_effort` (default 0.9) rather than `enable_thinking`, so the
# model always reasons from Orbit's point of view. llama.cpp parses it with its
# dedicated Inkling PEG parser. Image input (mmproj) is not wired: no multimodal.
INKLING2_MINI_PROFILE_ID = "orbit-inkling2-mini-native-v1"
INKLING2_MINI_VERIFIED_MODEL_NAME = "Inkling 2 Mini"
INKLING2_MINI_ARCHITECTURE = "inkling2_loop"
INKLING2_MINI_OFFICIAL_TEMPLATE_SHA256 = "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"
INKLING2_MINI_VERIFIED_FILE_TYPE = "25"
INKLING2_MINI_VERIFIED_QUANTIZATION = "IQ4_NL"
QWEN36_PROFILE_ID = "orbit-qwen36-native-v1"
QWEN36_VERIFIED_MODEL_NAME = "Qwen3.6-35B-A3B"
QWEN36_OFFICIAL_TEMPLATE_SHA256 = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
QWEN36_VERIFIED_FILE_TYPE = "15"
QWEN36_VERIFIED_QUANTIZATION = "Q4_K_M"
QWEN38_PROFILE_ID = "orbit-qwen38-native-v1"
QWEN38_VERIFIED_MODEL_NAME = "Qwen3.8-27B"
QWEN38_OFFICIAL_TEMPLATE_SHA256 = "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"
QWEN38_VERIFIED_FILE_TYPE = "15"
QWEN38_VERIFIED_QUANTIZATION = "Q4_K_M"
QWEN38_SALUKI_PROFILE_ID = "orbit-qwen38-saluki-native-v1"
QWEN38_SALUKI_VERIFIED_MODEL_NAME = "Qwen3.8-27B GSQ-RCO"
QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
QWEN38_SALUKI_VERIFIED_FILE_TYPE = "20"
QWEN38_SALUKI_VERIFIED_QUANTIZATION = "IQ2_XS"
# NousResearch Arete 27B (bartowski GGUF): dense `qwen35`, Qwen3.8-27B finetune.
# Its embedded chat template is byte-identical to the Saluki one already
# reviewed (same sha256), but it is a distinct model/quantization, so it gets
# its own isolated identity; nothing is inherited from the Saluki or full
# Qwen3.8 profiles. Identity read from the real Q4_K_M GGUF header.
QWEN38_ARETE_PROFILE_ID = "orbit-qwen38-arete-native-v1"
QWEN38_ARETE_VERIFIED_MODEL_NAME = "Arete 27B"
QWEN38_ARETE_OFFICIAL_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
QWEN38_ARETE_VERIFIED_FILE_TYPE = "15"
QWEN38_ARETE_VERIFIED_QUANTIZATION = "Q4_K_M"
# DavidAU Qwen3.8-27B TurboFCFusion 735/882 Heretic Uncensored NEO-CODER MAX
# (plain, non-MTP Q4_K_M GGUF): dense `qwen35`, Qwen3.8-27B finetune/merge. The
# GGUF's general.name is "Qwen3.8 27B Brainwaves NM HERETIC BR LOA1", not the
# repo name. Its embedded template differs from the Saluki/Arete one (it adds
# the reasoning_effort/preserve_thinking blocks) and was reviewed by reading the
# Jinja only; the tool-call format is the same Qwen XML. Distinct identity,
# nothing inherited. Identity read from the real GGUF header (first 16 MiB).
QWEN38_TURBO_FC_PROFILE_ID = "orbit-qwen38-turbo-fcfusion-native-v1"
QWEN38_TURBO_FC_VERIFIED_MODEL_NAME = "Qwen3.8 27B Brainwaves NM HERETIC BR LOA1"
QWEN38_TURBO_FC_OFFICIAL_TEMPLATE_SHA256 = "514d5304fcac63bda60e6faf860d5657010cf43cbc4c633c8bf678ee184e7a37"
QWEN38_TURBO_FC_VERIFIED_FILE_TYPE = "15"
QWEN38_TURBO_FC_VERIFIED_QUANTIZATION = "Q4_K_M"
# Principled Intelligence ScopeGuard-2 4B (mradermacher Q8_0 GGUF): a scope /
# guardrail classifier fine-tuned from Qwen3.5-4B (`qwen35`, 33 blocks). It is a
# single-purpose classifier (fixed system prompt, JSON `scope_class` output), not
# a general chat model; the profile only makes it loadable and identity-pinned.
# Its embedded template is the stock Qwen3.5 one (XML tool calls, `<think>`),
# a hash not previously pinned here. Identity read from the real Q8_0 header.
SCOPE_GUARD_PROFILE_ID = "orbit-scope-guard-v2-4b-native-v1"
SCOPE_GUARD_VERIFIED_MODEL_NAME = "Scope Guard v2 4B Q 2609"
SCOPE_GUARD_OFFICIAL_TEMPLATE_SHA256 = "8452ca85cb1e0ff04304c02f417a53305d5ba17f6eb9d5693343ad8355f985a8"
SCOPE_GUARD_VERIFIED_FILE_TYPE = "7"
SCOPE_GUARD_VERIFIED_QUANTIZATION = "Q8_0"
QWEN38_9B_DISTILL_PROFILE_ID = "orbit-qwen38-9b-distill-native-v1"
QWEN38_9B_DISTILL_VERIFIED_MODEL_NAME = "Qwen3.8 9B"
QWEN38_9B_DISTILL_OFFICIAL_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
QWEN38_9B_DISTILL_VERIFIED_FILE_TYPE = "15"
QWEN38_9B_DISTILL_VERIFIED_QUANTIZATION = "Q4_K_M"
# Qwen3.8 Flash Next: the `qwen4exp` MoE preview (48 blocks, 512 experts, 10
# used). Exactly one artifact is qualified for production, Unsloth UD-IQ1_M
# (general.file_type 31 = LLAMA_FTYPE_MOSTLY_IQ1_M), by
# QWEN38-ORBIT-PRODUCTION-ENABLEMENT-19; its GGUF embeds byte-for-byte the
# official Qwen3.8 chat template already pinned for the 27B model.
QWEN38_FLASH_NEXT_PROFILE_ID = "orbit-qwen38-flash-next-native-v1"
QWEN38_FLASH_NEXT_VERIFIED_MODEL_NAME = "Qwen3.8 Flash Next"
QWEN38_FLASH_NEXT_VERIFIED_FILE_TYPE = "31"
QWEN38_FLASH_NEXT_VERIFIED_QUANTIZATION = "UD-IQ1_M"
# Profiles admitted to the fixed-head, call-boundary-aligned ROUTE prefix reuse
# (qwen38-aligned-route-prefix-v1), each with the one general.file_type it was
# qualified at. Membership is an evidence decision, never inferred from family.
QWEN38_ALIGNED_ROUTE_PREFIX_FILE_TYPES = {
    QWEN38_FLASH_NEXT_PROFILE_ID: QWEN38_FLASH_NEXT_VERIFIED_FILE_TYPE,
    QWEN38_ARETE_PROFILE_ID: QWEN38_ARETE_VERIFIED_FILE_TYPE,
}
LFM25_PROFILE_ID = "orbit-lfm25-native-v1"
LFM25_Q8_PROFILE_ID = "orbit-lfm25-q8-native-v1"
LFM25_Q6_PROFILE_ID = "orbit-lfm25-q6-native-v1"
LFM25_VERIFIED_MODEL_NAME = "Lfm2.5-8B-A1B"
LFM25_OFFICIAL_TEMPLATE_SHA256 = "6d65c8804847ad74eea912dd7eca3dc1cf7a457b53a77f47d841a14121910963"
LFM25_VERIFIED_FILE_TYPE = "15"
LFM25_VERIFIED_QUANTIZATION = "Q4_K_M"
LFM25_VERIFIED_FILE_TYPES = {
    "15": "Q4_K_M",
    "7": "Q8_0",
    "18": "Q6_K",
}
# Metadata keys read when identifying a model profile. Every key any pinned
# identity below compares must appear here: a key absent from this set is read
# as the empty string, which silently fails the comparison and reports a
# verified model as unsupported. Callers that extract GGUF metadata filter
# through this one set, so adding a key to an identity and forgetting the
# extraction cannot happen twice in two places.
PROFILE_METADATA_KEYS = frozenset(
    {
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "gemma4.context_length",
        "gemma4.block_count",
        "gemma4.embedding_length_per_layer_input",
        "qwen3moe.context_length",
        "qwen3moe.block_count",
        "qwen3moe.expert_count",
        "qwen3moe.expert_used_count",
        "qwen35.context_length",
        "qwen35.block_count",
        "qwen35moe.context_length",
        "qwen35moe.block_count",
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
        "qwen4exp.context_length",
        "qwen4exp.block_count",
        "qwen4exp.expert_count",
        "qwen4exp.expert_used_count",
        "lfm2moe.context_length",
        "lfm2moe.block_count",
        "lfm2moe.expert_count",
        "lfm2moe.expert_used_count",
        "inkling2_loop.block_count",
        "inkling2_loop.expert_count",
        "inkling2_loop.expert_used_count",
        "inkling2_loop.loop.count",
        "llama.context_length",
        "llama.block_count",
        "granite.context_length",
        "granite.block_count",
    }
)

ORNITH15_PROFILE_ID = "orbit-ornith15-native-v1"
ORNITH15_VERIFIED_MODEL_NAME = "Ornith-1.5-35B"
ORNITH15_OFFICIAL_TEMPLATE_SHA256 = "f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b"
ORNITH15_VERIFIED_FILE_TYPE = "15"
ORNITH15_VERIFIED_QUANTIZATION = "Q4_K_M"
QWEN3_CODER_PROFILE_ID = "orbit-qwen3-coder-native-v1"
QWEN3_CODER_VERIFIED_MODEL_NAME = "Qwen3-Coder-30B-A3B-Instruct"
QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256 = "87710339d25b4e789c1d723f93c91ee861a86d305bb3d20a845536f251d6ea8a"
QWEN3_CODER_VERIFIED_FILE_TYPE = "15"
QWEN3_CODER_VERIFIED_QUANTIZATION = "Q4_K_M"
MINICPM5_PROFILE_ID = "orbit-minicpm5-native-v1"
MINICPM5_VERIFIED_MODEL_NAME = "MiniCPM5 2B"
MINICPM5_OFFICIAL_TEMPLATE_SHA256 = "cc945752db555d60949b16989df4ccfeb52a313d6b4b5c5229dd786e2e9fcf1c"
MINICPM5_VERIFIED_FILE_TYPE = "15"
MINICPM5_VERIFIED_QUANTIZATION = "Q4_K_M"
GRANITE42_PROFILE_ID = "orbit-granite42-native-v1"
GRANITE42_VERIFIED_MODEL_NAME = "Granite 4.2 3b"
GRANITE42_OFFICIAL_TEMPLATE_SHA256 = "f0ba43f79b3cabca5e5a7584c77aec92f49feae45cdf7779c87d9fc54cd90258"
GRANITE42_VERIFIED_FILE_TYPE = "15"
GRANITE42_VERIFIED_QUANTIZATION = "Q4_K_M"
GRANITE42_3B_VERIFIED_FILE_TYPES = frozenset({"7", "15"})
GRANITE42_8B_PROFILE_ID = "orbit-granite42-8b-native-v1"
GRANITE42_8B_VERIFIED_MODEL_NAME = "Granite 4.2 8b"
GRANITE42_8B_VERIFIED_FILE_TYPES = frozenset({"15", "18"})

_GRANITE42_QUANTIZATIONS = {
    "7": "Q8_0",
    "15": "Q4_K_M",
    "18": "Q6_K",
}


def granite42_quantization_for_file_type(
    profile_id: str, file_type: str
) -> str | None:
    allowed = (
        GRANITE42_3B_VERIFIED_FILE_TYPES
        if profile_id == GRANITE42_PROFILE_ID
        else GRANITE42_8B_VERIFIED_FILE_TYPES
        if profile_id == GRANITE42_8B_PROFILE_ID
        else frozenset()
    )
    if file_type not in allowed:
        return None
    return _GRANITE42_QUANTIZATIONS[file_type]


@dataclass(frozen=True)
class NativeModelProfile:
    profile_id: str
    family: str
    model_name: str
    architecture: str
    renderer: str
    reasoning_protocol: str
    tool_call_protocol: str
    history_serialization: str
    verified: bool
    failure_reason: str | None
    template_source: str
    template_sha256: str
    thinking_supported: bool
    mtp_supported: bool
    gemma_prefix_reuse_supported: bool
    verified_quantization: str | None = None
    artifact_content_protocol: str = "literal-content-v1"
    route_prefix_reuse_supported: bool = False
    multimodal_supported: bool = False
    # Each profile declares support explicitly; identity verification alone
    # grants no operation, and no support field certifies semantic quality.
    supported_capabilities: frozenset[str] = frozenset()
    reasoning_behavior: ReasoningBehavior | None = None
    separated_reasoning: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.supported_capabilities, frozenset)
            or not self.supported_capabilities <= TEXT_CAPABILITIES
        ):
            raise ValueError("unknown technical text capability")
        if self.reasoning_behavior is not None and (
            not isinstance(self.reasoning_behavior, ReasoningBehavior)
            or (self.reasoning_behavior != ReasoningBehavior.NONE) != self.thinking_supported
        ):
            raise ValueError("reasoning behavior contradicts thinking support")
        if type(self.separated_reasoning) is not bool:
            raise ValueError("separated_reasoning must be a boolean")

    @property
    def effective_reasoning_behavior(self) -> ReasoningBehavior:
        if self.reasoning_behavior is not None:
            return self.reasoning_behavior
        return ReasoningBehavior.OPTIONAL if self.thinking_supported else ReasoningBehavior.NONE

    @property
    def uses_native_chat_bridge(self) -> bool:
        return self.renderer == "llama.cpp-jinja"

    def diagnostics(self, *, thinking_enabled: bool) -> dict[str, object]:
        support = self.supported_capabilities
        return {
            "model_family": self.family,
            "model_name": self.model_name,
            "compatibility_profile": self.profile_id,
            "architecture": self.architecture,
            "template_source": self.template_source,
            "template_hash": self.template_sha256,
            "renderer": self.renderer,
            "thinking_supported": self.thinking_supported,
            "thinking_enabled": reasoning_enabled(self.effective_reasoning_behavior, thinking_enabled),
            "thinking_requested": thinking_enabled,
            "reasoning_behavior": self.effective_reasoning_behavior.value,
            "separated_reasoning": self.separated_reasoning,
            "reasoning_protocol": self.reasoning_protocol,
            "tool_call_protocol": self.tool_call_protocol,
            "history_serialization": self.history_serialization,
            "verified": self.verified,
            "failure_reason": self.failure_reason,
            "mtp_supported": self.mtp_supported,
            "verified_quantization": self.verified_quantization,
            "artifact_content_protocol": self.artifact_content_protocol,
            "capability_scope": "technical_profile_support_not_qualification",
            "qualification": {"status": "reported_separately", "automatic_promotion": False},
            "capabilities": {
                **{name: self.verified and name in support for name in sorted(TEXT_CAPABILITIES)},
                "mtp": self.mtp_supported,
                "route_prefix_reuse": self.route_prefix_reuse_supported,
                "multimodal": self.multimodal_supported,
                "arbitrary_exact_copy_artifact": False,
                "empty_artifact": False,
            },
        }


def supports_low_memory_mode(profile: NativeModelProfile | None) -> bool:
    return bool(
        profile is not None
        and profile.verified
        and profile.profile_id == QWEN3_CODER_PROFILE_ID
    )


_VERIFIED_TEMPLATE_HISTORY_SERIALIZATION = {
    QWEN36_OFFICIAL_TEMPLATE_SHA256: "qwen-leading-system-only",
    QWEN38_OFFICIAL_TEMPLATE_SHA256: "qwen-leading-system-only",
    ORNITH15_OFFICIAL_TEMPLATE_SHA256: "qwen-leading-system-only",
}


# Capabilities a specific ARTIFACT is qualified for, as opposed to what its
# profile supports. The distinction is the whole point: two GGUFs can share a
# profile, a model name and an architecture and still differ in what they are
# qualified to do -- the current and legacy Ornith builds are exactly that
# case, identical in every metadata field discovery reads.
#
# `self_mtp` means "this exact artifact is qualified to drive llama.cpp's
# single-GGUF NextN self-speculative decoding". It is deliberately NOT
# `NativeModelProfile.mtp_supported`, which gates the external-draft MTP path
# and is a property of the profile.
SELF_MTP_CAPABILITY = "self_mtp"

# Current official Ornith 1.5 35B-A3B Q4_K_M build. Its blk.40 NextN weights
# differ from the legacy local build; only this one is qualified.
ORNITH15_CURRENT_ARTIFACT_SHA256 = (
    "42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f"
)


@dataclass(frozen=True)
class VerifiedNativeModelIdentity:
    profile_id: str
    model_name: str
    architecture: str
    # sha256 -> frozenset of capability names. Keyed by content, never by path
    # or basename, so the same bytes qualify under any filename and a renamed
    # impostor never does. Empty by default: an artifact earns capabilities by
    # being listed, so an unknown digest fails closed without a special case.
    # Stored as a frozenset of (digest, capabilities) pairs rather than a
    # mapping, because the field has to be BOTH hashable and compared.
    #
    # A mappingproxy delegates hashing to the dict it wraps, so keeping it here
    # made the frozen record unhashable. Excluding it with `compare=False`
    # restored the hash but collapsed equality: two identities differing only
    # in which artifact they qualify became interchangeable, and a
    # legacy-qualified record would silently overwrite a current-qualified one
    # used as a dict key -- the very substitution this module exists to
    # prevent, relocated from the digest lookup into record identity.
    #
    # A frozenset of pairs has neither problem: hashable, and it distinguishes.
    # `artifact_capability_map()` reads it back as a mapping.
    artifact_capabilities: frozenset[tuple[str, frozenset[str]]] = frozenset()


VERIFIED_NATIVE_MODEL_IDENTITIES = (
    VerifiedNativeModelIdentity(GEMMA4_PROFILE_ID, GEMMA4_VERIFIED_MODEL_NAME, "gemma4"),
    VerifiedNativeModelIdentity(
        GEMMA4_E4B_HERETIC_PROFILE_ID, GEMMA4_E4B_HERETIC_VERIFIED_MODEL_NAME, "gemma4"
    ),
    VerifiedNativeModelIdentity(QWEN36_PROFILE_ID, QWEN36_VERIFIED_MODEL_NAME, "qwen35moe"),
    VerifiedNativeModelIdentity(
        ORNITH15_PROFILE_ID,
        ORNITH15_VERIFIED_MODEL_NAME,
        "qwen35moe",
        frozenset(
            {(ORNITH15_CURRENT_ARTIFACT_SHA256, frozenset({SELF_MTP_CAPABILITY}))}
        ),
    ),
    VerifiedNativeModelIdentity(QWEN38_PROFILE_ID, QWEN38_VERIFIED_MODEL_NAME, "qwen35"),
    VerifiedNativeModelIdentity(
        QWEN38_SALUKI_PROFILE_ID, QWEN38_SALUKI_VERIFIED_MODEL_NAME, "qwen35"
    ),
    VerifiedNativeModelIdentity(
        QWEN38_ARETE_PROFILE_ID, QWEN38_ARETE_VERIFIED_MODEL_NAME, "qwen35"
    ),
    VerifiedNativeModelIdentity(
        QWEN38_TURBO_FC_PROFILE_ID, QWEN38_TURBO_FC_VERIFIED_MODEL_NAME, "qwen35"
    ),
    VerifiedNativeModelIdentity(
        SCOPE_GUARD_PROFILE_ID, SCOPE_GUARD_VERIFIED_MODEL_NAME, "qwen35"
    ),
    VerifiedNativeModelIdentity(
        QWEN38_9B_DISTILL_PROFILE_ID, QWEN38_9B_DISTILL_VERIFIED_MODEL_NAME, "qwen35"
    ),
    VerifiedNativeModelIdentity(
        QWEN38_FLASH_NEXT_PROFILE_ID, QWEN38_FLASH_NEXT_VERIFIED_MODEL_NAME, "qwen4exp"
    ),
    VerifiedNativeModelIdentity(LFM25_PROFILE_ID, LFM25_VERIFIED_MODEL_NAME, "lfm2moe"),
    VerifiedNativeModelIdentity(
        INKLING2_MINI_PROFILE_ID, INKLING2_MINI_VERIFIED_MODEL_NAME, INKLING2_MINI_ARCHITECTURE
    ),
    VerifiedNativeModelIdentity(LFM25_Q8_PROFILE_ID, LFM25_VERIFIED_MODEL_NAME, "lfm2moe"),
    VerifiedNativeModelIdentity(LFM25_Q6_PROFILE_ID, LFM25_VERIFIED_MODEL_NAME, "lfm2moe"),
    VerifiedNativeModelIdentity(
        QWEN3_CODER_PROFILE_ID,
        QWEN3_CODER_VERIFIED_MODEL_NAME,
        "qwen3moe",
    ),
    VerifiedNativeModelIdentity(
        MINICPM5_PROFILE_ID,
        MINICPM5_VERIFIED_MODEL_NAME,
        "llama",
    ),
        VerifiedNativeModelIdentity(
            GRANITE42_PROFILE_ID,
            GRANITE42_VERIFIED_MODEL_NAME,
            "granite",
        ),
        VerifiedNativeModelIdentity(
            GRANITE42_8B_PROFILE_ID,
            GRANITE42_8B_VERIFIED_MODEL_NAME,
            "granite",
        ),
)


def artifact_capability_map(
    identity: VerifiedNativeModelIdentity,
) -> Mapping[str, frozenset[str]]:
    """Read the stored pairs back as a digest -> capabilities mapping."""
    return MappingProxyType(dict(identity.artifact_capabilities))


def verified_native_model_identity(profile_id: str) -> VerifiedNativeModelIdentity | None:
    return next(
        (identity for identity in VERIFIED_NATIVE_MODEL_IDENTITIES if identity.profile_id == profile_id),
        None,
    )


@dataclass(frozen=True)
class _Probe:
    """Normalized inputs shared by every ordered identity matcher."""

    architecture: str
    tokenizer_model: str
    tokenizer_pre: str
    model_name: str
    file_type: str
    template_hash: str
    metadata: Mapping[str, str]
    template: str
    gemma4_e_series: bool


def _build_probe(metadata: Mapping[str, str], template: str) -> _Probe:
    architecture = metadata.get("general.architecture", "").strip().lower()
    tokenizer_model = metadata.get("tokenizer.ggml.model", "").strip().lower()
    tokenizer_pre = metadata.get("tokenizer.ggml.pre", "").strip().lower()
    model_name = metadata.get("general.name", "").strip()
    file_type = metadata.get("general.file_type", "").strip()
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()

    # Per-layer-embedding (E2B/E4B) Gemma 4 models are a different family from
    # the qualified 26B-A4B; only the exact E4B Heretic artifact is verified.
    gemma4_e_series = (
        architecture == "gemma4"
        and metadata.get("gemma4.embedding_length_per_layer_input", "").strip() not in ("", "0")
    ) or model_name == GEMMA4_E4B_HERETIC_VERIFIED_MODEL_NAME
    return _Probe(
        architecture=architecture,
        tokenizer_model=tokenizer_model,
        tokenizer_pre=tokenizer_pre,
        model_name=model_name,
        file_type=file_type,
        template_hash=template_hash,
        metadata=metadata,
        template=template,
        gemma4_e_series=gemma4_e_series,
    )


def _match_gemma4_e4b_heretic(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    gemma4_e4b_heretic_identity = (
        architecture == "gemma4"
        and model_name == GEMMA4_E4B_HERETIC_VERIFIED_MODEL_NAME
        and tokenizer_model == "gemma4"
        and file_type == GEMMA4_E4B_HERETIC_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "2"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "1"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "0"
        and metadata.get("gemma4.context_length", "").strip() == "131072"
        and metadata.get("gemma4.block_count", "").strip() == "42"
        and metadata.get("gemma4.embedding_length_per_layer_input", "").strip() == "256"
        and template_hash == GEMMA4_E4B_HERETIC_TEMPLATE_SHA256
    )
    if gemma4_e4b_heretic_identity:
        return NativeModelProfile(
            profile_id=GEMMA4_E4B_HERETIC_PROFILE_ID,
            family="gemma4-e4b-heretic",
            model_name=model_name,
            architecture=architecture,
            renderer="orbit-gemma4",
            reasoning_protocol="gemma4-control-channel",
            tool_call_protocol="gemma4-native",
            history_serialization="orbit-native-roles",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-reviewed",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=GEMMA4_E4B_HERETIC_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=False,
            multimodal_supported=True,
        )
    return None


def _match_gemma4(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    model_name = probe.model_name
    template_hash = probe.template_hash
    gemma4_e_series = probe.gemma4_e_series
    if (
        architecture == "gemma4"
        and tokenizer_model == "gemma4"
        and not gemma4_e_series
    ):
        return NativeModelProfile(
            profile_id=GEMMA4_PROFILE_ID,
            family="gemma4",
            model_name=model_name,
            architecture=architecture,
            renderer="orbit-gemma4",
            reasoning_protocol="gemma4-control-channel",
            reasoning_behavior=ReasoningBehavior.OPTIONAL,
            tool_call_protocol="gemma4-native",
            history_serialization="orbit-native-roles",
            verified=True,
            supported_capabilities=TEXT_CAPABILITIES,
            failure_reason=None,
            template_source="orbit-reviewed",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=True,
            gemma_prefix_reuse_supported=True,
            route_prefix_reuse_supported=True,
            multimodal_supported=True,
        )
    return None


def _match_qwen36(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    qwen_identity = (
        architecture == "qwen35moe"
        and model_name == QWEN36_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN36_VERIFIED_FILE_TYPE
        and template_hash == QWEN36_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen_identity:
        return NativeModelProfile(
            profile_id=QWEN36_PROFILE_ID,
            family="qwen3.6",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            reasoning_behavior=ReasoningBehavior.OPTIONAL,
            separated_reasoning=True,
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            supported_capabilities=TEXT_CAPABILITIES,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN36_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
        )
    return None


def _match_ornith15(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    ornith15_identity = (
        architecture == "qwen35moe"
        and model_name == ORNITH15_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == ORNITH15_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("qwen35moe.context_length", "").strip() == "262144"
        and metadata.get("qwen35moe.block_count", "").strip() == "41"
        and metadata.get("qwen35moe.expert_count", "").strip() == "256"
        and metadata.get("qwen35moe.expert_used_count", "").strip() == "8"
        and template_hash == ORNITH15_OFFICIAL_TEMPLATE_SHA256
    )
    if ornith15_identity:
        return NativeModelProfile(
            profile_id=ORNITH15_PROFILE_ID,
            family="ornith1.5",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            reasoning_behavior=ReasoningBehavior.OPTIONAL,
            separated_reasoning=True,
            # Tool envelope verified byte-identical to the reviewed Qwen3.6
            # template: same <tool_call>/<function=>/<parameter=> form and
            # <tool_response> results.
            tool_call_protocol="qwen3.6-xml",
            # This template silently DROPS a system message past the leading
            # run instead of raising, so Orbit evidence and citation cards
            # would vanish without the leading-system-only contract.
            history_serialization="qwen-leading-system-only",
            verified=True,
            supported_capabilities=TEXT_CAPABILITIES,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=ORNITH15_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
        )
    return None


def _match_qwen38(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    qwen38_identity = (
        architecture == "qwen35"
        and model_name == QWEN38_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN38_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("qwen35.context_length", "").strip() == "262144"
        and metadata.get("qwen35.block_count", "").strip() == "65"
        and template_hash == QWEN38_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen38_identity:
        return NativeModelProfile(
            profile_id=QWEN38_PROFILE_ID,
            family="qwen3.8",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            reasoning_behavior=ReasoningBehavior.OPTIONAL,
            separated_reasoning=True,
            # Wire protocol verified byte-identical to Qwen3.6: same
            # <tool_call>/<function=>/<parameter=> envelope, same
            # <tool_response> result form. Qwen3.8 only adds template-side
            # argument validation, which does not change the emitted format.
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            supported_capabilities=TEXT_CAPABILITIES,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN38_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
        )
    return None


def _match_qwen38_saluki(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    qwen38_saluki_identity = (
        architecture == "qwen35"
        and model_name == QWEN38_SALUKI_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN38_SALUKI_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.add_bos_token", "").strip().lower() == "false"
        and metadata.get("qwen35.context_length", "").strip() == "262144"
        and metadata.get("qwen35.block_count", "").strip() == "64"
        and template_hash == QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen38_saluki_identity:
        return NativeModelProfile(
            profile_id=QWEN38_SALUKI_PROFILE_ID,
            family="qwen3.8-saluki",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-reviewed",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN38_SALUKI_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=False,
        )
    return None


def _match_qwen38_arete(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    qwen38_arete_identity = (
        architecture == "qwen35"
        and model_name == QWEN38_ARETE_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN38_ARETE_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.add_bos_token", "").strip().lower() == "false"
        and metadata.get("qwen35.context_length", "").strip() == "262144"
        and metadata.get("qwen35.block_count", "").strip() == "64"
        and template_hash == QWEN38_ARETE_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen38_arete_identity:
        return NativeModelProfile(
            profile_id=QWEN38_ARETE_PROFILE_ID,
            family="qwen3.8-arete",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-reviewed",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN38_ARETE_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
            multimodal_supported=True,
        )
    return None


def _match_qwen38_turbo_fc(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    qwen38_turbo_fc_identity = (
        architecture == "qwen35"
        and model_name == QWEN38_TURBO_FC_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN38_TURBO_FC_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.add_bos_token", "").strip().lower() in ("", "false")
        and metadata.get("qwen35.context_length", "").strip() == "262144"
        and metadata.get("qwen35.block_count", "").strip() == "64"
        and "qwen35.nextn_predict_layers" not in metadata
        and template_hash == QWEN38_TURBO_FC_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen38_turbo_fc_identity:
        return NativeModelProfile(
            profile_id=QWEN38_TURBO_FC_PROFILE_ID,
            family="qwen3.8-turbo-fcfusion",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-reviewed",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN38_TURBO_FC_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=False,
        )
    return None


def _match_qwen38_9b_distill(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    qwen38_9b_distill_identity = (
        architecture == "qwen35"
        and model_name == QWEN38_9B_DISTILL_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN38_9B_DISTILL_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.add_bos_token", "").strip().lower() == "false"
        and metadata.get("qwen35.context_length", "").strip() == "262144"
        and metadata.get("qwen35.block_count", "").strip() == "33"
        and template_hash == QWEN38_9B_DISTILL_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen38_9b_distill_identity:
        return NativeModelProfile(
            profile_id=QWEN38_9B_DISTILL_PROFILE_ID,
            family="qwen3.8-9b-distill",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-reviewed",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN38_9B_DISTILL_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=False,
        )
    return None


def _match_scope_guard(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    scope_guard_identity = (
        architecture == "qwen35"
        and model_name == SCOPE_GUARD_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == SCOPE_GUARD_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "248055"
        and metadata.get("tokenizer.ggml.add_bos_token", "").strip().lower() == "false"
        and metadata.get("qwen35.context_length", "").strip() == "262144"
        and metadata.get("qwen35.block_count", "").strip() == "33"
        and template_hash == SCOPE_GUARD_OFFICIAL_TEMPLATE_SHA256
    )
    if scope_guard_identity:
        return NativeModelProfile(
            profile_id=SCOPE_GUARD_PROFILE_ID,
            family="scope-guard",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-reviewed",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=SCOPE_GUARD_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=False,
        )
    return None


def _match_qwen38_flash_next(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    qwen38_flash_next_identity = (
        architecture == "qwen4exp"
        and model_name == QWEN38_FLASH_NEXT_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen35"
        and file_type == QWEN38_FLASH_NEXT_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "248044"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "248046"
        and metadata.get("qwen4exp.context_length", "").strip() == "262144"
        and metadata.get("qwen4exp.block_count", "").strip() == "48"
        and metadata.get("qwen4exp.expert_count", "").strip() == "512"
        and metadata.get("qwen4exp.expert_used_count", "").strip() == "10"
        and template_hash == QWEN38_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen38_flash_next_identity:
        return NativeModelProfile(
            profile_id=QWEN38_FLASH_NEXT_PROFILE_ID,
            family="qwen3.8-flash-next",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="qwen-think",
            reasoning_behavior=ReasoningBehavior.OPTIONAL,
            separated_reasoning=True,
            # Same template bytes (sha256) as the verified Qwen3.8 27B pin, so
            # the same <tool_call>/<function=>/<parameter=> envelope and
            # <tool_response> results.
            tool_call_protocol="qwen3.6-xml",
            history_serialization="qwen-leading-system-only",
            verified=True,
            supported_capabilities=TEXT_CAPABILITIES,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN38_FLASH_NEXT_VERIFIED_QUANTIZATION,
            # Full sequence state at an unchanged native decode-call boundary.
            # Runtime eligibility additionally pins the qualified configuration.
            route_prefix_reuse_supported=True,
        )
    return None


def _match_inkling2_mini(probe: _Probe) -> NativeModelProfile | None:
    metadata = probe.metadata
    identity = (
        probe.architecture == INKLING2_MINI_ARCHITECTURE
        and probe.model_name == INKLING2_MINI_VERIFIED_MODEL_NAME
        and probe.tokenizer_model == "gpt2"
        and probe.tokenizer_pre == "inkling2"
        and probe.file_type == INKLING2_MINI_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "200006"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "199999"
        and metadata.get("inkling2_loop.block_count", "").strip() == "24"
        and metadata.get("inkling2_loop.expert_count", "").strip() == "128"
        and metadata.get("inkling2_loop.expert_used_count", "").strip() == "8"
        and metadata.get("inkling2_loop.loop.count", "").strip() == "3"
        and probe.template_hash == INKLING2_MINI_OFFICIAL_TEMPLATE_SHA256
    )
    if not identity:
        return None
    return NativeModelProfile(
        profile_id=INKLING2_MINI_PROFILE_ID,
        family="inkling2",
        model_name=probe.model_name,
        architecture=probe.architecture,
        renderer="llama.cpp-jinja",
        reasoning_protocol="inkling-content-thinking",
        tool_call_protocol="inkling-json",
        history_serialization="orbit-native-roles",
        verified=True,
        failure_reason=None,
        template_source="gguf-embedded-official",
        template_sha256=probe.template_hash,
        supported_capabilities=TEXT_CAPABILITIES,
        thinking_supported=True,
        reasoning_behavior=ReasoningBehavior.ALWAYS,
        separated_reasoning=True,
        mtp_supported=False,
        gemma_prefix_reuse_supported=False,
        verified_quantization=INKLING2_MINI_VERIFIED_QUANTIZATION,
    )


def _match_lfm25(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    lfm25_identity = (
        architecture == "lfm2moe"
        and model_name == LFM25_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "lfm2"
        and file_type in LFM25_VERIFIED_FILE_TYPES
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "124894"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "124900"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "124893"
        and metadata.get("lfm2moe.context_length", "").strip() == "128000"
        and metadata.get("lfm2moe.block_count", "").strip() == "24"
        and metadata.get("lfm2moe.expert_count", "").strip() == "32"
        and metadata.get("lfm2moe.expert_used_count", "").strip() == "4"
        and template_hash == LFM25_OFFICIAL_TEMPLATE_SHA256
    )
    if lfm25_identity:
        return NativeModelProfile(
            profile_id={
                "7": LFM25_Q8_PROFILE_ID,
                "18": LFM25_Q6_PROFILE_ID,
            }.get(file_type, LFM25_PROFILE_ID),
            family="lfm2.5",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="lfm2-think",
            tool_call_protocol="lfm2-python",
            history_serialization="orbit-native-roles",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=LFM25_VERIFIED_FILE_TYPES[file_type],
        )
    return None


def _match_qwen3_coder(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    qwen3_coder_identity = (
        architecture == "qwen3moe"
        and model_name == QWEN3_CODER_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "qwen2"
        and file_type == QWEN3_CODER_VERIFIED_FILE_TYPE
        and metadata.get("general.quantization_version", "").strip() == "2"
        and metadata.get("tokenizer.ggml.add_bos_token", "").strip().lower() == "false"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "151645"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "151654"
        and metadata.get("qwen3moe.context_length", "").strip() == "262144"
        and metadata.get("qwen3moe.expert_count", "").strip() == "128"
        and metadata.get("qwen3moe.expert_used_count", "").strip() == "8"
        and template_hash == QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256
    )
    if qwen3_coder_identity:
        return NativeModelProfile(
            profile_id=QWEN3_CODER_PROFILE_ID,
            family="qwen3-coder",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="none",
            reasoning_behavior=ReasoningBehavior.NONE,
            tool_call_protocol="qwen3-coder-xml",
            history_serialization="qwen3-coder-chatml",
            verified=True,
            supported_capabilities=TEXT_CAPABILITIES,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            thinking_supported=False,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=QWEN3_CODER_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
            artifact_content_protocol="qwen3-coder-json-string-v1",
        )
    return None


def _match_minicpm5(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    minicpm5_identity = (
        architecture == "llama"
        and model_name == MINICPM5_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "minicpm5"
        and file_type == MINICPM5_VERIFIED_FILE_TYPE
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "0"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "1"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "1"
        and metadata.get("tokenizer.ggml.add_bos_token", "").strip().lower() == "false"
        and metadata.get("llama.context_length", "").strip() == "131072"
        and metadata.get("llama.block_count", "").strip() == "42"
        and template_hash == MINICPM5_OFFICIAL_TEMPLATE_SHA256
    )
    if minicpm5_identity:
        return NativeModelProfile(
            profile_id=MINICPM5_PROFILE_ID,
            family="minicpm5",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="minicpm5-think",
            tool_call_protocol="minicpm5-xml",
            history_serialization="orbit-native-roles",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=MINICPM5_VERIFIED_QUANTIZATION,
            route_prefix_reuse_supported=True,
        )
    return None


def _match_granite42(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    granite42_identity = (
        architecture == "granite"
        and model_name == GRANITE42_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "granite-docling"
        and file_type in GRANITE42_3B_VERIFIED_FILE_TYPES
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "100283"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "100257"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "100257"
        and metadata.get("granite.context_length", "").strip() == "131072"
        and metadata.get("granite.block_count", "").strip() == "40"
        and template_hash == GRANITE42_OFFICIAL_TEMPLATE_SHA256
    )
    if granite42_identity:
        return NativeModelProfile(
            profile_id=GRANITE42_PROFILE_ID,
            family="granite4.2",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="granite-think",
            tool_call_protocol="qwen3-coder-xml",
            history_serialization="granite-chatml",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=_GRANITE42_QUANTIZATIONS[file_type],
            route_prefix_reuse_supported=True,
        )
    return None


def _match_granite42_8b(probe: _Probe) -> NativeModelProfile | None:
    architecture = probe.architecture
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    model_name = probe.model_name
    file_type = probe.file_type
    template_hash = probe.template_hash
    metadata = probe.metadata
    granite42_8b_identity = (
        architecture == "granite"
        and model_name == GRANITE42_8B_VERIFIED_MODEL_NAME
        and tokenizer_model == "gpt2"
        and tokenizer_pre == "granite-docling"
        and file_type in GRANITE42_8B_VERIFIED_FILE_TYPES
        and metadata.get("tokenizer.ggml.bos_token_id", "").strip() == "100283"
        and metadata.get("tokenizer.ggml.eos_token_id", "").strip() == "100257"
        and metadata.get("tokenizer.ggml.padding_token_id", "").strip() == "100257"
        and metadata.get("granite.context_length", "").strip() == "131072"
        and metadata.get("granite.block_count", "").strip() == "40"
        and template_hash == GRANITE42_OFFICIAL_TEMPLATE_SHA256
    )
    if granite42_8b_identity:
        return NativeModelProfile(
            profile_id=GRANITE42_8B_PROFILE_ID,
            family="granite4.2",
            model_name=model_name,
            architecture=architecture,
            renderer="llama.cpp-jinja",
            reasoning_protocol="granite-think",
            tool_call_protocol="qwen3-coder-xml",
            history_serialization="granite-chatml",
            verified=True,
            failure_reason=None,
            template_source="gguf-embedded-official",
            template_sha256=template_hash,
            supported_capabilities=TEXT_CAPABILITIES,
            thinking_supported=True,
            mtp_supported=False,
            gemma_prefix_reuse_supported=False,
            verified_quantization=_GRANITE42_QUANTIZATIONS[file_type],
            route_prefix_reuse_supported=True,
        )
    return None


# Order is significant: the first matching identity wins.
_PROFILE_MATCHERS = (
    _match_gemma4_e4b_heretic,
    _match_gemma4,
    _match_qwen36,
    _match_ornith15,
    _match_qwen38,
    _match_qwen38_saluki,
    _match_qwen38_arete,
    _match_qwen38_turbo_fc,
    _match_qwen38_9b_distill,
    _match_scope_guard,
    _match_qwen38_flash_next,
    _match_lfm25,
    _match_inkling2_mini,
    _match_qwen3_coder,
    _match_minicpm5,
    _match_granite42,
    _match_granite42_8b,
)


def detect_native_model_profile(metadata: Mapping[str, str], template: str) -> NativeModelProfile:
    probe = _build_probe(metadata, template)
    for matcher in _PROFILE_MATCHERS:
        profile = matcher(probe)
        if profile is not None:
            return profile
    architecture = probe.architecture
    model_name = probe.model_name
    tokenizer_model = probe.tokenizer_model
    tokenizer_pre = probe.tokenizer_pre
    file_type = probe.file_type
    template_hash = probe.template_hash
    gemma4_e_series = probe.gemma4_e_series

    reason = _unverified_reason(
        architecture=architecture,
        model_name=model_name,
        tokenizer_model=tokenizer_model,
        tokenizer_pre=tokenizer_pre,
        file_type=file_type,
        template_hash=template_hash,
        gemma4_e_series=gemma4_e_series,
    )
    return NativeModelProfile(
        profile_id="unsupported",
        family=architecture or "unknown",
        model_name=model_name,
        architecture=architecture or "unknown",
        renderer="unsupported",
        reasoning_protocol="unsupported",
        tool_call_protocol="unsupported",
        history_serialization="unsupported",
        verified=False,
        failure_reason=reason,
        template_source="gguf-embedded" if template else "missing",
        template_sha256=template_hash,
        thinking_supported=False,
        mtp_supported=False,
        gemma_prefix_reuse_supported=False,
    )


def _unverified_reason(
    *,
    architecture: str,
    model_name: str,
    tokenizer_model: str,
    tokenizer_pre: str,
    file_type: str,
    template_hash: str,
    gemma4_e_series: bool = False,
) -> str:
    if architecture == "gemma4" and gemma4_e_series and model_name != GEMMA4_E4B_HERETIC_VERIFIED_MODEL_NAME:
        return "gemma4_e_series_unqualified"
    if architecture == "gemma4" and model_name == GEMMA4_E4B_HERETIC_VERIFIED_MODEL_NAME:
        if tokenizer_model != "gemma4":
            return "gemma4_e4b_heretic_tokenizer_identity_mismatch"
        if file_type != GEMMA4_E4B_HERETIC_VERIFIED_FILE_TYPE:
            return "gemma4_e4b_heretic_quantization_identity_mismatch"
        if template_hash != GEMMA4_E4B_HERETIC_TEMPLATE_SHA256:
            return "gemma4_e4b_heretic_template_identity_mismatch"
        return "gemma4_e4b_heretic_metadata_identity_mismatch"
    if architecture == "qwen35moe":
        if model_name == ORNITH15_VERIFIED_MODEL_NAME:
            if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
                return "ornith15_tokenizer_identity_mismatch"
            if file_type != ORNITH15_VERIFIED_FILE_TYPE:
                return "ornith15_quantization_identity_mismatch"
            if template_hash != ORNITH15_OFFICIAL_TEMPLATE_SHA256:
                return "ornith15_template_identity_mismatch"
            return "ornith15_metadata_identity_mismatch"
        if model_name != QWEN36_VERIFIED_MODEL_NAME:
            return "qwen36_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
            return "qwen36_tokenizer_identity_mismatch"
        if file_type != QWEN36_VERIFIED_FILE_TYPE:
            return "qwen36_quantization_identity_mismatch"
        if template_hash != QWEN36_OFFICIAL_TEMPLATE_SHA256:
            return "qwen36_template_identity_mismatch"
    if architecture == "qwen35":
        if model_name == QWEN38_SALUKI_VERIFIED_MODEL_NAME:
            if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
                return "qwen38_saluki_tokenizer_identity_mismatch"
            if file_type != QWEN38_SALUKI_VERIFIED_FILE_TYPE:
                return "qwen38_saluki_quantization_identity_mismatch"
            if template_hash != QWEN38_SALUKI_OFFICIAL_TEMPLATE_SHA256:
                return "qwen38_saluki_template_identity_mismatch"
            return "qwen38_saluki_metadata_identity_mismatch"
        if model_name == QWEN38_ARETE_VERIFIED_MODEL_NAME:
            if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
                return "qwen38_arete_tokenizer_identity_mismatch"
            if file_type != QWEN38_ARETE_VERIFIED_FILE_TYPE:
                return "qwen38_arete_quantization_identity_mismatch"
            if template_hash != QWEN38_ARETE_OFFICIAL_TEMPLATE_SHA256:
                return "qwen38_arete_template_identity_mismatch"
            return "qwen38_arete_metadata_identity_mismatch"
        if model_name == QWEN38_TURBO_FC_VERIFIED_MODEL_NAME:
            if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
                return "qwen38_turbo_fc_tokenizer_identity_mismatch"
            if file_type != QWEN38_TURBO_FC_VERIFIED_FILE_TYPE:
                return "qwen38_turbo_fc_quantization_identity_mismatch"
            if template_hash != QWEN38_TURBO_FC_OFFICIAL_TEMPLATE_SHA256:
                return "qwen38_turbo_fc_template_identity_mismatch"
            return "qwen38_turbo_fc_metadata_identity_mismatch"
        if model_name == SCOPE_GUARD_VERIFIED_MODEL_NAME:
            if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
                return "scope_guard_tokenizer_identity_mismatch"
            if file_type != SCOPE_GUARD_VERIFIED_FILE_TYPE:
                return "scope_guard_quantization_identity_mismatch"
            if template_hash != SCOPE_GUARD_OFFICIAL_TEMPLATE_SHA256:
                return "scope_guard_template_identity_mismatch"
            return "scope_guard_metadata_identity_mismatch"
        if model_name == QWEN38_9B_DISTILL_VERIFIED_MODEL_NAME:
            if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
                return "qwen38_9b_distill_tokenizer_identity_mismatch"
            if file_type != QWEN38_9B_DISTILL_VERIFIED_FILE_TYPE:
                return "qwen38_9b_distill_quantization_identity_mismatch"
            if template_hash != QWEN38_9B_DISTILL_OFFICIAL_TEMPLATE_SHA256:
                return "qwen38_9b_distill_template_identity_mismatch"
            return "qwen38_9b_distill_metadata_identity_mismatch"
        if model_name != QWEN38_VERIFIED_MODEL_NAME:
            return "qwen38_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
            return "qwen38_tokenizer_identity_mismatch"
        if file_type != QWEN38_VERIFIED_FILE_TYPE:
            return "qwen38_quantization_identity_mismatch"
        if template_hash != QWEN38_OFFICIAL_TEMPLATE_SHA256:
            return "qwen38_template_identity_mismatch"
        return "qwen38_metadata_identity_mismatch"
    if architecture == "qwen4exp":
        if model_name != QWEN38_FLASH_NEXT_VERIFIED_MODEL_NAME:
            return "qwen38_flash_next_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen35":
            return "qwen38_flash_next_tokenizer_identity_mismatch"
        if file_type != QWEN38_FLASH_NEXT_VERIFIED_FILE_TYPE:
            # Only UD-IQ1_M is qualified; any other Qwen3.8 Flash Next quant
            # (IQ1_S, Q2_K_XL, IQ3_XXS, ...) stays unsupported until qualified.
            return "qwen38_flash_next_quantization_identity_mismatch"
        if template_hash != QWEN38_OFFICIAL_TEMPLATE_SHA256:
            return "qwen38_flash_next_template_identity_mismatch"
        return "qwen38_flash_next_metadata_identity_mismatch"
    if architecture == "lfm2moe":
        if model_name != LFM25_VERIFIED_MODEL_NAME:
            return "lfm25_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "lfm2":
            return "lfm25_tokenizer_identity_mismatch"
        if file_type not in LFM25_VERIFIED_FILE_TYPES:
            return "lfm25_quantization_identity_mismatch"
        if template_hash != LFM25_OFFICIAL_TEMPLATE_SHA256:
            return "lfm25_template_identity_mismatch"
        return "lfm25_metadata_identity_mismatch"
    if architecture == "qwen3moe":
        if model_name != QWEN3_CODER_VERIFIED_MODEL_NAME:
            return "qwen3_coder_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "qwen2":
            return "qwen3_coder_tokenizer_identity_mismatch"
        if file_type != QWEN3_CODER_VERIFIED_FILE_TYPE:
            return "qwen3_coder_quantization_identity_mismatch"
        if template_hash != QWEN3_CODER_OFFICIAL_TEMPLATE_SHA256:
            return "qwen3_coder_template_identity_mismatch"
        return "qwen3_coder_metadata_identity_mismatch"
    if architecture == "llama":
        if model_name != MINICPM5_VERIFIED_MODEL_NAME:
            return "minicpm5_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "minicpm5":
            return "minicpm5_tokenizer_identity_mismatch"
        if file_type != MINICPM5_VERIFIED_FILE_TYPE:
            return "minicpm5_quantization_identity_mismatch"
        if template_hash != MINICPM5_OFFICIAL_TEMPLATE_SHA256:
            return "minicpm5_template_identity_mismatch"
        return "minicpm5_metadata_identity_mismatch"
    if architecture == "granite":
        if model_name not in (GRANITE42_VERIFIED_MODEL_NAME, GRANITE42_8B_VERIFIED_MODEL_NAME):
            return "granite42_model_identity_mismatch"
        if tokenizer_model != "gpt2" or tokenizer_pre != "granite-docling":
            return "granite42_tokenizer_identity_mismatch"
        if (
            model_name == GRANITE42_VERIFIED_MODEL_NAME
            and file_type not in GRANITE42_3B_VERIFIED_FILE_TYPES
        ):
            return "granite42_quantization_identity_mismatch"
        if (
            model_name == GRANITE42_8B_VERIFIED_MODEL_NAME
            and file_type not in GRANITE42_8B_VERIFIED_FILE_TYPES
        ):
            return "granite42_quantization_identity_mismatch"
        if template_hash != GRANITE42_OFFICIAL_TEMPLATE_SHA256:
            return "granite42_template_identity_mismatch"
        return "granite42_metadata_identity_mismatch"
    return "unsupported_model_profile"


def history_serialization_for_template(template: str) -> str | None:
    """History-serialization contract for a template that matches a verified pin.

    Matches the exact template text against the digests of verified profile
    templates, so a backend without Orbit-native metadata can still honour the
    profile's message-shape contract. Returns None when the template matches no
    verified profile, which keeps unverified models free of model-specific
    handling.
    """

    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return _VERIFIED_TEMPLATE_HISTORY_SERIALIZATION.get(digest)
