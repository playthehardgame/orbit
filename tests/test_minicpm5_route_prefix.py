from __future__ import annotations

import unittest

from orbit.native_llama.minicpm5_route_prefix import (
    MINICPM5_ROUTE_PREFIX_TOKEN_COUNT,
    derive_minicpm5_route_prefix_spec,
    resolve_minicpm5_route_prefix_reuse,
)


class MiniCPM5RoutePrefixConfigTests(unittest.TestCase):
    def test_default_is_enabled(self) -> None:
        self.assertTrue(resolve_minicpm5_route_prefix_reuse({}).enabled)

    def test_kill_switch_and_invalid_value_fail_closed(self) -> None:
        self.assertFalse(
            resolve_minicpm5_route_prefix_reuse(
                {"ORBIT_MINICPM5_ROUTE_PREFIX_REUSE": "0"}
            ).enabled
        )
        config = resolve_minicpm5_route_prefix_reuse(
            {"ORBIT_MINICPM5_ROUTE_PREFIX_REUSE": "yes"}
        )
        self.assertFalse(config.enabled)
        self.assertEqual(
            config.validation_error,
            "invalid_minicpm5_route_prefix_reuse_value",
        )


class MiniCPM5RoutePrefixBoundaryTests(unittest.TestCase):
    def test_derives_the_qualified_boundary(self) -> None:
        system = "s" * 900

        def render(user: str) -> str:
            return system + "\n<user>" + user + "</user>"

        full = render("request")
        spec, reason = derive_minicpm5_route_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(char) for char in full],
            render_reference=render,
            tokenize=lambda text: [ord(char) for char in text],
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(len(spec.prefix_tokens), MINICPM5_ROUTE_PREFIX_TOKEN_COUNT)

    def test_refuses_an_unstable_boundary(self) -> None:
        system = "s" * 700
        full = system + ("x" * 100)

        spec, reason = derive_minicpm5_route_prefix_spec(
            system_prompt=system,
            full_prompt=full,
            full_tokens=[ord(char) for char in full],
            render_reference=lambda user: system + user,
            tokenize=lambda text: [ord(char) for char in text],
        )

        self.assertIsNone(spec)
        self.assertEqual(reason, "stable_token_boundary_unavailable")


if __name__ == "__main__":
    unittest.main()
