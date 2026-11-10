"""Characterization of why granite42-3b json_artifact ends TECHNICAL_STOP.

Findings, not a fix: the official Granite 4.2 embedded template instructs the
XML ``<tool_call><function=...><parameter=...>`` format (not JSON), and Orbit's
``write_artifact`` schema has boolean parameters and deliberately no ``content``
(content is generated in a separate phase). The model's omission of ``content``
is therefore correct; the rejected output used Python-style ``True`` for the
boolean parameters, which llama.cpp's peg-native parser does not accept.
"""

import hashlib
import unittest
from pathlib import Path

from orbit.native_llama.gguf_split import read_gguf_header
from orbit.native_llama.model_profiles import GRANITE42_OFFICIAL_TEMPLATE_SHA256
from orbit.runtime.artifacts import write_artifact_definition

_GGUF = Path("/mnt/l/models/ibm-granite--granite-4.2-3b-GGUF/granite-4.2-3b-Q4_K_M.gguf")


class GraniteToolFormatCharacterizationTests(unittest.TestCase):
    def test_write_artifact_schema_has_boolean_params_and_no_content(self):
        params = write_artifact_definition()["function"]["parameters"]
        self.assertEqual(params["required"], ["path"])
        self.assertNotIn("content", params["properties"])
        self.assertEqual(params["properties"]["overwrite"]["type"], "boolean")
        self.assertEqual(params["properties"]["create_parents"]["type"], "boolean")

    @unittest.skipUnless(_GGUF.is_file(), "Granite 4.2 3b GGUF not present on this host")
    def test_official_template_prescribes_xml_tool_calls_not_json(self):
        header = read_gguf_header(_GGUF, keys=frozenset({"tokenizer.chat_template"}))
        template = header.kv["tokenizer.chat_template"]
        self.assertEqual(hashlib.sha256(template.encode("utf-8")).hexdigest(), GRANITE42_OFFICIAL_TEMPLATE_SHA256)
        self.assertIn("<tool_call>\\n<function=example_function_name>\\n<parameter=", template)
        self.assertNotIn('<tool_call>{', template)


if __name__ == "__main__":
    unittest.main()
