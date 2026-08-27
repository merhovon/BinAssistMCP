import unittest

from src.binassist_mcp.tools import BinAssistMCPTools


class FakeParameter:
    def __init__(self, name):
        self.name = name


class FakeLine:
    def __init__(self, address, text):
        self.address = address
        self.text = text

    def __str__(self):
        return self.text


class FakeIL:
    def __init__(self, lines):
        self.root = type("Root", (), {"lines": lines})()
        self._blocks = [lines]

    def __iter__(self):
        return iter(self._blocks)

    def __str__(self):
        return "\n".join(str(line) for line in self.root.lines)


class FakeBlock:
    start = 0x1000
    end = 0x1002


class FakeFunction:
    name = "decode_packet"
    start = 0x1000
    return_type = "char *"
    parameter_vars = [FakeParameter("buffer"), FakeParameter("length")]
    comment = "Parses a packet\nReturns its payload"
    comments = {0x1000: "validated header", 0x1001: "line one\nline two"}
    basic_blocks = [FakeBlock()]
    analysis_skipped = True
    mlil = None
    llil = None

    def __init__(self):
        self.hlil = FakeIL([
            FakeLine(0x1000, "if (length < 4)"),
            FakeLine(0x1001, "return buffer + 4"),
        ])

    def get_variable_type(self, parameter):
        return {"buffer": "uint8_t *", "length": "size_t"}[parameter.name]

    def get_comment_at(self, address):
        return self.comments.get(address)


class FakeBinaryView:
    def __init__(self, function):
        self.function = function
        self.functions = [function]
        self.analysis_updates = 0

    def get_symbol_by_raw_name(self, name):
        return None

    def get_function_at(self, address):
        return self.function if address == self.function.start else None

    def get_functions_containing(self, address):
        return [self.function] if 0x1000 <= address < 0x1002 else []

    def get_comment_at(self, address):
        return None

    def get_disassembly(self, address):
        return {0x1000: "push rbp", 0x1001: "ret"}.get(address)

    def update_analysis_and_wait(self):
        self.analysis_updates += 1


def call_get_code(tools, identifier, output_format):
    undecorated = BinAssistMCPTools.get_code.__wrapped__.__wrapped__
    return undecorated(tools, identifier, output_format)


class GetCodeContextTests(unittest.TestCase):
    def setUp(self):
        self.function = FakeFunction()
        self.tools = object.__new__(BinAssistMCPTools)
        self.tools.bv = FakeBinaryView(self.function)

    def test_decompile_includes_signature_and_comments(self):
        result = call_get_code(self.tools, "0x1000", "decompile")

        self.assertEqual(
            result["code"],
            "// Parses a packet\n"
            "// Returns its payload\n"
            "char * decode_packet(uint8_t * buffer, size_t length)\n"
            "if (length < 4)  // validated header\n"
            "return buffer + 4  // line one | line two",
        )

    def test_disassembly_includes_signature_and_instruction_comments(self):
        result = call_get_code(self.tools, "decode_packet", "disasm")

        self.assertIn("char * decode_packet(uint8_t * buffer, size_t length)", result["code"])
        self.assertIn("0x1000: push rbp  // validated header", result["code"])

    def test_pseudo_c_has_one_signature_and_closing_brace(self):
        result = call_get_code(self.tools, "decode_packet", "pseudo_c")

        self.assertEqual(result["code"].count("char * decode_packet"), 1)
        self.assertIn("char * decode_packet(uint8_t * buffer, size_t length) {", result["code"])
        self.assertTrue(result["code"].endswith("}"))


if __name__ == "__main__":
    unittest.main()
