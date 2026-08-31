import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.binassist_mcp.context import BinAssistMCPBinaryContextManager


class FakeBinaryView:
    def __init__(self, filename):
        self.file = types.SimpleNamespace(filename=filename)
        self.functions = [object()]
        self.analysis_progress = types.SimpleNamespace(state="idle")


class FakeTab:
    def __init__(self, binary_view):
        self.binary_view = binary_view

    def getCurrentBinaryView(self):
        return self.binary_view


class FakeUIContext:
    def __init__(self, open_result=True, binary_view=None):
        self.open_result = open_result
        self.binary_view = binary_view
        self.open_calls = []
        self.opened = False

    def openFilename(self, filename):
        self.open_calls.append(filename)
        self.opened = self.open_result
        return self.open_result

    def getTabs(self):
        if self.opened and self.binary_view is not None:
            return [FakeTab(self.binary_view)]
        return []


class ImmediateThread:
    def __init__(self, target, args=(), **kwargs):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class ContextOpenTests(unittest.TestCase):
    def setUp(self):
        with BinAssistMCPBinaryContextManager._open_operations_lock:
            BinAssistMCPBinaryContextManager._open_operations.clear()
            BinAssistMCPBinaryContextManager._active_open_paths.clear()

        temporary = tempfile.NamedTemporaryFile(suffix=".bndb", delete=False)
        temporary.close()
        self.binary_path = Path(temporary.name)

    def tearDown(self):
        self.binary_path.unlink(missing_ok=True)
        with BinAssistMCPBinaryContextManager._open_operations_lock:
            BinAssistMCPBinaryContextManager._open_operations.clear()
            BinAssistMCPBinaryContextManager._active_open_paths.clear()

    def _ui_modules(self, ui_context, callbacks):
        ui_type = type(
            "UIContext",
            (),
            {"allContexts": staticmethod(lambda: [ui_context])},
        )
        ui_module = types.ModuleType("binaryninjaui")
        ui_module.UIContext = ui_type

        binaryninja_module = types.ModuleType("binaryninja")
        binaryninja_module.__path__ = []
        mainthread_module = types.ModuleType("binaryninja.mainthread")
        mainthread_module.execute_on_main_thread = callbacks.append
        mainthread_module.execute_on_main_thread_and_wait = lambda callback: callback()
        mainthread_module.is_main_thread = lambda: False
        binaryninja_module.mainthread = mainthread_module
        return {
            "binaryninjaui": ui_module,
            "binaryninja": binaryninja_module,
            "binaryninja.mainthread": mainthread_module,
        }

    def test_nonblocking_ui_open_returns_before_open_filename_runs(self):
        manager = BinAssistMCPBinaryContextManager()
        ui_context = FakeUIContext()
        callbacks = []

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), patch.dict(
            sys.modules, self._ui_modules(ui_context, callbacks)
        ):
            binary_name, status = manager.open_binary(
                str(self.binary_path), wait_for_analysis=False
            )

        self.assertIsNone(binary_name)
        self.assertEqual(status["status"], "opening")
        self.assertIsNotNone(status["operation_id"])
        self.assertEqual(status["requested_name"], self.binary_path.name)
        self.assertEqual(ui_context.open_calls, [])
        self.assertEqual(len(callbacks), 1)

    def test_duplicate_inflight_open_reuses_operation(self):
        manager = BinAssistMCPBinaryContextManager()
        ui_context = FakeUIContext()
        callbacks = []

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), patch.dict(
            sys.modules, self._ui_modules(ui_context, callbacks)
        ):
            _, first = manager.open_binary(
                str(self.binary_path), wait_for_analysis=False
            )
            _, second = manager.open_binary(
                str(self.binary_path), wait_for_analysis=False
            )

        self.assertEqual(first["operation_id"], second["operation_id"])
        self.assertEqual(len(callbacks), 1)
        self.assertIn("already", second["message"])

    def test_ui_open_failure_is_available_across_context_managers(self):
        manager = BinAssistMCPBinaryContextManager()
        ui_context = FakeUIContext(open_result=False)
        callbacks = []

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), patch.dict(
            sys.modules, self._ui_modules(ui_context, callbacks)
        ):
            _, queued = manager.open_binary(
                str(self.binary_path), wait_for_analysis=False
            )
            callbacks[0]()

        fresh_manager = BinAssistMCPBinaryContextManager()
        status = fresh_manager.get_binary_status(queued["operation_id"])

        self.assertEqual(status["status"], "failed")
        self.assertIn("rejected", status["error"])
        self.assertEqual(ui_context.open_calls, [str(self.binary_path.resolve())])

    def test_successful_ui_open_reports_final_context_name(self):
        manager = BinAssistMCPBinaryContextManager()
        binary_view = FakeBinaryView(str(self.binary_path.resolve()))
        ui_context = FakeUIContext(binary_view=binary_view)
        callbacks = []
        analysis_state = types.SimpleNamespace(IdleState="idle")

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), patch(
            "src.binassist_mcp.context.AnalysisState", analysis_state
        ), patch.dict(sys.modules, self._ui_modules(ui_context, callbacks)), patch(
            "src.binassist_mcp.context.threading.Thread", ImmediateThread
        ):
            _, queued = manager.open_binary(
                str(self.binary_path), wait_for_analysis=False
            )
            callbacks[0]()
            status = manager.get_binary_status(queued["operation_id"])

        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["name"], self.binary_path.name)
        self.assertEqual(status["function_count"], 1)

    def test_wait_for_analysis_true_preserves_blocking_open(self):
        manager = BinAssistMCPBinaryContextManager()
        binary_view = FakeBinaryView(str(self.binary_path.resolve()))
        ui_context = FakeUIContext(binary_view=binary_view)
        callbacks = []
        analysis_state = types.SimpleNamespace(IdleState="idle")

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), patch(
            "src.binassist_mcp.context.AnalysisState", analysis_state
        ), patch.dict(sys.modules, self._ui_modules(ui_context, callbacks)), patch(
            "src.binassist_mcp.context.time.sleep", lambda _: None
        ):
            binary_name, status = manager.open_binary(
                str(self.binary_path), wait_for_analysis=True
            )

        self.assertEqual(binary_name, self.binary_path.name)
        self.assertEqual(status["status"], "ready")
        self.assertTrue(status["analysis_complete"])
        self.assertEqual(callbacks, [])

    def test_headless_open_runs_on_worker_and_reports_ready(self):
        manager = BinAssistMCPBinaryContextManager()
        binary_view = FakeBinaryView(str(self.binary_path.resolve()))
        fake_bn = types.SimpleNamespace(load=lambda _: binary_view)
        analysis_state = types.SimpleNamespace(IdleState="idle")

        with patch("src.binassist_mcp.context.BINJA_AVAILABLE", True), patch(
            "src.binassist_mcp.context.AnalysisState", analysis_state
        ), patch("src.binassist_mcp.context.bn", fake_bn, create=True), patch(
            "src.binassist_mcp.context.threading.Thread", ImmediateThread
        ):
            binary_name, status = manager._queue_binary_open(
                resolved_path=str(self.binary_path.resolve()),
                resolved_bndb_path=None,
                is_bndb=True,
                ui_context_type=None,
            )

        self.assertEqual(binary_name, self.binary_path.name)
        self.assertEqual(status["status"], "ready")
        self.assertTrue(status["analysis_complete"])


if __name__ == "__main__":
    unittest.main()
