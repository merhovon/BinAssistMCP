"""
Binary context management for BinAssistMCP

This module provides context management for multiple Binary Ninja BinaryViews
with automatic name deduplication and lifecycle management.
"""

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple
from pathlib import Path

from .logging import log

try:
    import binaryninja as bn
    from binaryninja import AnalysisState
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False
    AnalysisState = None
    log.log_warn("Binary Ninja not available")


@dataclass
class BinaryInfo:
    """Information about a loaded binary"""
    name: str
    view: Optional[object]  # bn.BinaryView when available
    file_path: Optional[Path] = None
    load_time: Optional[float] = None
    analysis_complete: bool = False
    bndb_path: Optional[Path] = None  # Where to save .bndb after analysis
    bndb_saved: bool = False  # Whether .bndb has been saved
    
    def __post_init__(self):
        if self.file_path and isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        if self.bndb_path and isinstance(self.bndb_path, str):
            self.bndb_path = Path(self.bndb_path)


@dataclass
class OpenOperation:
    """State for a binary open that may outlive an MCP request."""

    operation_id: str
    file_path: str
    bndb_path: Optional[str]
    requested_name: str
    status: str = "queued"
    name: Optional[str] = None
    analysis_complete: bool = False
    function_count: int = 0
    bndb_saved: bool = False
    error: Optional[str] = None
    message: str = "Binary open queued."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Return a stable MCP response for this operation."""
        return {
            "operation_id": self.operation_id,
            "status": self.status,
            "name": self.name,
            "requested_name": self.requested_name,
            "file_path": self.file_path,
            "bndb_path": self.bndb_path,
            "analysis_complete": self.analysis_complete,
            "function_count": self.function_count,
            "bndb_saved": self.bndb_saved,
            "error": self.error,
            "message": self.message,
        }


class BinAssistMCPBinaryContextManager:
    """Context manager for multiple Binary Ninja BinaryViews"""

    # Open operations are shared across context-manager instances because some
    # MCP clients establish a fresh lifespan for each operation.
    _open_operations: ClassVar[Dict[str, OpenOperation]] = {}
    _active_open_paths: ClassVar[Dict[str, str]] = {}
    _open_operations_lock: ClassVar[threading.RLock] = threading.RLock()
    _max_open_operations: ClassVar[int] = 100
    _view_discovery_timeout: ClassVar[float] = 30.0
    _view_discovery_interval: ClassVar[float] = 0.5
    
    def __init__(self, max_binaries: int = 10):
        """Initialize the context manager

        Args:
            max_binaries: Maximum number of binaries to keep loaded
        """
        self.max_binaries = max_binaries
        self._binaries: Dict[str, BinaryInfo] = {}
        self._name_counter: Dict[str, int] = {}
        self._lock = threading.RLock()  # Thread safety for binary operations
        
    def add_binary(self, binary_view: object, name: Optional[str] = None) -> str:
        """Add a BinaryView to the context with automatic name deduplication

        Args:
            binary_view: The BinaryView to add
            name: Optional name to use (defaults to filename)

        Returns:
            The name used for the BinaryView
        """
        if not BINJA_AVAILABLE:
            raise RuntimeError("Binary Ninja not available")

        if name is None:
            name = self._extract_name(binary_view)

        # Sanitize name for URL usage
        sanitized_name = self._sanitize_name(name)

        with self._lock:
            # Deduplicate name if needed
            unique_name = self._get_unique_name(sanitized_name)

            # Check if we need to evict old binaries
            if len(self._binaries) >= self.max_binaries:
                self._evict_oldest_binary()

            # Add binary info
            binary_info = BinaryInfo(
                name=unique_name,
                view=binary_view,
                file_path=self._get_file_path(binary_view),
                load_time=time.time(),
                analysis_complete=self._is_analysis_complete(binary_view)
            )

            self._binaries[unique_name] = binary_info
            log.log_info(f"Added binary '{unique_name}' to context (total: {len(self._binaries)})")

            return unique_name
        
    def open_binary(self, file_path: str, bndb_path: Optional[str] = None,
                    wait_for_analysis: bool = True) -> Tuple[Optional[str], dict]:
        """Open a binary file or existing .bndb database in Binary Ninja.

        Uses the Binary Ninja UI to open the file (via UIContext.openFilename
        dispatched on the main thread), then syncs the context manager with
        the UI to pick up the new BinaryView. Falls back to headless bn.load()
        if no UI is available.

        For raw binaries (non-.bndb files): bndb_path is required. The .bndb
        database is saved automatically when analysis completes (in background).

        For existing .bndb files: bndb_path is ignored — the database is
        already on disk and will be opened directly.

        With wait_for_analysis=False, the open is queued and this method returns
        immediately with an operation ID. Use get_binary_status() with that ID
        to discover the final context name and monitor analysis. This avoids
        blocking while Binary Ninja deserializes a large database.

        When wait_for_analysis=True (legacy behavior), this method will
        block until analysis completes. This is NOT recommended for large
        binaries as it blocks the MCP server from handling other requests.

        Args:
            file_path: Path to the binary file or .bndb database to open
            bndb_path: Path where the .bndb database file should be saved
                       after analysis completes. Required for raw binaries,
                       ignored for .bndb files. The LLM should ask the user
                       for the desired save location before calling this tool.
            wait_for_analysis: Whether to wait for Binary Ninja's initial analysis
                               to complete before returning (default True).
                               Set to False for faster return on large binaries.
                               When False, analysis runs in background and .bndb
                               is saved automatically when analysis completes.

        Returns:
            Tuple of (binary_name, status_dict). For an asynchronous open,
            binary_name is None until get_binary_status() reports ``ready``.
            status_dict contains:
            - operation_id: Identifier used to poll an asynchronous open
            - status: queued, opening, discovering, ready, or failed
            - file_path: Path of the opened file
            - bndb_path: Path of the .bndb database (saved or pending)
            - analysis_complete: Whether analysis finished
            - function_count: Number of functions discovered so far
            - error: Error message if any step failed (partial success possible)

        Raises:
            RuntimeError: If Binary Ninja is not available
            FileNotFoundError: If the file_path does not exist
            ValueError: If the file could not be loaded by Binary Ninja,
                        or if bndb_path is missing for a raw binary
        """
        if not BINJA_AVAILABLE:
            raise RuntimeError("Binary Ninja not available")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        resolved_path = str(path.resolve())
        is_bndb = path.suffix.lower() == '.bndb'

        # bndb_path is required for raw binaries, ignored for .bndb files
        if not is_bndb and (not bndb_path or not bndb_path.strip()):
            raise ValueError(
                "bndb_path is required when opening a raw binary — you must "
                "specify where to save the analyzed .bndb database. Ask the "
                "user for the desired save location."
            )

        # Resolve bndb_path early so we can store it in BinaryInfo
        resolved_bndb_path = None
        if not is_bndb and bndb_path:
            resolved_bndb_path = Path(bndb_path).resolve()

        status = {
            "file_path": resolved_path,
            "bndb_path": resolved_path if is_bndb else (str(resolved_bndb_path) if resolved_bndb_path else None),
            "analysis_complete": False,
            "function_count": 0,
            "error": None,
        }

        log.log_info(f"Opening {'database' if is_bndb else 'binary'}: {file_path}")

        # Determine if the UI is available
        ui_available = False
        ui_context_type = None
        try:
            from binaryninjaui import UIContext
            if UIContext.allContexts():
                ui_available = True
                ui_context_type = UIContext
        except ImportError:
            pass

        if not wait_for_analysis:
            return self._queue_binary_open(
                resolved_path=resolved_path,
                resolved_bndb_path=resolved_bndb_path,
                is_bndb=is_bndb,
                ui_context_type=ui_context_type if ui_available else None,
            )

        bv = None

        if ui_available:
            # ----- UI mode: open via UIContext on the main thread -----
            from binaryninja.mainthread import execute_on_main_thread_and_wait, is_main_thread

            open_result = [False]

            def _open_in_ui():
                try:
                    ctx = UIContext.allContexts()
                    if ctx:
                        open_result[0] = ctx[0].openFilename(resolved_path)
                except Exception as e:
                    log.log_warn(f"UIContext.openFilename failed: {e}")
                    open_result[0] = False

            log.log_info("Opening binary via UIContext.openFilename on main thread...")
            if is_main_thread():
                _open_in_ui()
            else:
                execute_on_main_thread_and_wait(_open_in_ui)

            if not open_result[0]:
                raise ValueError(
                    f"Binary Ninja UI failed to open '{file_path}' — "
                    "unsupported format, corrupt file, or no UI context available"
                )

            log.log_info(f"UI opened {file_path}, waiting for view to appear in context...")

            # Wait for BN to load the view into a tab. We ONLY wait for the
            # view to appear — not for analysis to complete. This keeps the
            # polling loop short (typically < 5 seconds).
            max_wait_for_view = 30  # seconds — generous for slow file parsing
            poll_interval = 0.5
            elapsed = 0.0

            while elapsed < max_wait_for_view:
                time.sleep(poll_interval)
                elapsed += poll_interval

                # Sync to discover the newly opened view
                self.sync_with_binja()

                # Look for the binary we just opened by matching file path
                with self._lock:
                    for name, info in self._binaries.items():
                        if info.file_path and str(info.file_path) == resolved_path:
                            bv = info.view
                            break

                if bv is not None:
                    break

            if bv is None:
                raise ValueError(
                    f"Binary '{file_path}' was opened in the UI but did not appear "
                    f"in context after {max_wait_for_view}s — this should not happen"
                )

            # Store bndb_path in BinaryInfo for deferred save
            if resolved_bndb_path:
                with self._lock:
                    for name, info in self._binaries.items():
                        if info.view is bv:
                            info.bndb_path = resolved_bndb_path
                            break

            # Optionally wait for analysis (blocking — NOT recommended)
            if wait_for_analysis:
                log.log_info(f"wait_for_analysis=True, blocking until analysis completes...")
                max_analysis_wait = 120
                analysis_elapsed = 0.0
                while analysis_elapsed < max_analysis_wait:
                    if self._is_analysis_complete(bv):
                        log.log_info(f"Analysis complete for {file_path}")
                        break
                    time.sleep(poll_interval)
                    analysis_elapsed += poll_interval

        else:
            # ----- Headless mode: fall back to bn.load() -----
            log.log_info("No UI available, loading binary headlessly via bn.load()...")
            try:
                bv = bn.load(resolved_path)
            except Exception as e:
                raise ValueError(f"Binary Ninja failed to load '{file_path}': {e}")

            if bv is None:
                raise ValueError(
                    f"Binary Ninja returned None for '{file_path}' — "
                    "unsupported format or corrupt file"
                )

            log.log_info(f"Binary loaded headlessly: {file_path} (arch={getattr(bv, 'arch', 'unknown')})")

            if wait_for_analysis:
                try:
                    log.log_info(f"Waiting for analysis to complete on {file_path}...")
                    bv.update_analysis_and_wait()
                    log.log_info(f"Analysis complete for {file_path}")
                except Exception as e:
                    log.log_warn(f"Analysis wait failed (non-fatal): {e}")
                    status["error"] = f"Analysis wait failed: {e}"

            # Add to context manager (in UI mode, sync_with_binja already did this)
            self.add_binary(bv)

            # Store bndb_path in BinaryInfo for deferred save
            if resolved_bndb_path:
                with self._lock:
                    for name, info in self._binaries.items():
                        if info.view is bv:
                            info.bndb_path = resolved_bndb_path
                            break

        # --- Common post-open logic ---

        analysis_done = self._is_analysis_complete(bv)
        status["analysis_complete"] = analysis_done

        # Count functions
        try:
            status["function_count"] = len(list(bv.functions))
        except Exception:
            pass

        # Save .bndb immediately if analysis is already complete,
        # otherwise it will be saved automatically when analysis finishes
        # (triggered by get_analysis_progress polling)
        if not is_bndb and resolved_bndb_path:
            if analysis_done:
                try:
                    resolved_bndb_path.parent.mkdir(parents=True, exist_ok=True)
                    bv.create_database(str(resolved_bndb_path))
                    status["bndb_path"] = str(resolved_bndb_path)
                    with self._lock:
                        for name, info in self._binaries.items():
                            if info.view is bv:
                                info.bndb_saved = True
                                break
                    log.log_info(f"Saved .bndb database to {bndb_path}")
                except Exception as e:
                    log.log_warn(f"Failed to save .bndb to '{bndb_path}': {e}")
                    if status["error"]:
                        status["error"] += f"; bndb save failed: {e}"
                    else:
                        status["error"] = f"bndb save failed: {e}"
            else:
                # Analysis not done — .bndb will be saved when analysis completes
                log.log_info(
                    f".bndb save deferred — will auto-save to {bndb_path} "
                    f"when analysis completes"
                )

        # Find the name this binary was registered under
        binary_name = None
        with self._lock:
            for name, info in self._binaries.items():
                if info.view is bv:
                    binary_name = name
                    break

        if binary_name is None:
            binary_name = self._extract_name(bv)

        log.log_info(
            f"Binary '{binary_name}' ready in context "
            f"(functions={status['function_count']}, "
            f"analysis_complete={status['analysis_complete']})"
        )

        status["operation_id"] = None
        status["status"] = "ready"
        status["message"] = f"Binary '{binary_name}' is ready."
        return binary_name, status

    @staticmethod
    def _path_key(file_path: str) -> str:
        """Normalize a path for operation deduplication and view matching."""
        try:
            return os.path.normcase(str(Path(file_path).resolve()))
        except Exception:
            return os.path.normcase(str(file_path))

    @classmethod
    def _prune_open_operations(cls) -> None:
        """Bound retained completed operation history."""
        if len(cls._open_operations) < cls._max_open_operations:
            return

        completed = sorted(
            (
                operation for operation in cls._open_operations.values()
                if operation.status in ("ready", "failed")
            ),
            key=lambda operation: operation.updated_at,
        )
        remove_count = len(cls._open_operations) - cls._max_open_operations + 1
        for operation in completed[:remove_count]:
            cls._open_operations.pop(operation.operation_id, None)

    @classmethod
    def _update_open_operation(cls, operation_id: str, **changes) -> Optional[dict]:
        """Atomically update an open operation and return its snapshot."""
        with cls._open_operations_lock:
            operation = cls._open_operations.get(operation_id)
            if operation is None:
                return None

            for key, value in changes.items():
                setattr(operation, key, value)
            operation.updated_at = time.time()

            if operation.status in ("ready", "failed"):
                path_key = cls._path_key(operation.file_path)
                if cls._active_open_paths.get(path_key) == operation_id:
                    cls._active_open_paths.pop(path_key, None)

            return operation.to_dict()

    @classmethod
    def _get_open_operation(cls, identifier: str) -> Optional[OpenOperation]:
        """Find an operation by ID, final name, requested name, or path."""
        with cls._open_operations_lock:
            direct = cls._open_operations.get(identifier)
            if direct is not None:
                return direct

            path_key = cls._path_key(identifier)
            candidates = sorted(
                cls._open_operations.values(),
                key=lambda operation: operation.created_at,
                reverse=True,
            )
            for operation in candidates:
                if identifier in (operation.name, operation.requested_name):
                    return operation
                if path_key == cls._path_key(operation.file_path):
                    return operation
            return None

    def _queue_binary_open(self, resolved_path: str,
                           resolved_bndb_path: Optional[Path],
                           is_bndb: bool,
                           ui_context_type: Optional[object]) -> Tuple[Optional[str], dict]:
        """Queue a UI or headless open without waiting for file loading."""
        path_key = self._path_key(resolved_path)
        with self._open_operations_lock:
            active_id = self._active_open_paths.get(path_key)
            if active_id:
                active = self._open_operations.get(active_id)
                if active and active.status not in ("ready", "failed"):
                    snapshot = active.to_dict()
                    snapshot["message"] = "This binary is already being opened."
                    return active.name, snapshot

            self._prune_open_operations()
            operation_id = str(uuid.uuid4())
            operation = OpenOperation(
                operation_id=operation_id,
                file_path=resolved_path,
                bndb_path=(
                    resolved_path if is_bndb
                    else str(resolved_bndb_path) if resolved_bndb_path else None
                ),
                requested_name=self._sanitize_name(Path(resolved_path).name),
            )
            self._open_operations[operation_id] = operation
            self._active_open_paths[path_key] = operation_id

        if ui_context_type is not None:
            self._queue_ui_open(operation_id, ui_context_type)
        else:
            self._queue_headless_open(operation_id)

        with self._open_operations_lock:
            snapshot = self._open_operations[operation_id].to_dict()
        return snapshot["name"], snapshot

    def _queue_ui_open(self, operation_id: str, ui_context_type: object) -> None:
        """Schedule UIContext.openFilename without waiting for it to return."""
        operation = self._get_open_operation(operation_id)
        if operation is None:
            return

        try:
            from binaryninja.mainthread import execute_on_main_thread
        except Exception as exc:
            error = f"Could not access Binary Ninja's main-thread scheduler: {exc}"
            log.log_warn(error)
            self._update_open_operation(
                operation_id,
                status="failed",
                error=error,
                message=error,
            )
            return

        self._update_open_operation(
            operation_id,
            status="opening",
            message="Binary Ninja is opening the file in the UI.",
        )

        def _open_in_ui():
            try:
                contexts = ui_context_type.allContexts()
                if not contexts:
                    raise RuntimeError("no UI context is available")
                if not contexts[0].openFilename(operation.file_path):
                    raise RuntimeError("Binary Ninja rejected the file")
            except Exception as exc:
                error = f"Binary Ninja UI failed to open '{operation.file_path}': {exc}"
                log.log_warn(error)
                self._update_open_operation(
                    operation_id,
                    status="failed",
                    error=error,
                    message=error,
                )
                return

            self._update_open_operation(
                operation_id,
                status="discovering",
                message="The file is open; waiting for its BinaryView to appear.",
            )
            monitor = threading.Thread(
                target=self._finalize_ui_open_operation,
                args=(operation_id,),
                name=f"binary-open-{operation_id[:8]}",
                daemon=True,
            )
            monitor.start()

        try:
            execute_on_main_thread(_open_in_ui)
        except Exception as exc:
            error = f"Could not schedule Binary Ninja UI open: {exc}"
            log.log_warn(error)
            self._update_open_operation(
                operation_id,
                status="failed",
                error=error,
                message=error,
            )

    def _queue_headless_open(self, operation_id: str) -> None:
        """Run the blocking bn.load call on a dedicated worker thread."""
        self._update_open_operation(
            operation_id,
            status="opening",
            message="Binary Ninja is opening the file headlessly.",
        )
        worker = threading.Thread(
            target=self._open_headless_operation,
            args=(operation_id,),
            name=f"binary-open-{operation_id[:8]}",
            daemon=True,
        )
        worker.start()

    def _open_headless_operation(self, operation_id: str) -> None:
        """Complete a queued headless open."""
        operation = self._get_open_operation(operation_id)
        if operation is None:
            return

        try:
            bv = bn.load(operation.file_path)
            if bv is None:
                raise RuntimeError("Binary Ninja returned no BinaryView")
            binary_name = self.add_binary(bv)
            self._complete_open_operation(operation_id, binary_name, bv)
        except Exception as exc:
            error = f"Binary Ninja failed to load '{operation.file_path}': {exc}"
            log.log_warn(error)
            self._update_open_operation(
                operation_id,
                status="failed",
                error=error,
                message=error,
            )

    def _find_binary_by_path(self, file_path: str) -> Tuple[Optional[str], Optional[object]]:
        """Return the registered name and view matching a resolved path."""
        path_key = self._path_key(file_path)
        with self._lock:
            for name, info in self._binaries.items():
                if info.file_path and self._path_key(str(info.file_path)) == path_key:
                    return name, info.view
        return None, None

    def _finalize_ui_open_operation(self, operation_id: str) -> None:
        """Discover and register a view after the UI open callback finishes."""
        operation = self._get_open_operation(operation_id)
        if operation is None:
            return

        deadline = time.monotonic() + self._view_discovery_timeout
        while time.monotonic() < deadline:
            self.sync_with_binja()
            binary_name, bv = self._find_binary_by_path(operation.file_path)
            if binary_name is not None and bv is not None:
                self._complete_open_operation(operation_id, binary_name, bv)
                return
            time.sleep(self._view_discovery_interval)

        error = (
            f"Binary '{operation.file_path}' opened in the UI but its BinaryView "
            f"did not appear within {self._view_discovery_timeout:g} seconds"
        )
        log.log_warn(error)
        self._update_open_operation(
            operation_id,
            status="failed",
            error=error,
            message=error,
        )

    def _complete_open_operation(self, operation_id: str, binary_name: str,
                                 bv: object) -> None:
        """Attach the loaded view and mark an asynchronous open ready."""
        operation = self._get_open_operation(operation_id)
        if operation is None:
            return

        target_bndb = None
        if operation.bndb_path and not operation.file_path.lower().endswith(".bndb"):
            target_bndb = Path(operation.bndb_path)

        with self._lock:
            info = self._binaries.get(binary_name)
            if info is not None and target_bndb is not None:
                info.bndb_path = target_bndb

        analysis_complete = self._is_analysis_complete(bv)
        try:
            function_count = len(bv.functions)
        except Exception:
            function_count = 0

        self._update_open_operation(
            operation_id,
            status="ready",
            name=binary_name,
            analysis_complete=analysis_complete,
            function_count=function_count,
            bndb_saved=operation.file_path.lower().endswith(".bndb"),
            message=(
                f"Binary '{binary_name}' is ready."
                if analysis_complete else
                f"Binary '{binary_name}' is open and analysis is in progress."
            ),
        )

        if analysis_complete and target_bndb is not None:
            self._save_bndb_async(binary_name, bv, target_bndb)

    def get_binary_status(self, identifier: str) -> dict:
        """Get open and analysis status by operation ID, name, or file path."""
        operation = self._get_open_operation(identifier)
        if operation is not None:
            if operation.status == "discovering":
                self.sync_with_binja()
                binary_name, bv = self._find_binary_by_path(operation.file_path)
                if binary_name is not None and bv is not None:
                    self._complete_open_operation(operation.operation_id, binary_name, bv)

            if operation.status == "ready":
                self.sync_with_binja()
                binary_name, bv = self._find_binary_by_path(operation.file_path)
                if binary_name is not None and bv is not None:
                    if operation.bndb_path and not operation.file_path.lower().endswith(".bndb"):
                        with self._lock:
                            info = self._binaries.get(binary_name)
                            if info is not None:
                                info.bndb_path = Path(operation.bndb_path)
                    progress = self.get_analysis_progress(binary_name)
                    self._update_open_operation(
                        operation.operation_id,
                        name=binary_name,
                        analysis_complete=progress["analysis_complete"],
                        function_count=progress["function_count"],
                        bndb_saved=progress["bndb_saved"],
                        message=progress["message"],
                    )

            with self._open_operations_lock:
                current = self._open_operations.get(operation.operation_id)
                return current.to_dict() if current else {
                    "operation_id": operation.operation_id,
                    "status": "unknown",
                    "error": "Open operation is no longer available",
                }

        # Compatibility path for binaries that were opened manually or by an
        # earlier BinAssistMCP version.
        try:
            binary_info = self.get_binary_info(identifier)
            progress = self.get_analysis_progress(identifier)
            return {
                "operation_id": None,
                "status": "ready",
                "name": binary_info.name,
                "requested_name": binary_info.name,
                "file_path": str(binary_info.file_path) if binary_info.file_path else None,
                "bndb_path": str(binary_info.bndb_path) if binary_info.bndb_path else None,
                "analysis_complete": progress["analysis_complete"],
                "function_count": progress["function_count"],
                "bndb_saved": progress["bndb_saved"],
                "error": None,
                "message": progress["message"],
            }
        except KeyError as exc:
            return {
                "operation_id": identifier,
                "status": "not_found",
                "name": None,
                "analysis_complete": False,
                "function_count": 0,
                "bndb_saved": False,
                "error": str(exc),
                "message": str(exc),
            }

    def _ensure_binary_registered(self, name: str) -> None:
        """Lazily refresh Binary Ninja UI state when a cached binary is unavailable.

        Some MCP clients create a new lifespan for each operation. Those lifespans
        start with an empty context even though Binary Ninja still has open views,
        so filename-based operations must not depend on list_binaries being called
        first in the same session.
        """
        with self._lock:
            binary_info = self._binaries.get(name)
            if binary_info and self._is_binary_valid(binary_info.view):
                return

            if binary_info:
                del self._binaries[name]
                log.log_warn(f"Binary '{name}' is no longer valid, refreshing context")

        sync_result = self.sync_with_binja()
        if sync_result.get("error"):
            log.log_debug(
                f"Could not refresh Binary Ninja context for '{name}': "
                f"{sync_result['error']}"
            )

    def get_binary(self, name: str) -> object:
        """Get a BinaryView by name

        Args:
            name: The name of the BinaryView

        Returns:
            The BinaryView if found

        Raises:
            KeyError: If the binary is not found
        """
        self._ensure_binary_registered(name)

        with self._lock:
            if name not in self._binaries:
                available = ", ".join(self._binaries.keys()) if self._binaries else "none"
                raise KeyError(f"Binary '{name}' not found. Available: {available}")

            binary_info = self._binaries[name]

            # Verify the binary view is still valid
            if not self._is_binary_valid(binary_info.view):
                log.log_warn(f"Binary '{name}' is no longer valid, removing from context")
                del self._binaries[name]
                raise KeyError(f"Binary '{name}' is no longer valid")

            return binary_info.view
        
    def get_binary_info(self, name: str) -> BinaryInfo:
        """Get binary information by name

        Args:
            name: The name of the binary

        Returns:
            BinaryInfo object

        Raises:
            KeyError: If the binary is not found
        """
        self._ensure_binary_registered(name)

        with self._lock:
            if name not in self._binaries:
                available = ", ".join(self._binaries.keys()) if self._binaries else "none"
                raise KeyError(f"Binary '{name}' not found. Available: {available}")

            return self._binaries[name]

    def list_binaries(self) -> List[str]:
        """List all loaded binary names

        Returns:
            List of binary names
        """
        with self._lock:
            return list(self._binaries.keys())

    def list_binary_info(self) -> Dict[str, BinaryInfo]:
        """Get information about all loaded binaries

        Returns:
            Dictionary mapping names to BinaryInfo objects
        """
        with self._lock:
            return self._binaries.copy()

    def remove_binary(self, name: str) -> bool:
        """Remove a binary from the context

        Args:
            name: Name of the binary to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._binaries:
                del self._binaries[name]
                log.log_info(f"Removed binary '{name}' from context")
                return True
            return False

    def clear(self):
        """Clear all binaries from the context"""
        with self._lock:
            count = len(self._binaries)
            self._binaries.clear()
            self._name_counter.clear()
            log.log_info(f"Cleared {count} binaries from context")

    def update_analysis_status(self, name: str):
        """Update the analysis status for a binary

        Args:
            name: Name of the binary to update
        """
        with self._lock:
            if name in self._binaries:
                binary_info = self._binaries[name]
                if binary_info.view:
                    binary_info.analysis_complete = self._is_analysis_complete(binary_info.view)
                    log.log_debug(f"Updated analysis status for '{name}': {binary_info.analysis_complete}")

    def get_analysis_progress(self, name: str) -> dict:
        """Get detailed analysis progress for a binary.

        Returns real-time progress information including state description
        and function count. This is safe to call frequently and does not
        block the event loop.

        Args:
            name: Name of the binary

        Returns:
            Dictionary with:
            - state: Human-readable analysis state string
            - analyzing: bool, True if analysis is still running
            - analysis_complete: bool, True if analysis finished
            - function_count: Number of functions discovered so far
            - bndb_saved: Whether the .bndb has been saved
            - message: Descriptive status message for LLM consumption
        """
        result = {
            "state": "unknown",
            "analyzing": False,
            "analysis_complete": False,
            "function_count": 0,
            "bndb_saved": False,
            "message": "",
        }

        self._ensure_binary_registered(name)

        with self._lock:
            if name not in self._binaries:
                result["state"] = "not_found"
                result["message"] = f"Binary '{name}' not found in context."
                return result

            info = self._binaries[name]
            result["bndb_saved"] = info.bndb_saved

        bv = info.view
        if not bv or not BINJA_AVAILABLE or not AnalysisState:
            result["state"] = "no_view"
            result["message"] = "Binary view is not available."
            return result

        # Get function count (safe even during analysis)
        try:
            result["function_count"] = len(list(bv.functions))
        except Exception:
            pass

        # Read analysis state from BinaryView
        try:
            if hasattr(bv, 'analysis_progress'):
                progress = bv.analysis_progress
                current_state = progress.state

                if current_state == AnalysisState.IdleState:
                    result["state"] = "idle"
                    result["analyzing"] = False
                    result["analysis_complete"] = True
                    result["message"] = (
                        f"Analysis complete. {result['function_count']} functions discovered."
                    )
                    # Update the cached flag
                    with self._lock:
                        if name in self._binaries:
                            self._binaries[name].analysis_complete = True
                            # Auto-save .bndb if analysis just finished and we haven't saved yet
                            if not info.bndb_saved and info.bndb_path:
                                self._save_bndb_async(name, bv, info.bndb_path)
                else:
                    result["state"] = str(current_state)
                    result["analyzing"] = True
                    result["analysis_complete"] = False
                    result["message"] = (
                        f"Analysis in progress ({result['state']}). "
                        f"{result['function_count']} functions discovered so far. "
                        f"Use get_binary_status to check progress. "
                        f"Some tools may return incomplete results."
                    )
            else:
                result["state"] = "unknown"
                result["message"] = "Cannot determine analysis state."
        except Exception as e:
            result["state"] = "error"
            result["message"] = f"Error checking analysis progress: {e}"
            log.log_debug(f"Error in get_analysis_progress for '{name}': {e}")

        return result

    def _save_bndb_async(self, name: str, bv: object, bndb_path: Path):
        """Save .bndb database in a background thread (non-blocking).

        Called automatically when analysis completes for a binary that
        was opened with a bndb_path specified.

        Args:
            name: Binary name (for logging)
            bv: The BinaryView to save
            bndb_path: Where to save the .bndb file
        """
        def _do_save():
            try:
                bndb_path.parent.mkdir(parents=True, exist_ok=True)
                bv.create_database(str(bndb_path.resolve()))
                with self._lock:
                    if name in self._binaries:
                        self._binaries[name].bndb_saved = True
                log.log_info(f"Saved .bndb database to {bndb_path}")
            except Exception as e:
                log.log_warn(f"Failed to save .bndb to '{bndb_path}': {e}")

        save_thread = threading.Thread(
            target=_do_save, name=f"bndb-save-{name}", daemon=True
        )
        save_thread.start()

    def _extract_name(self, binary_view: object) -> str:
        """Extract name from a BinaryView"""
        if not BINJA_AVAILABLE or not binary_view:
            return "unknown"
            
        try:
            if hasattr(binary_view, 'file') and hasattr(binary_view.file, 'filename'):
                filename = binary_view.file.filename
                if filename:
                    return Path(filename).name
                    
            if hasattr(binary_view, 'name'):
                return binary_view.name
                
        except Exception as e:
            log.log_warn(f"Failed to extract name from binary view: {e}")
            
        return "unknown"
        
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for URL usage"""
        if not name:
            return "unnamed"
            
        # Replace invalid characters
        invalid_chars = '/\\:*?"<>| '
        for char in invalid_chars:
            name = name.replace(char, '_')
            
        # Remove leading/trailing dots and underscores
        name = name.strip('_.')
        
        # Ensure non-empty name
        if not name:
            name = "unnamed"
            
        return name
        
    def _get_unique_name(self, base_name: str) -> str:
        """Get a unique name by adding a counter if needed"""
        if base_name not in self._binaries:
            return base_name
            
        # Find the next available counter value
        counter = self._name_counter.get(base_name, 1)
        while True:
            unique_name = f"{base_name}_{counter}"
            if unique_name not in self._binaries:
                self._name_counter[base_name] = counter + 1
                return unique_name
            counter += 1
            
    def _get_file_path(self, binary_view: object) -> Optional[Path]:
        """Get file path from a BinaryView"""
        if not BINJA_AVAILABLE or not binary_view:
            return None
            
        try:
            if hasattr(binary_view, 'file') and hasattr(binary_view.file, 'filename'):
                filename = binary_view.file.filename
                if filename:
                    return Path(filename)
        except Exception as e:
            log.log_debug(f"Failed to get file path: {e}")
            
        return None
        
    def _is_analysis_complete(self, binary_view: object) -> bool:
        """Check if analysis is complete for a BinaryView"""
        if not BINJA_AVAILABLE or not binary_view or not AnalysisState:
            return False
            
        try:
            # Method 1: Check analysis_progress state
            if hasattr(binary_view, 'analysis_progress'):
                progress = binary_view.analysis_progress
                current_state = progress.state
                log.log_debug(f"Analysis progress state: {current_state} (IdleState={AnalysisState.IdleState})")
                # Correct API: compare state directly to AnalysisState.IdleState
                return current_state == AnalysisState.IdleState
                
            # Method 2: Check analysis_info state (alternative)
            if hasattr(binary_view, 'analysis_info'):
                info_state = binary_view.analysis_info.state
                log.log_debug(f"Analysis info state: {info_state} (IdleState={AnalysisState.IdleState})")
                return info_state == AnalysisState.IdleState
                
            # Fallback: check if we have functions
            if hasattr(binary_view, 'functions'):
                func_count = len(list(binary_view.functions))
                log.log_debug(f"Analysis status fallback: {func_count} functions found")
                return func_count > 0
                
        except Exception as e:
            log.log_debug(f"Failed to check analysis status: {e}")
            # Additional debug info
            try:
                if hasattr(binary_view, 'analysis_progress'):
                    progress = binary_view.analysis_progress
                    log.log_debug(f"Progress object type: {type(progress)}")
                    log.log_debug(f"Progress state type: {type(progress.state)}")
                    log.log_debug(f"Available AnalysisState values: {[attr for attr in dir(AnalysisState) if not attr.startswith('_')]}")
            except Exception as debug_error:
                log.log_debug(f"Failed to get debug info: {debug_error}")
            
        return False
        
    def _is_binary_valid(self, binary_view: object) -> bool:
        """Check if a BinaryView is still valid"""
        if not BINJA_AVAILABLE or not binary_view:
            return False
            
        try:
            # Try to access a basic property
            if hasattr(binary_view, 'file'):
                _ = binary_view.file
                return True
        except Exception as e:
            log.log_debug(f"Binary view validation failed: {e}")
            
        return False
        
    def _evict_oldest_binary(self):
        """Evict the oldest binary to make room for a new one"""
        if not self._binaries:
            return
            
        # Find the binary with the oldest load time
        oldest_name = None
        oldest_time = float('inf')
        
        for name, binary_info in self._binaries.items():
            if binary_info.load_time and binary_info.load_time < oldest_time:
                oldest_time = binary_info.load_time
                oldest_name = name
                
        if oldest_name:
            log.log_info(f"Evicting oldest binary '{oldest_name}' to make room")
            del self._binaries[oldest_name]
            
    def __len__(self) -> int:
        """Return the number of loaded binaries"""
        with self._lock:
            return len(self._binaries)

    def __contains__(self, name: str) -> bool:
        """Check if a binary name is in the context"""
        with self._lock:
            return name in self._binaries

    def __repr__(self) -> str:
        """String representation of the context manager"""
        with self._lock:
            return f"BinaryContextManager(binaries={len(self._binaries)}, max={self.max_binaries})"

    def sync_with_binja(self) -> dict:
        """Synchronize context with Binary Ninja's currently open views.

        Enumerates all open BinaryViews via Binary Ninja UI context,
        adds newly opened binaries to context, and removes closed/invalid
        binaries from context.

        Returns:
            Dictionary with sync status report:
            - added: list of newly added binary names
            - removed: list of removed binary names
            - unchanged: list of binaries that remained
            - synced: bool indicating if sync was performed
            - error: optional error message if sync failed
        """
        result = {
            "added": [],
            "removed": [],
            "unchanged": [],
            "synced": False,
            "error": None
        }

        if not BINJA_AVAILABLE:
            result["error"] = "Binary Ninja not available"
            return result

        # Try to access UI context for open views
        try:
            from binaryninjaui import UIContext
            ui_available = True
        except ImportError:
            ui_available = False
            log.log_debug("binaryninjaui not available, running in headless mode")

        with self._lock:
            # First, remove invalid/closed binaries from context
            names_to_remove = []
            for name, binary_info in self._binaries.items():
                if not self._is_binary_valid(binary_info.view):
                    names_to_remove.append(name)

            for name in names_to_remove:
                del self._binaries[name]
                result["removed"].append(name)
                log.log_info(f"Removed invalid/closed binary '{name}' from context")

            # If UI is available, enumerate open views and add new ones
            if ui_available:
                try:
                    # Collect all open binary views from all UI contexts
                    open_views = []

                    # Get all UI contexts (windows)
                    all_contexts = UIContext.allContexts()

                    for ctx in all_contexts:
                        if ctx is None:
                            continue

                        # Use getTabs() to enumerate all open tabs and get their BinaryViews
                        # Note: getAvailableBinaryViews() returns underlying raw views which
                        # may not have proper filenames, so we use tab-based enumeration instead
                        if hasattr(ctx, 'getTabs'):
                            tabs = ctx.getTabs()
                            for tab in (tabs or []):
                                # Each tab has getCurrentBinaryView() method
                                if hasattr(tab, 'getCurrentBinaryView'):
                                    bv = tab.getCurrentBinaryView()
                                    if bv is not None and bv not in open_views:
                                        open_views.append(bv)

                    # Build a set of file paths currently in context for comparison
                    context_paths = set()
                    for binary_info in self._binaries.values():
                        if binary_info.file_path:
                            context_paths.add(str(binary_info.file_path))

                    # Add any open views not already in context
                    for bv in open_views:
                        if bv is None:
                            continue

                        file_path = self._get_file_path(bv)
                        file_path_str = str(file_path) if file_path else None

                        # Check if this view is already in context (by path)
                        if file_path_str and file_path_str in context_paths:
                            continue

                        # Check if view object is already tracked
                        already_tracked = False
                        for binary_info in self._binaries.values():
                            if binary_info.view is bv:
                                already_tracked = True
                                break

                        if already_tracked:
                            continue

                        # Add this new binary
                        try:
                            name = self._extract_name(bv)
                            sanitized_name = self._sanitize_name(name)
                            unique_name = self._get_unique_name(sanitized_name)

                            # Check if we need to evict old binaries
                            if len(self._binaries) >= self.max_binaries:
                                self._evict_oldest_binary()

                            binary_info = BinaryInfo(
                                name=unique_name,
                                view=bv,
                                file_path=file_path,
                                load_time=time.time(),
                                analysis_complete=self._is_analysis_complete(bv)
                            )

                            self._binaries[unique_name] = binary_info
                            result["added"].append(unique_name)
                            log.log_info(f"Added newly opened binary '{unique_name}' to context")
                        except Exception as add_error:
                            log.log_warn(f"Failed to add binary view to context: {add_error}")
                except Exception as ui_error:
                    result["error"] = f"UI context error: {ui_error}"

            # Record unchanged binaries
            for name in self._binaries.keys():
                if name not in result["added"]:
                    result["unchanged"].append(name)

            result["synced"] = True

        return result
