"""
Opens the computer's native "choose a file or folder" window. Every part of
MySelf that needs to browse the disk (models, dataset files, dataset
folders, ...) calls into this one module, so Browse always looks and behaves
the same way everywhere in the app.
"""

import concurrent.futures
import contextlib
import os
import shutil
import subprocess

# The dialog window only works if it's always opened from the same one
# thread - opening it from a different thread each time can crash the whole
# server, not just that request. This background thread handles every
# dialog, so that never happens.
_DIALOG_THREAD = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="fs-dialog")


def _run_on_dialog_thread(func, *args):
    return _DIALOG_THREAD.submit(func, *args).result()


QA_FILETYPES = [
    ("JSON", "*.json"),
    ("JSON Lines", "*.jsonl"),
    ("CSV", "*.csv"),
    ("All files", "*.*"),
]

DATASET_FOLDER_FILETYPES = [
    ("JSON", "*.json"),
    ("JSON Lines", "*.jsonl"),
    ("CSV", "*.csv"),
    ("Text notes", "*.txt *.md"),
    ("All files", "*.*"),
]


def _hide_dotfiles(root) -> None:
    """On Linux without a native file picker, Tk falls back to its own
    plain dialog, which shows hidden dotfiles/dotfolders by default. This
    switches that off (macOS/Windows use the real OS dialog instead, which
    already hides them, so this has nothing to do there)."""
    import tkinter as tk

    try:
        root.tk.call("tk_getOpenFile", "-foobarbaz")
    except tk.TclError:
        pass  # expected: this loads Tk's dialog script without opening a real window

    try:
        root.tk.call("set", "::tk::dialog::file::showHiddenBtn", "1")
        root.tk.call("set", "::tk::dialog::file::showHiddenVar", "0")
    except tk.TclError:
        pass


@contextlib.contextmanager
def _tk_root():
    """A hidden helper window needed to open a native file dialog."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    _hide_dotfiles(root)
    try:
        yield root
    finally:
        root.destroy()


def _resolve_initial_dir(path: str) -> str:
    """Figure out which folder a dialog should open in, falling back to the home folder."""
    if path and os.path.isdir(path):
        return os.path.abspath(path)
    if path and os.path.isfile(path):
        return os.path.dirname(os.path.abspath(path))
    return os.path.expanduser("~")


def _ensure_extension(path: str | None, default_extension: str) -> str | None:
    """If the user typed a filename with no extension at all, add the default one."""
    if not path or os.path.splitext(path)[1]:
        return path
    return path + default_extension


def _zenity_filter_args(filetypes: list) -> list:
    return [f"--file-filter={label} | {patterns}" for label, patterns in filetypes]


def _kdialog_pattern(filetypes: list) -> str:
    return "\n".join(f"{patterns}|{label}" for label, patterns in filetypes)


def pick_folder(initial_dir: str = "", title: str = "Select a folder") -> str | None:
    """Open a folder picker window and return the folder the user chose (or None if they cancelled)."""
    return _run_on_dialog_thread(_pick_folder_impl, initial_dir, title)


def _pick_folder_impl(initial_dir: str, title: str) -> str | None:
    initial_dir = _resolve_initial_dir(initial_dir)

    try:
        from tkinter import filedialog
        with _tk_root():
            selected = filedialog.askdirectory(initialdir=initial_dir, title=title)
        return selected or None
    except ImportError:
        pass

    if shutil.which("zenity"):
        result = subprocess.run(
            ["zenity", "--file-selection", "--directory", f"--title={title}",
             f"--filename={initial_dir}/"],
            capture_output=True, text=True,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None

    if shutil.which("kdialog"):
        result = subprocess.run(
            ["kdialog", "--getexistingdirectory", initial_dir],
            capture_output=True, text=True,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None

    return None


def pick_save_file(
    initial_path: str = "",
    title: str = "Choose or create a file",
    filetypes: list | None = None,
    default_extension: str = ".json",
) -> str | None:
    """Open a dialog for picking an existing file, or naming a new one to create."""
    return _run_on_dialog_thread(_pick_save_file_impl, initial_path, title, filetypes, default_extension)


def _pick_save_file_impl(
    initial_path: str, title: str, filetypes: list | None, default_extension: str
) -> str | None:
    initial_dir = _resolve_initial_dir(initial_path)
    filetypes = filetypes or QA_FILETYPES

    try:
        from tkinter import filedialog
        with _tk_root():
            selected = filedialog.asksaveasfilename(
                initialdir=initial_dir, title=title,
                filetypes=filetypes, defaultextension=default_extension,
            )
        return _ensure_extension(selected or None, default_extension)
    except ImportError:
        pass

    if shutil.which("zenity"):
        result = subprocess.run(
            ["zenity", "--file-selection", "--save", f"--title={title}",
             f"--filename={initial_dir}/", *_zenity_filter_args(filetypes)],
            capture_output=True, text=True,
        )
        path = result.stdout.strip() or None if result.returncode == 0 else None
        return _ensure_extension(path, default_extension)

    if shutil.which("kdialog"):
        result = subprocess.run(
            ["kdialog", "--getsavefilename", initial_dir, _kdialog_pattern(filetypes)],
            capture_output=True, text=True,
        )
        path = result.stdout.strip() or None if result.returncode == 0 else None
        return _ensure_extension(path, default_extension)

    return None


def pick_folder_by_browsing_files(
    initial_dir: str = "",
    title: str = "Open any file inside the folder you want to use",
    filetypes: list | None = None,
) -> str | None:
    """A plain folder picker hides regular files, which makes it hard to tell
    folders apart by what's actually inside them. This opens a normal file
    browser instead - so you can see the files as you navigate - and hands
    back the folder that held whichever file you picked."""
    return _run_on_dialog_thread(_pick_folder_by_browsing_files_impl, initial_dir, title, filetypes)


def _pick_folder_by_browsing_files_impl(
    initial_dir: str, title: str, filetypes: list | None
) -> str | None:
    initial_dir = _resolve_initial_dir(initial_dir)
    filetypes = filetypes or DATASET_FOLDER_FILETYPES

    try:
        from tkinter import filedialog
        with _tk_root():
            selected = filedialog.askopenfilename(
                initialdir=initial_dir, title=title, filetypes=filetypes,
            )
        return os.path.dirname(selected) if selected else None
    except ImportError:
        pass

    if shutil.which("zenity"):
        result = subprocess.run(
            ["zenity", "--file-selection", f"--title={title}",
             f"--filename={initial_dir}/", *_zenity_filter_args(filetypes)],
            capture_output=True, text=True,
        )
        path = result.stdout.strip() or None if result.returncode == 0 else None
        return os.path.dirname(path) if path else None

    if shutil.which("kdialog"):
        result = subprocess.run(
            ["kdialog", "--getopenfilename", initial_dir, _kdialog_pattern(filetypes)],
            capture_output=True, text=True,
        )
        path = result.stdout.strip() or None if result.returncode == 0 else None
        return os.path.dirname(path) if path else None

    return None
