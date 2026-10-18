"""Render the documentation screenshots without a display.

Run from the repository root with an offscreen Qt platform and a font directory, e.g.::

    QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR="C:/Windows/Fonts" \
        python -m tools.screenshots

The script builds a throwaway three-camera project so the pages show real values, forces the
light theme (the screenshots in the docs are all light), and writes PNGs into
``docs/assets/screenshots``. It never touches the user's saved theme or any real project.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:      # running as a script, not a package
    sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from animalposetracker.gui.mainpage import AnimalPoseTrackerPage  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "screenshots"
WINDOW = (1180, 860)


def _build_project(root: Path) -> Path:
    """Create a real project through the CLI, so every page shows a full configuration."""
    import subprocess

    dataset = root / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({
        "path": "datasets", "train": "images/train", "val": "images/train",
        "kpt_shape": [17, 3], "names": {0: "mouse"},
        "skeleton": [[0, 1], [1, 2], [2, 3], [3, 4]],
    }, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "animalposetracker.cli", "create",
         "--dataset", str(dataset), "--workspace", str(root / "projects"),
         "--name", "mouse-3cam", "--worker", "wux024",
         "--model", "AnimalRTPose", "--scale", "N", "--date", "20261018",
         "--epochs", "300", "--batch-size", "8", "--image-size", "640", "--pose3d"],
        check=True, capture_output=True, cwd=str(_ROOT),
    )
    project = root / "projects" / "mouse-3cam-wux024-AnimalRTPose-N-20261018"
    (project / "configs" / "cameras.yaml").write_text(
        yaml.safe_dump({"world_unit": "mm", "cameras": {}}, sort_keys=False), encoding="utf-8")
    return project


def _capture(page, name: str) -> Path:
    page.repaint()
    QApplication.processEvents()
    target = OUTPUT / f"{name}.png"
    page.grab().save(str(target))
    print(f"wrote {target.relative_to(OUTPUT.parent.parent.parent)}")
    return target


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication(sys.argv)
    page = AnimalPoseTrackerPage()
    page.resize(*WINDOW)
    # The docs are light; the saved preference is deliberately not read or written here.
    page.onChangeTheme("light")
    page.show()
    application.processEvents()

    _capture(page, "welcome")

    with tempfile.TemporaryDirectory() as temporary:
        project = _build_project(Path(temporary))
        page.project.load_project_config(str(project / "project.yaml"))
        page.AnimalPoseTrackerPage.setCurrentIndex(1)
        page.ConfigureFilePathDisplay.setText(
            f"projects/{project.name}/project.yaml"    # the layout the docs use
        )
        page.config_data = page.project.project_config
        page.DisplayConfigureFile()
        page.initialize_other_config()
        page.refreshCameraTable()
        page.CameraTable.setRowCount(0)
        for camera in ("camA", "camB", "camC"):
            page._appendCameraRow(
                camera,
                f"data/calibration/{camera}",
                f"results/predict/{camera}/predictions.json",
            )
        page.Pose3DLog.setPlainText(
            "=== Calibration ===\n"
            "$ python -m animalposetracker.cli calibrate --view camA=data/calibration/camA "
            "--view camB=data/calibration/camB --view camC=data/calibration/camC\n"
        )
        application.processEvents()

        tabs = [
            ("configure", 0),
            ("training", 2),
            ("evaluate", 3),
            ("inference", 4),
            ("export", 5),
            ("pose3d", 6),
        ]
        for name, index in tabs:
            page.ConfigureTabPage.setCurrentIndex(index)
            _capture(page, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
