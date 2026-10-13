# __init__.py
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
import os

BASE_DIR = os.path.dirname(__file__)
LOGO_PATH_TRANSPARENT = os.path.join(BASE_DIR, "assets", "logo_transparent.png")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
LOGO_SMALL_PATH = os.path.join(BASE_DIR, "media", "logo.png")
WELCOME_PATH = os.path.join(BASE_DIR, "assets", "welcome.png")

DARK_THEME_PATH = os.path.join(BASE_DIR, "style", "dark.qss")
LIGHT_THEME_PATH = os.path.join(BASE_DIR, "style", "light.qss")


class WindowFactory:
    @staticmethod
    def run(window_class,
            *args,
            parent=None,
            modal=False,
            delete_on_close=False,
            stylesheet=None,
            **kwargs):
        """Build and show one tool window.

        ``parent``, ``modal``, ``delete_on_close`` and ``stylesheet`` are keyword-only: they used
        to sit after ``*args``-style parameters, so a second positional argument silently bound to
        ``parent``. ``stylesheet`` is applied to the new window because a tool window is
        top-level and does not inherit the main window's theme.
        """
        window = window_class(parent, *args, **kwargs)

        if parent:
            window.setParent(parent)

        if modal:
            window.setWindowModality(Qt.ApplicationModal)

        if delete_on_close:
            window.setAttribute(Qt.WA_DeleteOnClose)

        if stylesheet:
            window.setStyleSheet(stylesheet)

        window.show()
        return window


COLORS = [
        Qt.red, Qt.green, Qt.blue, Qt.yellow, Qt.magenta, Qt.cyan,
        QColor(255, 128, 0), 
        QColor(128, 255, 0), 
        QColor(0, 255, 128), 
        QColor(0, 128, 255),
        QColor(128, 0, 255),
        QColor(255, 0, 128),
        Qt.darkRed, Qt.darkGreen, Qt.darkBlue, Qt.darkYellow,
        QColor(128, 64, 0), 
        QColor(64, 128, 0),
        QColor(0, 128, 64),
        QColor(0, 64, 128),
    ]