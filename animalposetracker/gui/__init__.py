# __init__.py
import os
import sys
from PySide6.QtCore import Qt

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
            parent=None, 
            modal=False, 
            delete_on_close=False, 
            *args, 
            **kwargs):
        try:
            window = window_class(parent, *args, **kwargs)
            
            if parent:
                window.setParent(parent)
            
            if modal:
                window.setWindowModality(Qt.ApplicationModal)

            if delete_on_close:
                window.setAttribute(Qt.WA_DeleteOnClose)
            
            window.show()
            return window
        
        except Exception as e:
            print(f"Window creation failed: {str(e)}")
            raise