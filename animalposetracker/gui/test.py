# main.py
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from ui_animalposeinference import Ui_AnimalPoseInference

import sys
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

class AnimalPoseInferenceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AnimalPoseInference()
        self.ui.setupUi(self)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = AnimalPoseInferenceWindow()
    window.show()
    
    sys.exit(app.exec())