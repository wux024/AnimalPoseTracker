from PySide6.QtCore import  Qt, QSettings, QProcess, QFile, QTextStream
from PySide6.QtWidgets import  (QMenu, QApplication, QFileDialog, QMainWindow,
                                 QMessageBox, QTreeWidget, QTreeWidgetItem, 
                                 QTreeWidgetItemIterator)
from PySide6.QtGui import QCursor, QPixmap
import os
import sys
from pathlib import Path
from collections import deque


class AnimalPoseAnnotatorPage(QMainWindow):
    def __init__(self, parent=None):
        pass