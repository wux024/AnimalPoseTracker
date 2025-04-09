# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'publicdatasetsprojectbUJqZf.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout,
    QLabel, QLayout, QLineEdit, QPushButton,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from animalposetracker.gui import LOGO_SMALL_PATH

class Ui_PublicDatasetProject(object):
    def setupUi(self, PublicDatasetProject):
        if not PublicDatasetProject.objectName():
            PublicDatasetProject.setObjectName(u"PublicDatasetProject")
        PublicDatasetProject.resize(759, 203)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PublicDatasetProject.sizePolicy().hasHeightForWidth())
        PublicDatasetProject.setSizePolicy(sizePolicy)
        icon = QIcon()
        icon.addFile(LOGO_SMALL_PATH, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        PublicDatasetProject.setWindowIcon(icon)
        self.PublicDatasetProjectLayout = QVBoxLayout(PublicDatasetProject)
        self.PublicDatasetProjectLayout.setObjectName(u"PublicDatasetProjectLayout")
        self.PublicDatasetProjectLayout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.PublicDatasetProjectGroup = QGroupBox(PublicDatasetProject)
        self.PublicDatasetProjectGroup.setObjectName(u"PublicDatasetProjectGroup")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.PublicDatasetProjectGroup.sizePolicy().hasHeightForWidth())
        self.PublicDatasetProjectGroup.setSizePolicy(sizePolicy1)
        self.PublicDatasetProjectGroup.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.PublicDatasetProjectGroup.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.PublicDatasetProjectGroupHLayout = QHBoxLayout(self.PublicDatasetProjectGroup)
        self.PublicDatasetProjectGroupHLayout.setObjectName(u"PublicDatasetProjectGroupHLayout")
        self.PublicDatasetProjectGroupHLayout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.PublicDatasetProjectGroupHLayout.setContentsMargins(-1, 9, -1, -1)
        self.PublicDatasetProjectGroupLayout = QVBoxLayout()
        self.PublicDatasetProjectGroupLayout.setObjectName(u"PublicDatasetProjectGroupLayout")
        self.PublicDatasetProjectGroupLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.ProjectLayout = QHBoxLayout()
        self.ProjectLayout.setObjectName(u"ProjectLayout")
        self.ProjectLabel = QLabel(self.PublicDatasetProjectGroup)
        self.ProjectLabel.setObjectName(u"ProjectLabel")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.ProjectLabel.sizePolicy().hasHeightForWidth())
        self.ProjectLabel.setSizePolicy(sizePolicy2)
        self.ProjectLabel.setMinimumSize(QSize(60, 25))
        self.ProjectLabel.setMaximumSize(QSize(60, 25))

        self.ProjectLayout.addWidget(self.ProjectLabel)

        self.ProjectConfig = QLineEdit(self.PublicDatasetProjectGroup)
        self.ProjectConfig.setObjectName(u"ProjectConfig")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.ProjectConfig.sizePolicy().hasHeightForWidth())
        self.ProjectConfig.setSizePolicy(sizePolicy3)
        self.ProjectConfig.setMinimumSize(QSize(600, 25))
        self.ProjectConfig.setMaximumSize(QSize(16777215, 25))

        self.ProjectLayout.addWidget(self.ProjectConfig)


        self.PublicDatasetProjectGroupLayout.addLayout(self.ProjectLayout)

        self.WorkerLayout = QHBoxLayout()
        self.WorkerLayout.setObjectName(u"WorkerLayout")
        self.WorkerLabel = QLabel(self.PublicDatasetProjectGroup)
        self.WorkerLabel.setObjectName(u"WorkerLabel")
        sizePolicy2.setHeightForWidth(self.WorkerLabel.sizePolicy().hasHeightForWidth())
        self.WorkerLabel.setSizePolicy(sizePolicy2)
        self.WorkerLabel.setMinimumSize(QSize(60, 25))
        self.WorkerLabel.setMaximumSize(QSize(60, 25))

        self.WorkerLayout.addWidget(self.WorkerLabel)

        self.WorkerConfig = QLineEdit(self.PublicDatasetProjectGroup)
        self.WorkerConfig.setObjectName(u"WorkerConfig")
        sizePolicy3.setHeightForWidth(self.WorkerConfig.sizePolicy().hasHeightForWidth())
        self.WorkerConfig.setSizePolicy(sizePolicy3)
        self.WorkerConfig.setMinimumSize(QSize(600, 25))
        self.WorkerConfig.setMaximumSize(QSize(16777215, 25))

        self.WorkerLayout.addWidget(self.WorkerConfig)


        self.PublicDatasetProjectGroupLayout.addLayout(self.WorkerLayout)

        self.LocationLayout = QHBoxLayout()
        self.LocationLayout.setObjectName(u"LocationLayout")
        self.LocationLabel = QLabel(self.PublicDatasetProjectGroup)
        self.LocationLabel.setObjectName(u"LocationLabel")
        sizePolicy2.setHeightForWidth(self.LocationLabel.sizePolicy().hasHeightForWidth())
        self.LocationLabel.setSizePolicy(sizePolicy2)
        self.LocationLabel.setMinimumSize(QSize(60, 25))
        self.LocationLabel.setMaximumSize(QSize(60, 25))

        self.LocationLayout.addWidget(self.LocationLabel)

        self.LocationPathLayout = QHBoxLayout()
        self.LocationPathLayout.setSpacing(0)
        self.LocationPathLayout.setObjectName(u"LocationPathLayout")
        self.LocationPathDisplay = QLineEdit(self.PublicDatasetProjectGroup)
        self.LocationPathDisplay.setObjectName(u"LocationPathDisplay")
        sizePolicy3.setHeightForWidth(self.LocationPathDisplay.sizePolicy().hasHeightForWidth())
        self.LocationPathDisplay.setSizePolicy(sizePolicy3)
        self.LocationPathDisplay.setMinimumSize(QSize(575, 25))
        self.LocationPathDisplay.setMaximumSize(QSize(16777215, 25))

        self.LocationPathLayout.addWidget(self.LocationPathDisplay)

        self.LocationPathSelection = QToolButton(self.PublicDatasetProjectGroup)
        self.LocationPathSelection.setObjectName(u"LocationPathSelection")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.LocationPathSelection.sizePolicy().hasHeightForWidth())
        self.LocationPathSelection.setSizePolicy(sizePolicy4)
        self.LocationPathSelection.setMinimumSize(QSize(25, 25))
        self.LocationPathSelection.setMaximumSize(QSize(25, 25))
        icon1 = QIcon(QIcon.fromTheme(u"folder-open"))
        self.LocationPathSelection.setIcon(icon1)

        self.LocationPathLayout.addWidget(self.LocationPathSelection)


        self.LocationLayout.addLayout(self.LocationPathLayout)


        self.PublicDatasetProjectGroupLayout.addLayout(self.LocationLayout)

        self.ModelLayout = QHBoxLayout()
        self.ModelLayout.setObjectName(u"ModelLayout")
        self.ModelLabel = QLabel(self.PublicDatasetProjectGroup)
        self.ModelLabel.setObjectName(u"ModelLabel")
        sizePolicy2.setHeightForWidth(self.ModelLabel.sizePolicy().hasHeightForWidth())
        self.ModelLabel.setSizePolicy(sizePolicy2)
        self.ModelLabel.setMinimumSize(QSize(60, 25))
        self.ModelLabel.setMaximumSize(QSize(60, 25))

        self.ModelLayout.addWidget(self.ModelLabel)

        self.ModelSelectionLayout = QHBoxLayout()
        self.ModelSelectionLayout.setSpacing(0)
        self.ModelSelectionLayout.setObjectName(u"ModelSelectionLayout")
        self.ModelTypeSelection = QComboBox(self.PublicDatasetProjectGroup)
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.setObjectName(u"ModelTypeSelection")
        sizePolicy3.setHeightForWidth(self.ModelTypeSelection.sizePolicy().hasHeightForWidth())
        self.ModelTypeSelection.setSizePolicy(sizePolicy3)
        self.ModelTypeSelection.setMinimumSize(QSize(300, 25))
        self.ModelTypeSelection.setMaximumSize(QSize(16777215, 25))

        self.ModelSelectionLayout.addWidget(self.ModelTypeSelection)

        self.ModelScaleSelection = QComboBox(self.PublicDatasetProjectGroup)
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.setObjectName(u"ModelScaleSelection")
        self.ModelScaleSelection.setEnabled(True)
        sizePolicy3.setHeightForWidth(self.ModelScaleSelection.sizePolicy().hasHeightForWidth())
        self.ModelScaleSelection.setSizePolicy(sizePolicy3)
        self.ModelScaleSelection.setMinimumSize(QSize(300, 25))
        self.ModelScaleSelection.setMaximumSize(QSize(16777215, 25))

        self.ModelSelectionLayout.addWidget(self.ModelScaleSelection)


        self.ModelLayout.addLayout(self.ModelSelectionLayout)


        self.PublicDatasetProjectGroupLayout.addLayout(self.ModelLayout)

        self.DatasetLayout = QHBoxLayout()
        self.DatasetLayout.setObjectName(u"DatasetLayout")
        self.OthersLabel = QLabel(self.PublicDatasetProjectGroup)
        self.OthersLabel.setObjectName(u"OthersLabel")
        sizePolicy2.setHeightForWidth(self.OthersLabel.sizePolicy().hasHeightForWidth())
        self.OthersLabel.setSizePolicy(sizePolicy2)
        self.OthersLabel.setMinimumSize(QSize(60, 25))
        self.OthersLabel.setMaximumSize(QSize(60, 25))

        self.DatasetLayout.addWidget(self.OthersLabel)

        self.DatasetSelection = QComboBox(self.PublicDatasetProjectGroup)
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.addItem("")
        self.DatasetSelection.setObjectName(u"DatasetSelection")
        sizePolicy3.setHeightForWidth(self.DatasetSelection.sizePolicy().hasHeightForWidth())
        self.DatasetSelection.setSizePolicy(sizePolicy3)
        self.DatasetSelection.setMinimumSize(QSize(300, 25))
        self.DatasetSelection.setMaximumSize(QSize(16777215, 25))

        self.DatasetLayout.addWidget(self.DatasetSelection)

        self.CreateProjectBase = QPushButton(self.PublicDatasetProjectGroup)
        self.CreateProjectBase.setObjectName(u"CreateProjectBase")
        sizePolicy1.setHeightForWidth(self.CreateProjectBase.sizePolicy().hasHeightForWidth())
        self.CreateProjectBase.setSizePolicy(sizePolicy1)
        self.CreateProjectBase.setMinimumSize(QSize(345, 25))
        self.CreateProjectBase.setMaximumSize(QSize(16777215, 25))

        self.DatasetLayout.addWidget(self.CreateProjectBase)


        self.PublicDatasetProjectGroupLayout.addLayout(self.DatasetLayout)


        self.PublicDatasetProjectGroupHLayout.addLayout(self.PublicDatasetProjectGroupLayout)


        self.PublicDatasetProjectLayout.addWidget(self.PublicDatasetProjectGroup)


        self.retranslateUi(PublicDatasetProject)

        QMetaObject.connectSlotsByName(PublicDatasetProject)
    # setupUi

    def retranslateUi(self, PublicDatasetProject):
        PublicDatasetProject.setWindowTitle(QCoreApplication.translate("PublicDatasetProject", u"New Public Dataset Project", None))
        self.PublicDatasetProjectGroup.setTitle("")
        self.ProjectLabel.setText(QCoreApplication.translate("PublicDatasetProject", u"Project", None))
        self.WorkerLabel.setText(QCoreApplication.translate("PublicDatasetProject", u"Worker", None))
        self.LocationLabel.setText(QCoreApplication.translate("PublicDatasetProject", u"Location", None))
        self.LocationPathSelection.setText("")
        self.ModelLabel.setText(QCoreApplication.translate("PublicDatasetProject", u"Model", None))
        self.ModelTypeSelection.setItemText(0, QCoreApplication.translate("PublicDatasetProject", u"AnimalRTPose", None))
        self.ModelTypeSelection.setItemText(1, QCoreApplication.translate("PublicDatasetProject", u"AnimalViTPose", None))
        self.ModelTypeSelection.setItemText(2, QCoreApplication.translate("PublicDatasetProject", u"SPIPose", None))
        self.ModelTypeSelection.setItemText(3, QCoreApplication.translate("PublicDatasetProject", u"YOLO12-Pose", None))
        self.ModelTypeSelection.setItemText(4, QCoreApplication.translate("PublicDatasetProject", u"YOLO11-Pose", None))
        self.ModelTypeSelection.setItemText(5, QCoreApplication.translate("PublicDatasetProject", u"YOLOv8-Pose", None))
        self.ModelTypeSelection.setItemText(6, QCoreApplication.translate("PublicDatasetProject", u"YOLOv8-Pose-P6", None))

        self.ModelScaleSelection.setItemText(0, QCoreApplication.translate("PublicDatasetProject", u"N", None))
        self.ModelScaleSelection.setItemText(1, QCoreApplication.translate("PublicDatasetProject", u"S", None))
        self.ModelScaleSelection.setItemText(2, QCoreApplication.translate("PublicDatasetProject", u"M", None))
        self.ModelScaleSelection.setItemText(3, QCoreApplication.translate("PublicDatasetProject", u"L", None))
        self.ModelScaleSelection.setItemText(4, QCoreApplication.translate("PublicDatasetProject", u"H", None))

        self.OthersLabel.setText(QCoreApplication.translate("PublicDatasetProject", u"Dataset", None))
        self.DatasetSelection.setItemText(0, QCoreApplication.translate("PublicDatasetProject", u"AcinoSet", None))
        self.DatasetSelection.setItemText(1, QCoreApplication.translate("PublicDatasetProject", u"AnimalKingdom", None))
        self.DatasetSelection.setItemText(2, QCoreApplication.translate("PublicDatasetProject", u"AnimalPose", None))
        self.DatasetSelection.setItemText(3, QCoreApplication.translate("PublicDatasetProject", u"AniposeFly", None))
        self.DatasetSelection.setItemText(4, QCoreApplication.translate("PublicDatasetProject", u"AnimalMouse", None))
        self.DatasetSelection.setItemText(5, QCoreApplication.translate("PublicDatasetProject", u"AP10K", None))
        self.DatasetSelection.setItemText(6, QCoreApplication.translate("PublicDatasetProject", u"APT36K", None))
        self.DatasetSelection.setItemText(7, QCoreApplication.translate("PublicDatasetProject", u"APTv2", None))
        self.DatasetSelection.setItemText(8, QCoreApplication.translate("PublicDatasetProject", u"ATRW", None))
        self.DatasetSelection.setItemText(9, QCoreApplication.translate("PublicDatasetProject", u"AwA", None))
        self.DatasetSelection.setItemText(10, QCoreApplication.translate("PublicDatasetProject", u"COCO", None))
        self.DatasetSelection.setItemText(11, QCoreApplication.translate("PublicDatasetProject", u"Fish", None))
        self.DatasetSelection.setItemText(12, QCoreApplication.translate("PublicDatasetProject", u"Fly", None))
        self.DatasetSelection.setItemText(13, QCoreApplication.translate("PublicDatasetProject", u"Horse10", None))
        self.DatasetSelection.setItemText(14, QCoreApplication.translate("PublicDatasetProject", u"Locust", None))
        self.DatasetSelection.setItemText(15, QCoreApplication.translate("PublicDatasetProject", u"LoTE", None))
        self.DatasetSelection.setItemText(16, QCoreApplication.translate("PublicDatasetProject", u"LSP", None))
        self.DatasetSelection.setItemText(17, QCoreApplication.translate("PublicDatasetProject", u"Macaque", None))
        self.DatasetSelection.setItemText(18, QCoreApplication.translate("PublicDatasetProject", u"Marmoset", None))
        self.DatasetSelection.setItemText(19, QCoreApplication.translate("PublicDatasetProject", u"OpenMonkeyChallenge", None))
        self.DatasetSelection.setItemText(20, QCoreApplication.translate("PublicDatasetProject", u"Pups", None))
        self.DatasetSelection.setItemText(21, QCoreApplication.translate("PublicDatasetProject", u"Quadruped-80K", None))
        self.DatasetSelection.setItemText(22, QCoreApplication.translate("PublicDatasetProject", u"StanfordExtra", None))
        self.DatasetSelection.setItemText(23, QCoreApplication.translate("PublicDatasetProject", u"TopViewMouse", None))
        self.DatasetSelection.setItemText(24, QCoreApplication.translate("PublicDatasetProject", u"Tri-Mouse", None))
        self.DatasetSelection.setItemText(25, QCoreApplication.translate("PublicDatasetProject", u"Zebra", None))

        self.CreateProjectBase.setText(QCoreApplication.translate("PublicDatasetProject", u"Create Public Dataset Project", None))
    # retranslateUi

