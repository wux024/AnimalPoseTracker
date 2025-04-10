# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'createnewprojectdRPRkw.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLayout,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QToolButton,
    QVBoxLayout, QWidget)

from animalposetracker.gui import LOGO_SMALL_PATH

class Ui_CreateNewProject(object):
    def setupUi(self, CreateNewProject):
        if not CreateNewProject.objectName():
            CreateNewProject.setObjectName(u"CreateNewProject")
        CreateNewProject.resize(968, 575)
        icon = QIcon()
        icon.addFile(LOGO_SMALL_PATH, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        CreateNewProject.setWindowIcon(icon)
        self.CreateNewProjectLayout = QVBoxLayout(CreateNewProject)
        self.CreateNewProjectLayout.setObjectName(u"CreateNewProjectLayout")
        self.BaseConfigGroup = QGroupBox(CreateNewProject)
        self.BaseConfigGroup.setObjectName(u"BaseConfigGroup")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BaseConfigGroup.sizePolicy().hasHeightForWidth())
        self.BaseConfigGroup.setSizePolicy(sizePolicy)
        self.BaseConfigGroup.setMinimumSize(QSize(900, 0))
        self.BaseConfigGroupHLayout = QHBoxLayout(self.BaseConfigGroup)
        self.BaseConfigGroupHLayout.setObjectName(u"BaseConfigGroupHLayout")
        self.BaseConfigGroupLayout = QVBoxLayout()
        self.BaseConfigGroupLayout.setObjectName(u"BaseConfigGroupLayout")
        self.BaseConfigGroupGridLayout = QGridLayout()
        self.BaseConfigGroupGridLayout.setSpacing(0)
        self.BaseConfigGroupGridLayout.setObjectName(u"BaseConfigGroupGridLayout")
        self.PoseConfigLayout = QHBoxLayout()
        self.PoseConfigLayout.setSpacing(105)
        self.PoseConfigLayout.setObjectName(u"PoseConfigLayout")
        self.KeypointConfigLayout = QHBoxLayout()
        self.KeypointConfigLayout.setSpacing(10)
        self.KeypointConfigLayout.setObjectName(u"KeypointConfigLayout")
        self.KeypointLabel = QLabel(self.BaseConfigGroup)
        self.KeypointLabel.setObjectName(u"KeypointLabel")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.KeypointLabel.sizePolicy().hasHeightForWidth())
        self.KeypointLabel.setSizePolicy(sizePolicy1)
        self.KeypointLabel.setMinimumSize(QSize(60, 25))
        self.KeypointLabel.setMaximumSize(QSize(60, 25))

        self.KeypointConfigLayout.addWidget(self.KeypointLabel)

        self.KeypointConfig = QSpinBox(self.BaseConfigGroup)
        self.KeypointConfig.setObjectName(u"KeypointConfig")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.KeypointConfig.sizePolicy().hasHeightForWidth())
        self.KeypointConfig.setSizePolicy(sizePolicy2)
        self.KeypointConfig.setMinimumSize(QSize(60, 25))
        self.KeypointConfig.setMaximumSize(QSize(16777215, 25))

        self.KeypointConfigLayout.addWidget(self.KeypointConfig)


        self.PoseConfigLayout.addLayout(self.KeypointConfigLayout)

        self.ClassConfigLayout = QHBoxLayout()
        self.ClassConfigLayout.setSpacing(10)
        self.ClassConfigLayout.setObjectName(u"ClassConfigLayout")
        self.ClassLabel = QLabel(self.BaseConfigGroup)
        self.ClassLabel.setObjectName(u"ClassLabel")
        sizePolicy1.setHeightForWidth(self.ClassLabel.sizePolicy().hasHeightForWidth())
        self.ClassLabel.setSizePolicy(sizePolicy1)
        self.ClassLabel.setMinimumSize(QSize(60, 25))
        self.ClassLabel.setMaximumSize(QSize(60, 25))

        self.ClassConfigLayout.addWidget(self.ClassLabel)

        self.ClassConfig = QSpinBox(self.BaseConfigGroup)
        self.ClassConfig.setObjectName(u"ClassConfig")
        sizePolicy2.setHeightForWidth(self.ClassConfig.sizePolicy().hasHeightForWidth())
        self.ClassConfig.setSizePolicy(sizePolicy2)
        self.ClassConfig.setMinimumSize(QSize(60, 25))
        self.ClassConfig.setMaximumSize(QSize(16777215, 25))

        self.ClassConfigLayout.addWidget(self.ClassConfig)


        self.PoseConfigLayout.addLayout(self.ClassConfigLayout)

        self.VisibleConfigLayout = QHBoxLayout()
        self.VisibleConfigLayout.setSpacing(10)
        self.VisibleConfigLayout.setObjectName(u"VisibleConfigLayout")
        self.VisibleLabel = QLabel(self.BaseConfigGroup)
        self.VisibleLabel.setObjectName(u"VisibleLabel")
        sizePolicy1.setHeightForWidth(self.VisibleLabel.sizePolicy().hasHeightForWidth())
        self.VisibleLabel.setSizePolicy(sizePolicy1)
        self.VisibleLabel.setMinimumSize(QSize(60, 25))
        self.VisibleLabel.setMaximumSize(QSize(60, 25))

        self.VisibleConfigLayout.addWidget(self.VisibleLabel)

        self.VisibleSelection = QCheckBox(self.BaseConfigGroup)
        self.VisibleSelection.setObjectName(u"VisibleSelection")
        sizePolicy1.setHeightForWidth(self.VisibleSelection.sizePolicy().hasHeightForWidth())
        self.VisibleSelection.setSizePolicy(sizePolicy1)
        self.VisibleSelection.setMinimumSize(QSize(16, 16))
        self.VisibleSelection.setMaximumSize(QSize(16, 16))

        self.VisibleConfigLayout.addWidget(self.VisibleSelection)


        self.PoseConfigLayout.addLayout(self.VisibleConfigLayout)


        self.BaseConfigGroupGridLayout.addLayout(self.PoseConfigLayout, 5, 1, 1, 1)

        self.LocationPathLayout = QHBoxLayout()
        self.LocationPathLayout.setSpacing(0)
        self.LocationPathLayout.setObjectName(u"LocationPathLayout")
        self.LocationPathDisplay = QLineEdit(self.BaseConfigGroup)
        self.LocationPathDisplay.setObjectName(u"LocationPathDisplay")
        sizePolicy2.setHeightForWidth(self.LocationPathDisplay.sizePolicy().hasHeightForWidth())
        self.LocationPathDisplay.setSizePolicy(sizePolicy2)
        self.LocationPathDisplay.setMinimumSize(QSize(575, 25))
        self.LocationPathDisplay.setMaximumSize(QSize(16777215, 25))

        self.LocationPathLayout.addWidget(self.LocationPathDisplay)

        self.LocationPathSelection = QToolButton(self.BaseConfigGroup)
        self.LocationPathSelection.setObjectName(u"LocationPathSelection")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.LocationPathSelection.sizePolicy().hasHeightForWidth())
        self.LocationPathSelection.setSizePolicy(sizePolicy3)
        self.LocationPathSelection.setMinimumSize(QSize(25, 25))
        self.LocationPathSelection.setMaximumSize(QSize(25, 25))
        icon1 = QIcon(QIcon.fromTheme(u"folder-open"))
        self.LocationPathSelection.setIcon(icon1)

        self.LocationPathLayout.addWidget(self.LocationPathSelection)


        self.BaseConfigGroupGridLayout.addLayout(self.LocationPathLayout, 3, 1, 1, 1)

        self.LocationLabel = QLabel(self.BaseConfigGroup)
        self.LocationLabel.setObjectName(u"LocationLabel")
        sizePolicy1.setHeightForWidth(self.LocationLabel.sizePolicy().hasHeightForWidth())
        self.LocationLabel.setSizePolicy(sizePolicy1)
        self.LocationLabel.setMinimumSize(QSize(60, 25))
        self.LocationLabel.setMaximumSize(QSize(60, 25))

        self.BaseConfigGroupGridLayout.addWidget(self.LocationLabel, 3, 0, 1, 1)

        self.OthersLabel = QLabel(self.BaseConfigGroup)
        self.OthersLabel.setObjectName(u"OthersLabel")
        sizePolicy1.setHeightForWidth(self.OthersLabel.sizePolicy().hasHeightForWidth())
        self.OthersLabel.setSizePolicy(sizePolicy1)
        self.OthersLabel.setMinimumSize(QSize(60, 25))
        self.OthersLabel.setMaximumSize(QSize(60, 25))

        self.BaseConfigGroupGridLayout.addWidget(self.OthersLabel, 5, 0, 1, 1)

        self.ProjectLabel = QLabel(self.BaseConfigGroup)
        self.ProjectLabel.setObjectName(u"ProjectLabel")
        sizePolicy1.setHeightForWidth(self.ProjectLabel.sizePolicy().hasHeightForWidth())
        self.ProjectLabel.setSizePolicy(sizePolicy1)
        self.ProjectLabel.setMinimumSize(QSize(60, 25))
        self.ProjectLabel.setMaximumSize(QSize(60, 25))

        self.BaseConfigGroupGridLayout.addWidget(self.ProjectLabel, 0, 0, 1, 1)

        self.WorkerConfig = QLineEdit(self.BaseConfigGroup)
        self.WorkerConfig.setObjectName(u"WorkerConfig")
        sizePolicy2.setHeightForWidth(self.WorkerConfig.sizePolicy().hasHeightForWidth())
        self.WorkerConfig.setSizePolicy(sizePolicy2)
        self.WorkerConfig.setMinimumSize(QSize(600, 25))
        self.WorkerConfig.setMaximumSize(QSize(16777215, 25))

        self.BaseConfigGroupGridLayout.addWidget(self.WorkerConfig, 2, 1, 1, 1)

        self.WorkerLabel = QLabel(self.BaseConfigGroup)
        self.WorkerLabel.setObjectName(u"WorkerLabel")
        sizePolicy1.setHeightForWidth(self.WorkerLabel.sizePolicy().hasHeightForWidth())
        self.WorkerLabel.setSizePolicy(sizePolicy1)
        self.WorkerLabel.setMinimumSize(QSize(60, 25))
        self.WorkerLabel.setMaximumSize(QSize(60, 25))

        self.BaseConfigGroupGridLayout.addWidget(self.WorkerLabel, 2, 0, 1, 1)

        self.ModelLabel = QLabel(self.BaseConfigGroup)
        self.ModelLabel.setObjectName(u"ModelLabel")
        sizePolicy1.setHeightForWidth(self.ModelLabel.sizePolicy().hasHeightForWidth())
        self.ModelLabel.setSizePolicy(sizePolicy1)
        self.ModelLabel.setMinimumSize(QSize(60, 25))
        self.ModelLabel.setMaximumSize(QSize(60, 25))

        self.BaseConfigGroupGridLayout.addWidget(self.ModelLabel, 4, 0, 1, 1)

        self.ModelSelectionLayout = QHBoxLayout()
        self.ModelSelectionLayout.setSpacing(0)
        self.ModelSelectionLayout.setObjectName(u"ModelSelectionLayout")
        self.ModelTypeSelection = QComboBox(self.BaseConfigGroup)
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.addItem("")
        self.ModelTypeSelection.setObjectName(u"ModelTypeSelection")
        sizePolicy2.setHeightForWidth(self.ModelTypeSelection.sizePolicy().hasHeightForWidth())
        self.ModelTypeSelection.setSizePolicy(sizePolicy2)
        self.ModelTypeSelection.setMinimumSize(QSize(300, 25))
        self.ModelTypeSelection.setMaximumSize(QSize(16777215, 25))

        self.ModelSelectionLayout.addWidget(self.ModelTypeSelection)

        self.ModelScaleSelection = QComboBox(self.BaseConfigGroup)
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.addItem("")
        self.ModelScaleSelection.setObjectName(u"ModelScaleSelection")
        self.ModelScaleSelection.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.ModelScaleSelection.sizePolicy().hasHeightForWidth())
        self.ModelScaleSelection.setSizePolicy(sizePolicy2)
        self.ModelScaleSelection.setMinimumSize(QSize(300, 25))
        self.ModelScaleSelection.setMaximumSize(QSize(16777215, 25))

        self.ModelSelectionLayout.addWidget(self.ModelScaleSelection)


        self.BaseConfigGroupGridLayout.addLayout(self.ModelSelectionLayout, 4, 1, 1, 1)

        self.ProjectConfig = QLineEdit(self.BaseConfigGroup)
        self.ProjectConfig.setObjectName(u"ProjectConfig")
        sizePolicy2.setHeightForWidth(self.ProjectConfig.sizePolicy().hasHeightForWidth())
        self.ProjectConfig.setSizePolicy(sizePolicy2)
        self.ProjectConfig.setMinimumSize(QSize(600, 25))
        self.ProjectConfig.setMaximumSize(QSize(16777215, 25))

        self.BaseConfigGroupGridLayout.addWidget(self.ProjectConfig, 0, 1, 1, 1)


        self.BaseConfigGroupLayout.addLayout(self.BaseConfigGroupGridLayout)

        self.KeypointsClassesNameLabelLayout = QHBoxLayout()
        self.KeypointsClassesNameLabelLayout.setSpacing(0)
        self.KeypointsClassesNameLabelLayout.setObjectName(u"KeypointsClassesNameLabelLayout")
        self.KeypointNameLabel = QLabel(self.BaseConfigGroup)
        self.KeypointNameLabel.setObjectName(u"KeypointNameLabel")
        sizePolicy2.setHeightForWidth(self.KeypointNameLabel.sizePolicy().hasHeightForWidth())
        self.KeypointNameLabel.setSizePolicy(sizePolicy2)
        self.KeypointNameLabel.setMinimumSize(QSize(0, 25))
        self.KeypointNameLabel.setMaximumSize(QSize(16777215, 25))

        self.KeypointsClassesNameLabelLayout.addWidget(self.KeypointNameLabel)

        self.ClassesNameLabel = QLabel(self.BaseConfigGroup)
        self.ClassesNameLabel.setObjectName(u"ClassesNameLabel")
        sizePolicy2.setHeightForWidth(self.ClassesNameLabel.sizePolicy().hasHeightForWidth())
        self.ClassesNameLabel.setSizePolicy(sizePolicy2)
        self.ClassesNameLabel.setMinimumSize(QSize(0, 25))
        self.ClassesNameLabel.setMaximumSize(QSize(16777215, 25))

        self.KeypointsClassesNameLabelLayout.addWidget(self.ClassesNameLabel)


        self.BaseConfigGroupLayout.addLayout(self.KeypointsClassesNameLabelLayout)

        self.KeypointClassListGroupLayout = QHBoxLayout()
        self.KeypointClassListGroupLayout.setSpacing(0)
        self.KeypointClassListGroupLayout.setObjectName(u"KeypointClassListGroupLayout")
        self.KeypointClassListGroupLayout.setContentsMargins(-1, 0, -1, -1)
        self.KeypointList = QScrollArea(self.BaseConfigGroup)
        self.KeypointList.setObjectName(u"KeypointList")
        self.KeypointList.setMinimumSize(QSize(450, 0))
        self.KeypointList.setWidgetResizable(True)
        self.KeypointList.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.KeypointListLayout = QWidget()
        self.KeypointListLayout.setObjectName(u"KeypointListLayout")
        self.KeypointListLayout.setGeometry(QRect(0, 0, 461, 56))
        sizePolicy.setHeightForWidth(self.KeypointListLayout.sizePolicy().hasHeightForWidth())
        self.KeypointListLayout.setSizePolicy(sizePolicy)
        self.KeypointListLayout.setMinimumSize(QSize(450, 0))
        self.KeypointListVLayout = QVBoxLayout(self.KeypointListLayout)
        self.KeypointListVLayout.setSpacing(10)
        self.KeypointListVLayout.setObjectName(u"KeypointListVLayout")
        self.KeypointListVLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.KeypointListVLayout.setContentsMargins(0, 10, 0, 10)
        self.KeypointLayout = QHBoxLayout()
        self.KeypointLayout.setSpacing(10)
        self.KeypointLayout.setObjectName(u"KeypointLayout")
        self.KeypointLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.KeypointLayout.setContentsMargins(10, 10, -1, -1)
        self.KeypointNumber1 = QLabel(self.KeypointListLayout)
        self.KeypointNumber1.setObjectName(u"KeypointNumber1")
        sizePolicy3.setHeightForWidth(self.KeypointNumber1.sizePolicy().hasHeightForWidth())
        self.KeypointNumber1.setSizePolicy(sizePolicy3)
        self.KeypointNumber1.setMinimumSize(QSize(10, 25))
        self.KeypointNumber1.setMaximumSize(QSize(16777215, 25))
        self.KeypointNumber1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.KeypointLayout.addWidget(self.KeypointNumber1)

        self.Keypoint1 = QLineEdit(self.KeypointListLayout)
        self.Keypoint1.setObjectName(u"Keypoint1")
        sizePolicy2.setHeightForWidth(self.Keypoint1.sizePolicy().hasHeightForWidth())
        self.Keypoint1.setSizePolicy(sizePolicy2)
        self.Keypoint1.setMinimumSize(QSize(400, 25))
        self.Keypoint1.setMaximumSize(QSize(16777215, 25))

        self.KeypointLayout.addWidget(self.Keypoint1)


        self.KeypointListVLayout.addLayout(self.KeypointLayout)

        self.KeypointList.setWidget(self.KeypointListLayout)

        self.KeypointClassListGroupLayout.addWidget(self.KeypointList)

        self.ClassList = QScrollArea(self.BaseConfigGroup)
        self.ClassList.setObjectName(u"ClassList")
        self.ClassList.setMinimumSize(QSize(450, 0))
        self.ClassList.setWidgetResizable(True)
        self.ClassList.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.ClassListLayout = QWidget()
        self.ClassListLayout.setObjectName(u"ClassListLayout")
        self.ClassListLayout.setGeometry(QRect(0, 0, 461, 56))
        sizePolicy.setHeightForWidth(self.ClassListLayout.sizePolicy().hasHeightForWidth())
        self.ClassListLayout.setSizePolicy(sizePolicy)
        self.ClassListLayout.setMinimumSize(QSize(450, 0))
        self.ClassListVLayout = QVBoxLayout(self.ClassListLayout)
        self.ClassListVLayout.setSpacing(10)
        self.ClassListVLayout.setObjectName(u"ClassListVLayout")
        self.ClassListVLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.ClassListVLayout.setContentsMargins(0, 10, 0, 10)
        self.ClassLayout = QHBoxLayout()
        self.ClassLayout.setSpacing(10)
        self.ClassLayout.setObjectName(u"ClassLayout")
        self.ClassLayout.setContentsMargins(10, 10, -1, -1)
        self.ClassNumber1 = QLabel(self.ClassListLayout)
        self.ClassNumber1.setObjectName(u"ClassNumber1")
        sizePolicy3.setHeightForWidth(self.ClassNumber1.sizePolicy().hasHeightForWidth())
        self.ClassNumber1.setSizePolicy(sizePolicy3)
        self.ClassNumber1.setMinimumSize(QSize(10, 25))
        self.ClassNumber1.setMaximumSize(QSize(16777215, 25))
        self.ClassNumber1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ClassLayout.addWidget(self.ClassNumber1)

        self.Class1 = QLineEdit(self.ClassListLayout)
        self.Class1.setObjectName(u"Class1")
        sizePolicy2.setHeightForWidth(self.Class1.sizePolicy().hasHeightForWidth())
        self.Class1.setSizePolicy(sizePolicy2)
        self.Class1.setMinimumSize(QSize(400, 25))
        self.Class1.setMaximumSize(QSize(16777215, 25))

        self.ClassLayout.addWidget(self.Class1)


        self.ClassListVLayout.addLayout(self.ClassLayout)

        self.ClassList.setWidget(self.ClassListLayout)

        self.KeypointClassListGroupLayout.addWidget(self.ClassList)


        self.BaseConfigGroupLayout.addLayout(self.KeypointClassListGroupLayout)


        self.BaseConfigGroupHLayout.addLayout(self.BaseConfigGroupLayout)


        self.CreateNewProjectLayout.addWidget(self.BaseConfigGroup)

        self.SourceDataSelected = QCheckBox(CreateNewProject)
        self.SourceDataSelected.setObjectName(u"SourceDataSelected")

        self.CreateNewProjectLayout.addWidget(self.SourceDataSelected)

        self.SourceDataListDisplayGroup = QGroupBox(CreateNewProject)
        self.SourceDataListDisplayGroup.setObjectName(u"SourceDataListDisplayGroup")
        sizePolicy.setHeightForWidth(self.SourceDataListDisplayGroup.sizePolicy().hasHeightForWidth())
        self.SourceDataListDisplayGroup.setSizePolicy(sizePolicy)
        self.SourceDataListDisplayGroup.setMinimumSize(QSize(950, 0))
        self.SourceDataListDisplayGroupLayout = QHBoxLayout(self.SourceDataListDisplayGroup)
        self.SourceDataListDisplayGroupLayout.setObjectName(u"SourceDataListDisplayGroupLayout")
        self.SourceList = QListWidget(self.SourceDataListDisplayGroup)
        self.SourceList.setObjectName(u"SourceList")

        self.SourceDataListDisplayGroupLayout.addWidget(self.SourceList)


        self.CreateNewProjectLayout.addWidget(self.SourceDataListDisplayGroup)

        self.OtherConfigGridLayout = QGridLayout()
        self.OtherConfigGridLayout.setSpacing(0)
        self.OtherConfigGridLayout.setObjectName(u"OtherConfigGridLayout")
        self.CreateProjectBase = QPushButton(CreateNewProject)
        self.CreateProjectBase.setObjectName(u"CreateProjectBase")
        self.CreateProjectBase.setMinimumSize(QSize(345, 25))
        self.CreateProjectBase.setMaximumSize(QSize(16777215, 25))

        self.OtherConfigGridLayout.addWidget(self.CreateProjectBase, 1, 1, 1, 1)

        self.BrowseSourceData = QPushButton(CreateNewProject)
        self.BrowseSourceData.setObjectName(u"BrowseSourceData")
        sizePolicy2.setHeightForWidth(self.BrowseSourceData.sizePolicy().hasHeightForWidth())
        self.BrowseSourceData.setSizePolicy(sizePolicy2)
        self.BrowseSourceData.setMinimumSize(QSize(345, 25))
        self.BrowseSourceData.setMaximumSize(QSize(16777215, 25))

        self.OtherConfigGridLayout.addWidget(self.BrowseSourceData, 0, 0, 1, 1)

        self.CopySourceData = QCheckBox(CreateNewProject)
        self.CopySourceData.setObjectName(u"CopySourceData")
        sizePolicy2.setHeightForWidth(self.CopySourceData.sizePolicy().hasHeightForWidth())
        self.CopySourceData.setSizePolicy(sizePolicy2)
        self.CopySourceData.setMinimumSize(QSize(0, 25))
        self.CopySourceData.setMaximumSize(QSize(16777215, 25))

        self.OtherConfigGridLayout.addWidget(self.CopySourceData, 1, 0, 1, 1)

        self.ClearSourceDataList = QPushButton(CreateNewProject)
        self.ClearSourceDataList.setObjectName(u"ClearSourceDataList")
        sizePolicy2.setHeightForWidth(self.ClearSourceDataList.sizePolicy().hasHeightForWidth())
        self.ClearSourceDataList.setSizePolicy(sizePolicy2)
        self.ClearSourceDataList.setMinimumSize(QSize(345, 25))
        self.ClearSourceDataList.setMaximumSize(QSize(16777215, 25))

        self.OtherConfigGridLayout.addWidget(self.ClearSourceDataList, 0, 1, 1, 1)


        self.CreateNewProjectLayout.addLayout(self.OtherConfigGridLayout)


        self.retranslateUi(CreateNewProject)

        QMetaObject.connectSlotsByName(CreateNewProject)
    # setupUi

    def retranslateUi(self, CreateNewProject):
        CreateNewProject.setWindowTitle(QCoreApplication.translate("CreateNewProject", u"New Project", None))
        self.BaseConfigGroup.setTitle("")
        self.KeypointLabel.setText(QCoreApplication.translate("CreateNewProject", u"Keypoints", None))
        self.ClassLabel.setText(QCoreApplication.translate("CreateNewProject", u"Classes", None))
        self.VisibleLabel.setText(QCoreApplication.translate("CreateNewProject", u"Visible", None))
        self.VisibleSelection.setText("")
        self.LocationPathSelection.setText("")
        self.LocationLabel.setText(QCoreApplication.translate("CreateNewProject", u"Location", None))
        self.OthersLabel.setText(QCoreApplication.translate("CreateNewProject", u"Others", None))
        self.ProjectLabel.setText(QCoreApplication.translate("CreateNewProject", u"Project", None))
        self.WorkerLabel.setText(QCoreApplication.translate("CreateNewProject", u"Worker", None))
        self.ModelLabel.setText(QCoreApplication.translate("CreateNewProject", u"Model", None))
        self.ModelTypeSelection.setItemText(0, QCoreApplication.translate("CreateNewProject", u"AnimalRTPose", None))
        self.ModelTypeSelection.setItemText(1, QCoreApplication.translate("CreateNewProject", u"AnimalViTPose", None))
        self.ModelTypeSelection.setItemText(2, QCoreApplication.translate("CreateNewProject", u"SPIPose", None))
        self.ModelTypeSelection.setItemText(3, QCoreApplication.translate("CreateNewProject", u"YOLO11-Pose", None))
        self.ModelTypeSelection.setItemText(4, QCoreApplication.translate("CreateNewProject", u"YOLOv8-Pose", None))

        self.ModelScaleSelection.setItemText(0, QCoreApplication.translate("CreateNewProject", u"N", None))
        self.ModelScaleSelection.setItemText(1, QCoreApplication.translate("CreateNewProject", u"S", None))
        self.ModelScaleSelection.setItemText(2, QCoreApplication.translate("CreateNewProject", u"M", None))
        self.ModelScaleSelection.setItemText(3, QCoreApplication.translate("CreateNewProject", u"L", None))
        self.ModelScaleSelection.setItemText(4, QCoreApplication.translate("CreateNewProject", u"H", None))

        self.KeypointNameLabel.setText(QCoreApplication.translate("CreateNewProject", u"Keypoints Name", None))
        self.ClassesNameLabel.setText(QCoreApplication.translate("CreateNewProject", u"Classes Name", None))
        self.KeypointNumber1.setText(QCoreApplication.translate("CreateNewProject", u"1.", None))
        self.ClassNumber1.setText(QCoreApplication.translate("CreateNewProject", u"1.", None))
        self.SourceDataSelected.setText(QCoreApplication.translate("CreateNewProject", u"0 source data selected", None))
        self.SourceDataListDisplayGroup.setTitle("")
        self.CreateProjectBase.setText(QCoreApplication.translate("CreateNewProject", u"Create Project", None))
        self.BrowseSourceData.setText(QCoreApplication.translate("CreateNewProject", u"Browse folders for images or videos", None))
        self.CopySourceData.setText(QCoreApplication.translate("CreateNewProject", u"Copy Source (inlcude images and videos) to project folder", None))
        self.ClearSourceDataList.setText(QCoreApplication.translate("CreateNewProject", u"Clear", None))
    # retranslateUi

