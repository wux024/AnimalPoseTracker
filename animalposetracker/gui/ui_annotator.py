# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'annotatorQnpgCF.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGraphicsView,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMenu, QMenuBar, QPushButton,
    QSizePolicy, QSlider, QStatusBar, QToolBar,
    QTreeView, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget)

from animalposetracker.gui import LOGO_SMALL_PATH

class Ui_Annotator(object):
    def setupUi(self, Annotator):
        if not Annotator.objectName():
            Annotator.setObjectName(u"Annotator")
        Annotator.resize(1200, 800)
        icon = QIcon()
        icon.addFile(LOGO_SMALL_PATH, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        Annotator.setWindowIcon(icon)
        self.actionOpenConfigureFile = QAction(Annotator)
        self.actionOpenConfigureFile.setObjectName(u"actionOpenConfigureFile")
        icon1 = QIcon(QIcon.fromTheme(u"document-open"))
        self.actionOpenConfigureFile.setIcon(icon1)
        self.actionOpenFileFolder = QAction(Annotator)
        self.actionOpenFileFolder.setObjectName(u"actionOpenFileFolder")
        icon2 = QIcon(QIcon.fromTheme(u"folder-open"))
        self.actionOpenFileFolder.setIcon(icon2)
        self.actionNextFrame = QAction(Annotator)
        self.actionNextFrame.setObjectName(u"actionNextFrame")
        icon3 = QIcon(QIcon.fromTheme(u"go-next"))
        self.actionNextFrame.setIcon(icon3)
        self.actionNextFrame.setMenuRole(QAction.MenuRole.NoRole)
        self.actionPreviousFrame = QAction(Annotator)
        self.actionPreviousFrame.setObjectName(u"actionPreviousFrame")
        icon4 = QIcon(QIcon.fromTheme(u"go-previous"))
        self.actionPreviousFrame.setIcon(icon4)
        self.actionPreviousFrame.setMenuRole(QAction.MenuRole.NoRole)
        self.actionSaveLabel = QAction(Annotator)
        self.actionSaveLabel.setObjectName(u"actionSaveLabel")
        icon5 = QIcon(QIcon.fromTheme(u"document-save"))
        self.actionSaveLabel.setIcon(icon5)
        self.actionSaveLabel.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDeleteLabel = QAction(Annotator)
        self.actionDeleteLabel.setObjectName(u"actionDeleteLabel")
        icon6 = QIcon(QIcon.fromTheme(u"edit-delete"))
        self.actionDeleteLabel.setIcon(icon6)
        self.actionDeleteLabel.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDrawBBox = QAction(Annotator)
        self.actionDrawBBox.setObjectName(u"actionDrawBBox")
        icon7 = QIcon(QIcon.fromTheme(u"media-playback-stop"))
        self.actionDrawBBox.setIcon(icon7)
        self.actionDrawBBox.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDrawPoint = QAction(Annotator)
        self.actionDrawPoint.setObjectName(u"actionDrawPoint")
        icon8 = QIcon(QIcon.fromTheme(u"media-optical"))
        self.actionDrawPoint.setIcon(icon8)
        self.actionDrawPoint.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDrawLine = QAction(Annotator)
        self.actionDrawLine.setObjectName(u"actionDrawLine")
        icon9 = QIcon(QIcon.fromTheme(u"format-text-italic"))
        self.actionDrawLine.setIcon(icon9)
        self.actionDrawLine.setMenuRole(QAction.MenuRole.NoRole)
        self.actionNoLabel = QAction(Annotator)
        self.actionNoLabel.setObjectName(u"actionNoLabel")
        icon10 = QIcon(QIcon.fromTheme(u"zoom-original"))
        self.actionNoLabel.setIcon(icon10)
        self.actionNoLabel.setMenuRole(QAction.MenuRole.NoRole)
        self.actionHelp = QAction(Annotator)
        self.actionHelp.setObjectName(u"actionHelp")
        icon11 = QIcon(QIcon.fromTheme(u"help-browser"))
        self.actionHelp.setIcon(icon11)
        self.actionDark = QAction(Annotator)
        self.actionDark.setObjectName(u"actionDark")
        self.actionLight = QAction(Annotator)
        self.actionLight.setObjectName(u"actionLight")
        self.actionBuildSkeleton = QAction(Annotator)
        self.actionBuildSkeleton.setObjectName(u"actionBuildSkeleton")
        icon12 = QIcon(QIcon.fromTheme(u"format-text-strikethrough"))
        self.actionBuildSkeleton.setIcon(icon12)
        self.actionBuildSkeleton.setMenuRole(QAction.MenuRole.NoRole)
        self.actionExportYOLOFormat = QAction(Annotator)
        self.actionExportYOLOFormat.setObjectName(u"actionExportYOLOFormat")
        self.actionExportCOCOFormat = QAction(Annotator)
        self.actionExportCOCOFormat.setObjectName(u"actionExportCOCOFormat")
        self.AnnotatorLayout = QWidget(Annotator)
        self.AnnotatorLayout.setObjectName(u"AnnotatorLayout")
        self.AnnotatorHLayout = QHBoxLayout(self.AnnotatorLayout)
        self.AnnotatorHLayout.setObjectName(u"AnnotatorHLayout")
        self.ToolsLayout = QVBoxLayout()
        self.ToolsLayout.setObjectName(u"ToolsLayout")
        self.SubToolsLayout = QHBoxLayout()
        self.SubToolsLayout.setSpacing(5)
        self.SubToolsLayout.setObjectName(u"SubToolsLayout")
        self.KeypointsSeletion = QComboBox(self.AnnotatorLayout)
        self.KeypointsSeletion.setObjectName(u"KeypointsSeletion")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.KeypointsSeletion.sizePolicy().hasHeightForWidth())
        self.KeypointsSeletion.setSizePolicy(sizePolicy)
        self.KeypointsSeletion.setMinimumSize(QSize(120, 0))
        self.KeypointsSeletion.setMaximumSize(QSize(120, 16777215))

        self.SubToolsLayout.addWidget(self.KeypointsSeletion)

        self.ClassesSelection = QComboBox(self.AnnotatorLayout)
        self.ClassesSelection.setObjectName(u"ClassesSelection")
        sizePolicy.setHeightForWidth(self.ClassesSelection.sizePolicy().hasHeightForWidth())
        self.ClassesSelection.setSizePolicy(sizePolicy)
        self.ClassesSelection.setMinimumSize(QSize(120, 0))
        self.ClassesSelection.setMaximumSize(QSize(120, 16777215))

        self.SubToolsLayout.addWidget(self.ClassesSelection)

        self.Visible = QComboBox(self.AnnotatorLayout)
        self.Visible.setObjectName(u"Visible")
        sizePolicy.setHeightForWidth(self.Visible.sizePolicy().hasHeightForWidth())
        self.Visible.setSizePolicy(sizePolicy)
        self.Visible.setMinimumSize(QSize(120, 0))
        self.Visible.setMaximumSize(QSize(120, 16777215))

        self.SubToolsLayout.addWidget(self.Visible)


        self.ToolsLayout.addLayout(self.SubToolsLayout)

        self.SkeletonSelection = QComboBox(self.AnnotatorLayout)
        self.SkeletonSelection.setObjectName(u"SkeletonSelection")
        sizePolicy.setHeightForWidth(self.SkeletonSelection.sizePolicy().hasHeightForWidth())
        self.SkeletonSelection.setSizePolicy(sizePolicy)
        self.SkeletonSelection.setMinimumSize(QSize(370, 0))
        self.SkeletonSelection.setMaximumSize(QSize(370, 16777215))

        self.ToolsLayout.addWidget(self.SkeletonSelection)

        self.EditLayout = QHBoxLayout()
        self.EditLayout.setSpacing(5)
        self.EditLayout.setObjectName(u"EditLayout")
        self.EditClasses = QPushButton(self.AnnotatorLayout)
        self.EditClasses.setObjectName(u"EditClasses")
        sizePolicy.setHeightForWidth(self.EditClasses.sizePolicy().hasHeightForWidth())
        self.EditClasses.setSizePolicy(sizePolicy)
        self.EditClasses.setMinimumSize(QSize(120, 25))
        self.EditClasses.setMaximumSize(QSize(120, 25))

        self.EditLayout.addWidget(self.EditClasses)

        self.EditKeypoints = QPushButton(self.AnnotatorLayout)
        self.EditKeypoints.setObjectName(u"EditKeypoints")
        sizePolicy.setHeightForWidth(self.EditKeypoints.sizePolicy().hasHeightForWidth())
        self.EditKeypoints.setSizePolicy(sizePolicy)
        self.EditKeypoints.setMinimumSize(QSize(120, 25))
        self.EditKeypoints.setMaximumSize(QSize(120, 25))

        self.EditLayout.addWidget(self.EditKeypoints)

        self.EditSkeleton = QPushButton(self.AnnotatorLayout)
        self.EditSkeleton.setObjectName(u"EditSkeleton")
        sizePolicy.setHeightForWidth(self.EditSkeleton.sizePolicy().hasHeightForWidth())
        self.EditSkeleton.setSizePolicy(sizePolicy)
        self.EditSkeleton.setMinimumSize(QSize(120, 25))
        self.EditSkeleton.setMaximumSize(QSize(120, 25))

        self.EditLayout.addWidget(self.EditSkeleton)


        self.ToolsLayout.addLayout(self.EditLayout)

        self.ConfigureDisplay = QTreeWidget(self.AnnotatorLayout)
        self.ConfigureDisplay.setObjectName(u"ConfigureDisplay")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.ConfigureDisplay.sizePolicy().hasHeightForWidth())
        self.ConfigureDisplay.setSizePolicy(sizePolicy1)
        self.ConfigureDisplay.setMinimumSize(QSize(370, 0))
        self.ConfigureDisplay.setMaximumSize(QSize(370, 16777215))

        self.ToolsLayout.addWidget(self.ConfigureDisplay)

        self.LabelInformationViewLabel = QLabel(self.AnnotatorLayout)
        self.LabelInformationViewLabel.setObjectName(u"LabelInformationViewLabel")
        self.LabelInformationViewLabel.setMinimumSize(QSize(370, 25))
        self.LabelInformationViewLabel.setMaximumSize(QSize(370, 25))

        self.ToolsLayout.addWidget(self.LabelInformationViewLabel)

        self.LabelInformationView = QTreeView(self.AnnotatorLayout)
        self.LabelInformationView.setObjectName(u"LabelInformationView")
        sizePolicy1.setHeightForWidth(self.LabelInformationView.sizePolicy().hasHeightForWidth())
        self.LabelInformationView.setSizePolicy(sizePolicy1)
        self.LabelInformationView.setMinimumSize(QSize(370, 0))
        self.LabelInformationView.setMaximumSize(QSize(370, 16777215))

        self.ToolsLayout.addWidget(self.LabelInformationView)


        self.AnnotatorHLayout.addLayout(self.ToolsLayout)

        self.LabelAreaLayout = QVBoxLayout()
        self.LabelAreaLayout.setObjectName(u"LabelAreaLayout")
        self.LabelToolsLayout = QHBoxLayout()
        self.LabelToolsLayout.setObjectName(u"LabelToolsLayout")
        self.LineWidthLayout = QHBoxLayout()
        self.LineWidthLayout.setObjectName(u"LineWidthLayout")
        self.LineWidthLabel = QLabel(self.AnnotatorLayout)
        self.LineWidthLabel.setObjectName(u"LineWidthLabel")

        self.LineWidthLayout.addWidget(self.LineWidthLabel)

        self.LineWidth = QSlider(self.AnnotatorLayout)
        self.LineWidth.setObjectName(u"LineWidth")
        self.LineWidth.setMinimum(1)
        self.LineWidth.setMaximum(10)
        self.LineWidth.setValue(2)
        self.LineWidth.setOrientation(Qt.Orientation.Horizontal)

        self.LineWidthLayout.addWidget(self.LineWidth)


        self.LabelToolsLayout.addLayout(self.LineWidthLayout)

        self.RadiusLayout = QHBoxLayout()
        self.RadiusLayout.setObjectName(u"RadiusLayout")
        self.RadiusLabel = QLabel(self.AnnotatorLayout)
        self.RadiusLabel.setObjectName(u"RadiusLabel")

        self.RadiusLayout.addWidget(self.RadiusLabel)

        self.Radius = QSlider(self.AnnotatorLayout)
        self.Radius.setObjectName(u"Radius")
        self.Radius.setMinimum(1)
        self.Radius.setMaximum(10)
        self.Radius.setValue(3)
        self.Radius.setOrientation(Qt.Orientation.Horizontal)

        self.RadiusLayout.addWidget(self.Radius)


        self.LabelToolsLayout.addLayout(self.RadiusLayout)

        self.IsCrowd = QCheckBox(self.AnnotatorLayout)
        self.IsCrowd.setObjectName(u"IsCrowd")

        self.LabelToolsLayout.addWidget(self.IsCrowd)


        self.LabelAreaLayout.addLayout(self.LabelToolsLayout)

        self.FrameDisplayGroup = QGroupBox(self.AnnotatorLayout)
        self.FrameDisplayGroup.setObjectName(u"FrameDisplayGroup")
        self.FrameDisplayGroupLayout = QVBoxLayout(self.FrameDisplayGroup)
        self.FrameDisplayGroupLayout.setObjectName(u"FrameDisplayGroupLayout")
        self.FrameDisplay = QGraphicsView(self.FrameDisplayGroup)
        self.FrameDisplay.setObjectName(u"FrameDisplay")

        self.FrameDisplayGroupLayout.addWidget(self.FrameDisplay)


        self.LabelAreaLayout.addWidget(self.FrameDisplayGroup)


        self.AnnotatorHLayout.addLayout(self.LabelAreaLayout)

        Annotator.setCentralWidget(self.AnnotatorLayout)
        self.menubar = QMenuBar(Annotator)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1200, 33))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName(u"menuView")
        self.menuTheme = QMenu(self.menuView)
        self.menuTheme.setObjectName(u"menuTheme")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        self.menuExport = QMenu(self.menubar)
        self.menuExport.setObjectName(u"menuExport")
        Annotator.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Annotator)
        self.statusbar.setObjectName(u"statusbar")
        Annotator.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(Annotator)
        self.toolBar.setObjectName(u"toolBar")
        Annotator.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.toolBar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuExport.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionOpenConfigureFile)
        self.menuFile.addAction(self.actionOpenFileFolder)
        self.menuView.addAction(self.menuTheme.menuAction())
        self.menuTheme.addAction(self.actionDark)
        self.menuTheme.addAction(self.actionLight)
        self.menuHelp.addAction(self.actionHelp)
        self.menuExport.addAction(self.actionExportYOLOFormat)
        self.menuExport.addAction(self.actionExportCOCOFormat)
        self.toolBar.addAction(self.actionOpenConfigureFile)
        self.toolBar.addAction(self.actionOpenFileFolder)
        self.toolBar.addAction(self.actionNextFrame)
        self.toolBar.addAction(self.actionPreviousFrame)
        self.toolBar.addAction(self.actionSaveLabel)
        self.toolBar.addAction(self.actionDeleteLabel)
        self.toolBar.addAction(self.actionNoLabel)
        self.toolBar.addAction(self.actionDrawBBox)
        self.toolBar.addAction(self.actionDrawPoint)
        self.toolBar.addAction(self.actionDrawLine)
        self.toolBar.addAction(self.actionBuildSkeleton)

        self.retranslateUi(Annotator)

        QMetaObject.connectSlotsByName(Annotator)
    # setupUi

    def retranslateUi(self, Annotator):
        Annotator.setWindowTitle(QCoreApplication.translate("Annotator", u"AnimalPoseAnnotator", None))
        self.actionOpenConfigureFile.setText(QCoreApplication.translate("Annotator", u"Open Configure File", None))
        self.actionOpenFileFolder.setText(QCoreApplication.translate("Annotator", u"Open File Folder", None))
        self.actionNextFrame.setText(QCoreApplication.translate("Annotator", u"NextFrame", None))
        self.actionPreviousFrame.setText(QCoreApplication.translate("Annotator", u"PreviousFrame", None))
        self.actionSaveLabel.setText(QCoreApplication.translate("Annotator", u"SaveLabel", None))
        self.actionDeleteLabel.setText(QCoreApplication.translate("Annotator", u"DeleteLabel", None))
        self.actionDrawBBox.setText(QCoreApplication.translate("Annotator", u"DrawBBox", None))
        self.actionDrawPoint.setText(QCoreApplication.translate("Annotator", u"DrawPoint", None))
        self.actionDrawLine.setText(QCoreApplication.translate("Annotator", u"DrawLine", None))
        self.actionNoLabel.setText(QCoreApplication.translate("Annotator", u"NoLabel", None))
        self.actionHelp.setText(QCoreApplication.translate("Annotator", u"Help", None))
        self.actionDark.setText(QCoreApplication.translate("Annotator", u"Dark", None))
        self.actionLight.setText(QCoreApplication.translate("Annotator", u"Light", None))
        self.actionBuildSkeleton.setText(QCoreApplication.translate("Annotator", u"BuildSkeleton", None))
        self.actionExportYOLOFormat.setText(QCoreApplication.translate("Annotator", u"Export YOLO Format", None))
        self.actionExportCOCOFormat.setText(QCoreApplication.translate("Annotator", u"Export COCO Format", None))
        self.EditClasses.setText(QCoreApplication.translate("Annotator", u"Edit Classes", None))
        self.EditKeypoints.setText(QCoreApplication.translate("Annotator", u"Edit Keypoints", None))
        self.EditSkeleton.setText(QCoreApplication.translate("Annotator", u"Edit Sekelton", None))
        ___qtreewidgetitem = self.ConfigureDisplay.headerItem()
        ___qtreewidgetitem.setText(1, QCoreApplication.translate("Annotator", u"Keypoint Name", None));
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("Annotator", u"Keypoint Number", None));
        self.LabelInformationViewLabel.setText(QCoreApplication.translate("Annotator", u"Label Information View", None))
        self.LineWidthLabel.setText(QCoreApplication.translate("Annotator", u"Line Width", None))
        self.RadiusLabel.setText(QCoreApplication.translate("Annotator", u"Radius", None))
        self.IsCrowd.setText(QCoreApplication.translate("Annotator", u"iscrowd", None))
        self.FrameDisplayGroup.setTitle("")
        self.menuFile.setTitle(QCoreApplication.translate("Annotator", u"File", None))
        self.menuView.setTitle(QCoreApplication.translate("Annotator", u"View", None))
        self.menuTheme.setTitle(QCoreApplication.translate("Annotator", u"Theme", None))
        self.menuHelp.setTitle(QCoreApplication.translate("Annotator", u"Help", None))
        self.menuExport.setTitle(QCoreApplication.translate("Annotator", u"Export", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("Annotator", u"toolBar", None))
    # retranslateUi

