# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'animalposeannotatorygOtJR.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QGraphicsView, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QToolBar, QTreeView, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget)

from animalposetracker.gui import LOGO_SMALL_PATH

class Ui_AnimalPoseAnnotator(object):
    def setupUi(self, AnimalPoseAnnotator):
        if not AnimalPoseAnnotator.objectName():
            AnimalPoseAnnotator.setObjectName(u"AnimalPoseAnnotator")
        AnimalPoseAnnotator.resize(1120, 642)
        icon = QIcon()
        icon.addFile(LOGO_SMALL_PATH, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        AnimalPoseAnnotator.setWindowIcon(icon)
        self.actionOpenConfigureFile = QAction(AnimalPoseAnnotator)
        self.actionOpenConfigureFile.setObjectName(u"actionOpenConfigureFile")
        icon1 = QIcon(QIcon.fromTheme(u"document-open"))
        self.actionOpenConfigureFile.setIcon(icon1)
        self.actionOpenFileFolder = QAction(AnimalPoseAnnotator)
        self.actionOpenFileFolder.setObjectName(u"actionOpenFileFolder")
        icon2 = QIcon(QIcon.fromTheme(u"folder-open"))
        self.actionOpenFileFolder.setIcon(icon2)
        self.actionNextFrame = QAction(AnimalPoseAnnotator)
        self.actionNextFrame.setObjectName(u"actionNextFrame")
        icon3 = QIcon(QIcon.fromTheme(u"go-next"))
        self.actionNextFrame.setIcon(icon3)
        self.actionNextFrame.setMenuRole(QAction.MenuRole.NoRole)
        self.actionPreviousFrame = QAction(AnimalPoseAnnotator)
        self.actionPreviousFrame.setObjectName(u"actionPreviousFrame")
        icon4 = QIcon(QIcon.fromTheme(u"go-previous"))
        self.actionPreviousFrame.setIcon(icon4)
        self.actionPreviousFrame.setMenuRole(QAction.MenuRole.NoRole)
        self.actionSaveLabel = QAction(AnimalPoseAnnotator)
        self.actionSaveLabel.setObjectName(u"actionSaveLabel")
        icon5 = QIcon(QIcon.fromTheme(u"document-save"))
        self.actionSaveLabel.setIcon(icon5)
        self.actionSaveLabel.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDeleteLabel = QAction(AnimalPoseAnnotator)
        self.actionDeleteLabel.setObjectName(u"actionDeleteLabel")
        icon6 = QIcon(QIcon.fromTheme(u"edit-delete"))
        self.actionDeleteLabel.setIcon(icon6)
        self.actionDeleteLabel.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDrawBBox = QAction(AnimalPoseAnnotator)
        self.actionDrawBBox.setObjectName(u"actionDrawBBox")
        icon7 = QIcon(QIcon.fromTheme(u"media-playback-stop"))
        self.actionDrawBBox.setIcon(icon7)
        self.actionDrawBBox.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDrawPoint = QAction(AnimalPoseAnnotator)
        self.actionDrawPoint.setObjectName(u"actionDrawPoint")
        icon8 = QIcon(QIcon.fromTheme(u"media-optical"))
        self.actionDrawPoint.setIcon(icon8)
        self.actionDrawPoint.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDrawLine = QAction(AnimalPoseAnnotator)
        self.actionDrawLine.setObjectName(u"actionDrawLine")
        icon9 = QIcon(QIcon.fromTheme(u"format-text-italic"))
        self.actionDrawLine.setIcon(icon9)
        self.actionDrawLine.setMenuRole(QAction.MenuRole.NoRole)
        self.actionUndoLasted = QAction(AnimalPoseAnnotator)
        self.actionUndoLasted.setObjectName(u"actionUndoLasted")
        icon10 = QIcon(QIcon.fromTheme(u"edit-undo"))
        self.actionUndoLasted.setIcon(icon10)
        self.actionUndoLasted.setMenuRole(QAction.MenuRole.NoRole)
        self.actionAdd = QAction(AnimalPoseAnnotator)
        self.actionAdd.setObjectName(u"actionAdd")
        icon11 = QIcon(QIcon.fromTheme(u"list-add"))
        self.actionAdd.setIcon(icon11)
        self.actionAdd.setMenuRole(QAction.MenuRole.NoRole)
        self.actionNoLabel = QAction(AnimalPoseAnnotator)
        self.actionNoLabel.setObjectName(u"actionNoLabel")
        icon12 = QIcon(QIcon.fromTheme(u"zoom-original"))
        self.actionNoLabel.setIcon(icon12)
        self.actionNoLabel.setMenuRole(QAction.MenuRole.NoRole)
        self.actionDelete = QAction(AnimalPoseAnnotator)
        self.actionDelete.setObjectName(u"actionDelete")
        icon13 = QIcon(QIcon.fromTheme(u"list-remove"))
        self.actionDelete.setIcon(icon13)
        self.actionDelete.setMenuRole(QAction.MenuRole.NoRole)
        self.actionHelp = QAction(AnimalPoseAnnotator)
        self.actionHelp.setObjectName(u"actionHelp")
        icon14 = QIcon(QIcon.fromTheme(u"help-browser"))
        self.actionHelp.setIcon(icon14)
        self.actionDark = QAction(AnimalPoseAnnotator)
        self.actionDark.setObjectName(u"actionDark")
        self.actionLight = QAction(AnimalPoseAnnotator)
        self.actionLight.setObjectName(u"actionLight")
        self.actionBuildSkeleton = QAction(AnimalPoseAnnotator)
        self.actionBuildSkeleton.setObjectName(u"actionBuildSkeleton")
        icon15 = QIcon(QIcon.fromTheme(u"format-text-strikethrough"))
        self.actionBuildSkeleton.setIcon(icon15)
        self.actionBuildSkeleton.setMenuRole(QAction.MenuRole.NoRole)
        self.AnimalPoseAnnotatorLayout = QWidget(AnimalPoseAnnotator)
        self.AnimalPoseAnnotatorLayout.setObjectName(u"AnimalPoseAnnotatorLayout")
        self.AnimalPoseAnnotatorHLayout = QHBoxLayout(self.AnimalPoseAnnotatorLayout)
        self.AnimalPoseAnnotatorHLayout.setObjectName(u"AnimalPoseAnnotatorHLayout")
        self.ToolsLayout = QVBoxLayout()
        self.ToolsLayout.setObjectName(u"ToolsLayout")
        self.SubToolsLayout = QHBoxLayout()
        self.SubToolsLayout.setSpacing(10)
        self.SubToolsLayout.setObjectName(u"SubToolsLayout")
        self.KeypointsSeletion = QComboBox(self.AnimalPoseAnnotatorLayout)
        self.KeypointsSeletion.setObjectName(u"KeypointsSeletion")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.KeypointsSeletion.sizePolicy().hasHeightForWidth())
        self.KeypointsSeletion.setSizePolicy(sizePolicy)
        self.KeypointsSeletion.setMinimumSize(QSize(180, 0))
        self.KeypointsSeletion.setMaximumSize(QSize(180, 16777215))

        self.SubToolsLayout.addWidget(self.KeypointsSeletion)

        self.ClassesSelection = QComboBox(self.AnimalPoseAnnotatorLayout)
        self.ClassesSelection.setObjectName(u"ClassesSelection")
        sizePolicy.setHeightForWidth(self.ClassesSelection.sizePolicy().hasHeightForWidth())
        self.ClassesSelection.setSizePolicy(sizePolicy)
        self.ClassesSelection.setMinimumSize(QSize(180, 0))
        self.ClassesSelection.setMaximumSize(QSize(180, 16777215))

        self.SubToolsLayout.addWidget(self.ClassesSelection)


        self.ToolsLayout.addLayout(self.SubToolsLayout)

        self.SkeletonSelection = QComboBox(self.AnimalPoseAnnotatorLayout)
        self.SkeletonSelection.setObjectName(u"SkeletonSelection")
        sizePolicy.setHeightForWidth(self.SkeletonSelection.sizePolicy().hasHeightForWidth())
        self.SkeletonSelection.setSizePolicy(sizePolicy)
        self.SkeletonSelection.setMinimumSize(QSize(370, 0))
        self.SkeletonSelection.setMaximumSize(QSize(370, 16777215))

        self.ToolsLayout.addWidget(self.SkeletonSelection)

        self.EditLayout = QHBoxLayout()
        self.EditLayout.setSpacing(5)
        self.EditLayout.setObjectName(u"EditLayout")
        self.EditClasses = QPushButton(self.AnimalPoseAnnotatorLayout)
        self.EditClasses.setObjectName(u"EditClasses")
        sizePolicy.setHeightForWidth(self.EditClasses.sizePolicy().hasHeightForWidth())
        self.EditClasses.setSizePolicy(sizePolicy)
        self.EditClasses.setMinimumSize(QSize(120, 25))
        self.EditClasses.setMaximumSize(QSize(120, 25))

        self.EditLayout.addWidget(self.EditClasses)

        self.EditKeypoints = QPushButton(self.AnimalPoseAnnotatorLayout)
        self.EditKeypoints.setObjectName(u"EditKeypoints")
        sizePolicy.setHeightForWidth(self.EditKeypoints.sizePolicy().hasHeightForWidth())
        self.EditKeypoints.setSizePolicy(sizePolicy)
        self.EditKeypoints.setMinimumSize(QSize(120, 25))
        self.EditKeypoints.setMaximumSize(QSize(120, 25))

        self.EditLayout.addWidget(self.EditKeypoints)

        self.EditSkeleton = QPushButton(self.AnimalPoseAnnotatorLayout)
        self.EditSkeleton.setObjectName(u"EditSkeleton")
        sizePolicy.setHeightForWidth(self.EditSkeleton.sizePolicy().hasHeightForWidth())
        self.EditSkeleton.setSizePolicy(sizePolicy)
        self.EditSkeleton.setMinimumSize(QSize(120, 25))
        self.EditSkeleton.setMaximumSize(QSize(120, 25))

        self.EditLayout.addWidget(self.EditSkeleton)


        self.ToolsLayout.addLayout(self.EditLayout)

        self.ConfigureDisplay = QTreeWidget(self.AnimalPoseAnnotatorLayout)
        self.ConfigureDisplay.setObjectName(u"ConfigureDisplay")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.ConfigureDisplay.sizePolicy().hasHeightForWidth())
        self.ConfigureDisplay.setSizePolicy(sizePolicy1)
        self.ConfigureDisplay.setMinimumSize(QSize(370, 0))
        self.ConfigureDisplay.setMaximumSize(QSize(370, 16777215))

        self.ToolsLayout.addWidget(self.ConfigureDisplay)

        self.LabelInformationViewLabel = QLabel(self.AnimalPoseAnnotatorLayout)
        self.LabelInformationViewLabel.setObjectName(u"LabelInformationViewLabel")
        self.LabelInformationViewLabel.setMinimumSize(QSize(370, 25))
        self.LabelInformationViewLabel.setMaximumSize(QSize(370, 25))

        self.ToolsLayout.addWidget(self.LabelInformationViewLabel)

        self.LabelInformationView = QTreeView(self.AnimalPoseAnnotatorLayout)
        self.LabelInformationView.setObjectName(u"LabelInformationView")
        sizePolicy1.setHeightForWidth(self.LabelInformationView.sizePolicy().hasHeightForWidth())
        self.LabelInformationView.setSizePolicy(sizePolicy1)
        self.LabelInformationView.setMinimumSize(QSize(370, 0))
        self.LabelInformationView.setMaximumSize(QSize(370, 16777215))

        self.ToolsLayout.addWidget(self.LabelInformationView)


        self.AnimalPoseAnnotatorHLayout.addLayout(self.ToolsLayout)

        self.FrameDisplayGroup = QGroupBox(self.AnimalPoseAnnotatorLayout)
        self.FrameDisplayGroup.setObjectName(u"FrameDisplayGroup")
        self.FrameDisplayGroupLayout = QVBoxLayout(self.FrameDisplayGroup)
        self.FrameDisplayGroupLayout.setObjectName(u"FrameDisplayGroupLayout")
        self.FrameDisplay = QGraphicsView(self.FrameDisplayGroup)
        self.FrameDisplay.setObjectName(u"FrameDisplay")

        self.FrameDisplayGroupLayout.addWidget(self.FrameDisplay)


        self.AnimalPoseAnnotatorHLayout.addWidget(self.FrameDisplayGroup)

        AnimalPoseAnnotator.setCentralWidget(self.AnimalPoseAnnotatorLayout)
        self.menubar = QMenuBar(AnimalPoseAnnotator)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1120, 33))
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
        AnimalPoseAnnotator.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(AnimalPoseAnnotator)
        self.statusbar.setObjectName(u"statusbar")
        AnimalPoseAnnotator.setStatusBar(self.statusbar)
        self.toolBar = QToolBar(AnimalPoseAnnotator)
        self.toolBar.setObjectName(u"toolBar")
        AnimalPoseAnnotator.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.toolBar)

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
        self.toolBar.addAction(self.actionAdd)
        self.toolBar.addAction(self.actionDelete)
        self.toolBar.addAction(self.actionUndoLasted)

        self.retranslateUi(AnimalPoseAnnotator)

        QMetaObject.connectSlotsByName(AnimalPoseAnnotator)
    # setupUi

    def retranslateUi(self, AnimalPoseAnnotator):
        AnimalPoseAnnotator.setWindowTitle(QCoreApplication.translate("AnimalPoseAnnotator", u"AnimalPoseAnnotator", None))
        self.actionOpenConfigureFile.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Open Configure File", None))
        self.actionOpenFileFolder.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Open File Folder", None))
        self.actionNextFrame.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"NextFrame", None))
        self.actionPreviousFrame.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"PreviousFrame", None))
        self.actionSaveLabel.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"SaveLabel", None))
        self.actionDeleteLabel.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"DeleteLabel", None))
        self.actionDrawBBox.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"DrawBBox", None))
        self.actionDrawPoint.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"DrawPoint", None))
        self.actionDrawLine.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"DrawLine", None))
        self.actionUndoLasted.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"UndoLasted", None))
        self.actionAdd.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Add", None))
#if QT_CONFIG(tooltip)
        self.actionAdd.setToolTip(QCoreApplication.translate("AnimalPoseAnnotator", u"Add", None))
#endif // QT_CONFIG(tooltip)
        self.actionNoLabel.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"NoLabel", None))
        self.actionDelete.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Delete", None))
        self.actionHelp.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Help", None))
        self.actionDark.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Dark", None))
        self.actionLight.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Light", None))
        self.actionBuildSkeleton.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"BuildSkeleton", None))
        self.EditClasses.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Edit Classes", None))
        self.EditKeypoints.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Edit Keypoints", None))
        self.EditSkeleton.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Edit Sekelton", None))
        ___qtreewidgetitem = self.ConfigureDisplay.headerItem()
        ___qtreewidgetitem.setText(1, QCoreApplication.translate("AnimalPoseAnnotator", u"Keypoint Name", None));
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("AnimalPoseAnnotator", u"Keypoint Number", None));
        self.LabelInformationViewLabel.setText(QCoreApplication.translate("AnimalPoseAnnotator", u"Label Information View", None))
        self.FrameDisplayGroup.setTitle("")
        self.menuFile.setTitle(QCoreApplication.translate("AnimalPoseAnnotator", u"File", None))
        self.menuView.setTitle(QCoreApplication.translate("AnimalPoseAnnotator", u"View", None))
        self.menuTheme.setTitle(QCoreApplication.translate("AnimalPoseAnnotator", u"Theme", None))
        self.menuHelp.setTitle(QCoreApplication.translate("AnimalPoseAnnotator", u"Help", None))
        self.menuExport.setTitle(QCoreApplication.translate("AnimalPoseAnnotator", u"Export", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("AnimalPoseAnnotator", u"toolBar", None))
    # retranslateUi

