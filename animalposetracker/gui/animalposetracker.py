# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'animalposetrackerdWJChz.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QSpacerItem, QToolBar, QVBoxLayout, QWidget)

class AnimalPoseTracker(object):
    def setupUi(self, AnimalPoseTracker):
        if not AnimalPoseTracker.objectName():
            AnimalPoseTracker.setObjectName(u"AnimalPoseTracker")
        AnimalPoseTracker.setWindowModality(Qt.WindowModality.NonModal)
        AnimalPoseTracker.resize(636, 639)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AnimalPoseTracker.sizePolicy().hasHeightForWidth())
        AnimalPoseTracker.setSizePolicy(sizePolicy)
        font = QFont()
        font.setFamilies([u"Times New Roman"])
        AnimalPoseTracker.setFont(font)
        icon = QIcon()
        icon.addFile(u"assets/logo.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        AnimalPoseTracker.setWindowIcon(icon)
        AnimalPoseTracker.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        AnimalPoseTracker.setAutoFillBackground(False)
        self.actionFileLoadProject = QAction(AnimalPoseTracker)
        self.actionFileLoadProject.setObjectName(u"actionFileLoadProject")
        icon1 = QIcon(QIcon.fromTheme(u"document-open"))
        self.actionFileLoadProject.setIcon(icon1)
        self.actionOpenRecent = QAction(AnimalPoseTracker)
        self.actionOpenRecent.setObjectName(u"actionOpenRecent")
        icon2 = QIcon(QIcon.fromTheme(u"document-open-recent"))
        self.actionOpenRecent.setIcon(icon2)
        self.actionSave = QAction(AnimalPoseTracker)
        self.actionSave.setObjectName(u"actionSave")
        icon3 = QIcon(QIcon.fromTheme(u"document-save"))
        self.actionSave.setIcon(icon3)
        self.actionExit = QAction(AnimalPoseTracker)
        self.actionExit.setObjectName(u"actionExit")
        icon4 = QIcon(QIcon.fromTheme(u"application-exit"))
        self.actionExit.setIcon(icon4)
        self.actionAnimalPoseLabelling = QAction(AnimalPoseTracker)
        self.actionAnimalPoseLabelling.setObjectName(u"actionAnimalPoseLabelling")
        self.actionAnimalPoseInfercence = QAction(AnimalPoseTracker)
        self.actionAnimalPoseInfercence.setObjectName(u"actionAnimalPoseInfercence")
        self.actionDoc = QAction(AnimalPoseTracker)
        self.actionDoc.setObjectName(u"actionDoc")
        self.actionCheckUpdated = QAction(AnimalPoseTracker)
        self.actionCheckUpdated.setObjectName(u"actionCheckUpdated")
        icon5 = QIcon(QIcon.fromTheme(u"update"))
        self.actionCheckUpdated.setIcon(icon5)
        self.actionHelp = QAction(AnimalPoseTracker)
        self.actionHelp.setObjectName(u"actionHelp")
        icon6 = QIcon(QIcon.fromTheme(u"help-browser"))
        self.actionHelp.setIcon(icon6)
        self.actionCreateNewProject = QAction(AnimalPoseTracker)
        self.actionCreateNewProject.setObjectName(u"actionCreateNewProject")
        self.actionCreateNewProject.setCheckable(False)
        icon7 = QIcon(QIcon.fromTheme(u"document-new"))
        self.actionCreateNewProject.setIcon(icon7)
        self.actionLoadProject = QAction(AnimalPoseTracker)
        self.actionLoadProject.setObjectName(u"actionLoadProject")
        self.actionLoadProject.setIcon(icon1)
        self.actionPublicDatasetsProject = QAction(AnimalPoseTracker)
        self.actionPublicDatasetsProject.setObjectName(u"actionPublicDatasetsProject")
        icon8 = QIcon(QIcon.fromTheme(u"edit-copy"))
        self.actionPublicDatasetsProject.setIcon(icon8)
        self.actionDark = QAction(AnimalPoseTracker)
        self.actionDark.setObjectName(u"actionDark")
        self.actionLight = QAction(AnimalPoseTracker)
        self.actionLight.setObjectName(u"actionLight")
        self.actionFileCreateNewProject = QAction(AnimalPoseTracker)
        self.actionFileCreateNewProject.setObjectName(u"actionFileCreateNewProject")
        self.actionFileCreateNewProject.setIcon(icon7)
        self.MainPage = QWidget(AnimalPoseTracker)
        self.MainPage.setObjectName(u"MainPage")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.MainPage.sizePolicy().hasHeightForWidth())
        self.MainPage.setSizePolicy(sizePolicy1)
        self.MainPage.setAutoFillBackground(True)
        self.verticalLayout = QVBoxLayout(self.MainPage)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(20, 20, 20, 20)
        self.Welcome = QLabel(self.MainPage)
        self.Welcome.setObjectName(u"Welcome")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.Welcome.sizePolicy().hasHeightForWidth())
        self.Welcome.setSizePolicy(sizePolicy2)
        font1 = QFont()
        font1.setFamilies([u"Times New Roman"])
        font1.setPointSize(24)
        self.Welcome.setFont(font1)
        self.Welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.Welcome)

        self.LogoLayout = QHBoxLayout()
        self.LogoLayout.setObjectName(u"LogoLayout")
        self.horizontalSpacer2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.LogoLayout.addItem(self.horizontalSpacer2)

        self.label = QLabel(self.MainPage)
        self.label.setObjectName(u"label")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy3)
        self.label.setMaximumSize(QSize(200, 200))
        self.label.setTextFormat(Qt.TextFormat.AutoText)
        self.label.setPixmap(QPixmap(u"assets/logo_transparent.png"))
        self.label.setScaledContents(True)

        self.LogoLayout.addWidget(self.label)

        self.horizontalSpacer1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.LogoLayout.addItem(self.horizontalSpacer1)


        self.verticalLayout.addLayout(self.LogoLayout)

        self.Introdution = QLabel(self.MainPage)
        self.Introdution.setObjectName(u"Introdution")
        sizePolicy2.setHeightForWidth(self.Introdution.sizePolicy().hasHeightForWidth())
        self.Introdution.setSizePolicy(sizePolicy2)
        font2 = QFont()
        font2.setFamilies([u"Times New Roman"])
        font2.setPointSize(11)
        self.Introdution.setFont(font2)
        self.Introdution.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.Introdution.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.Introdution.setWordWrap(True)

        self.verticalLayout.addWidget(self.Introdution)

        self.Note = QLabel(self.MainPage)
        self.Note.setObjectName(u"Note")
        sizePolicy2.setHeightForWidth(self.Note.sizePolicy().hasHeightForWidth())
        self.Note.setSizePolicy(sizePolicy2)
        self.Note.setFont(font2)
        self.Note.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.Note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.Note.setWordWrap(True)

        self.verticalLayout.addWidget(self.Note)

        self.ToolsButtonLayout = QHBoxLayout()
        self.ToolsButtonLayout.setObjectName(u"ToolsButtonLayout")
        self.horizontalSpacer4 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.ToolsButtonLayout.addItem(self.horizontalSpacer4)

        self.CreateNewProjectButton = QPushButton(self.MainPage)
        self.CreateNewProjectButton.setObjectName(u"CreateNewProjectButton")
        sizePolicy.setHeightForWidth(self.CreateNewProjectButton.sizePolicy().hasHeightForWidth())
        self.CreateNewProjectButton.setSizePolicy(sizePolicy)
        self.CreateNewProjectButton.setMinimumSize(QSize(160, 25))

        self.ToolsButtonLayout.addWidget(self.CreateNewProjectButton)

        self.horizontalSpacer6 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.ToolsButtonLayout.addItem(self.horizontalSpacer6)

        self.LoadProjectButton = QPushButton(self.MainPage)
        self.LoadProjectButton.setObjectName(u"LoadProjectButton")
        sizePolicy2.setHeightForWidth(self.LoadProjectButton.sizePolicy().hasHeightForWidth())
        self.LoadProjectButton.setSizePolicy(sizePolicy2)
        self.LoadProjectButton.setMinimumSize(QSize(160, 25))

        self.ToolsButtonLayout.addWidget(self.LoadProjectButton)

        self.horizontalSpacer3 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.ToolsButtonLayout.addItem(self.horizontalSpacer3)

        self.PublicDatasetsProjectButton = QPushButton(self.MainPage)
        self.PublicDatasetsProjectButton.setObjectName(u"PublicDatasetsProjectButton")
        sizePolicy2.setHeightForWidth(self.PublicDatasetsProjectButton.sizePolicy().hasHeightForWidth())
        self.PublicDatasetsProjectButton.setSizePolicy(sizePolicy2)
        self.PublicDatasetsProjectButton.setMinimumSize(QSize(160, 25))

        self.ToolsButtonLayout.addWidget(self.PublicDatasetsProjectButton)

        self.horizontalSpacer5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.ToolsButtonLayout.addItem(self.horizontalSpacer5)


        self.verticalLayout.addLayout(self.ToolsButtonLayout)

        AnimalPoseTracker.setCentralWidget(self.MainPage)
        self.menubar = QMenuBar(AnimalPoseTracker)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 636, 33))
        self.menuAnimalPoseTracker = QMenu(self.menubar)
        self.menuAnimalPoseTracker.setObjectName(u"menuAnimalPoseTracker")
        sizePolicy1.setHeightForWidth(self.menuAnimalPoseTracker.sizePolicy().hasHeightForWidth())
        self.menuAnimalPoseTracker.setSizePolicy(sizePolicy1)
        self.menuTools = QMenu(self.menubar)
        self.menuTools.setObjectName(u"menuTools")
        sizePolicy1.setHeightForWidth(self.menuTools.sizePolicy().hasHeightForWidth())
        self.menuTools.setSizePolicy(sizePolicy1)
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        sizePolicy1.setHeightForWidth(self.menuHelp.sizePolicy().hasHeightForWidth())
        self.menuHelp.setSizePolicy(sizePolicy1)
        self.menuView = QMenu(self.menubar)
        self.menuView.setObjectName(u"menuView")
        self.menuTheme = QMenu(self.menuView)
        self.menuTheme.setObjectName(u"menuTheme")
        AnimalPoseTracker.setMenuBar(self.menubar)
        self.toolBar = QToolBar(AnimalPoseTracker)
        self.toolBar.setObjectName(u"toolBar")
        sizePolicy1.setHeightForWidth(self.toolBar.sizePolicy().hasHeightForWidth())
        self.toolBar.setSizePolicy(sizePolicy1)
        AnimalPoseTracker.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolBar)

        self.menubar.addAction(self.menuAnimalPoseTracker.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuAnimalPoseTracker.addAction(self.actionFileCreateNewProject)
        self.menuAnimalPoseTracker.addAction(self.actionFileLoadProject)
        self.menuAnimalPoseTracker.addAction(self.actionOpenRecent)
        self.menuAnimalPoseTracker.addAction(self.actionSave)
        self.menuAnimalPoseTracker.addAction(self.actionExit)
        self.menuTools.addAction(self.actionAnimalPoseLabelling)
        self.menuTools.addAction(self.actionAnimalPoseInfercence)
        self.menuHelp.addAction(self.actionDoc)
        self.menuHelp.addAction(self.actionCheckUpdated)
        self.menuHelp.addAction(self.actionHelp)
        self.menuView.addAction(self.menuTheme.menuAction())
        self.menuTheme.addAction(self.actionDark)
        self.menuTheme.addAction(self.actionLight)
        self.toolBar.addAction(self.actionCreateNewProject)
        self.toolBar.addAction(self.actionLoadProject)
        self.toolBar.addAction(self.actionPublicDatasetsProject)
        self.toolBar.addAction(self.actionHelp)

        self.retranslateUi(AnimalPoseTracker)

        QMetaObject.connectSlotsByName(AnimalPoseTracker)
    # setupUi

    def retranslateUi(self, AnimalPoseTracker):
        AnimalPoseTracker.setWindowTitle(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker", None))
        self.actionFileLoadProject.setText(QCoreApplication.translate("AnimalPoseTracker", u"Open Old Project Ctrl+O", None))
        self.actionOpenRecent.setText(QCoreApplication.translate("AnimalPoseTracker", u"Open Recent", None))
        self.actionSave.setText(QCoreApplication.translate("AnimalPoseTracker", u"Save", None))
        self.actionExit.setText(QCoreApplication.translate("AnimalPoseTracker", u"Exit", None))
        self.actionAnimalPoseLabelling.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseLabelling", None))
        self.actionAnimalPoseInfercence.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseInfercence", None))
        self.actionDoc.setText(QCoreApplication.translate("AnimalPoseTracker", u"Doc", None))
        self.actionCheckUpdated.setText(QCoreApplication.translate("AnimalPoseTracker", u"Check for Updating", None))
        self.actionHelp.setText(QCoreApplication.translate("AnimalPoseTracker", u"Help", None))
        self.actionCreateNewProject.setText(QCoreApplication.translate("AnimalPoseTracker", u"Create New Project", None))
        self.actionLoadProject.setText(QCoreApplication.translate("AnimalPoseTracker", u"Load Project", None))
        self.actionPublicDatasetsProject.setText(QCoreApplication.translate("AnimalPoseTracker", u"Public Datasets Project", None))
        self.actionDark.setText(QCoreApplication.translate("AnimalPoseTracker", u"Dark", None))
        self.actionLight.setText(QCoreApplication.translate("AnimalPoseTracker", u"Light", None))
        self.actionFileCreateNewProject.setText(QCoreApplication.translate("AnimalPoseTracker", u"New Project Ctrl+N", None))
        self.Welcome.setText(QCoreApplication.translate("AnimalPoseTracker", u"Welcome to the AnimalPoseTracker!", None))
        self.label.setText("")
        self.Introdution.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker is an open source tool for animal pose estimation with deep learning. Wu X. | https://wux024.github.io", None))
        self.Note.setText(QCoreApplication.translate("AnimalPoseTracker", u"To get started, you can create a new project, load an existing project, or use opensource public datasets.", None))
        self.CreateNewProjectButton.setText(QCoreApplication.translate("AnimalPoseTracker", u"Create New Project", None))
        self.LoadProjectButton.setText(QCoreApplication.translate("AnimalPoseTracker", u"Load Project", None))
        self.PublicDatasetsProjectButton.setText(QCoreApplication.translate("AnimalPoseTracker", u"Public Datasets Project", None))
        self.menuAnimalPoseTracker.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"File", None))
        self.menuTools.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"Tools", None))
        self.menuHelp.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"Help", None))
        self.menuView.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"View", None))
        self.menuTheme.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"Theme", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("AnimalPoseTracker", u"toolBar", None))
    # retranslateUi

