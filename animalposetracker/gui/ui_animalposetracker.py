# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'animalposetrackeroroliq.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLayout, QLineEdit, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QSpacerItem,
    QSpinBox, QStackedWidget, QTabWidget, QToolBar,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from animalposetracker.gui import LOGO_PATH_TRANSPARENT, LOGO_SMALL_PATH

class Ui_AnimalPoseTracker(object):
    def setupUi(self, AnimalPoseTracker):
        if not AnimalPoseTracker.objectName():
            AnimalPoseTracker.setObjectName(u"AnimalPoseTracker")
        AnimalPoseTracker.setWindowModality(Qt.WindowModality.NonModal)
        AnimalPoseTracker.resize(800, 600)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AnimalPoseTracker.sizePolicy().hasHeightForWidth())
        AnimalPoseTracker.setSizePolicy(sizePolicy)
        font = QFont()
        font.setFamilies([u"Times New Roman"])
        AnimalPoseTracker.setFont(font)
        icon = QIcon()
        icon.addFile(LOGO_SMALL_PATH, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        AnimalPoseTracker.setWindowIcon(icon)
        AnimalPoseTracker.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        AnimalPoseTracker.setAutoFillBackground(False)
        self.actionLoadProject = QAction(AnimalPoseTracker)
        self.actionLoadProject.setObjectName(u"actionLoadProject")
        icon1 = QIcon(QIcon.fromTheme(u"document-open"))
        self.actionLoadProject.setIcon(icon1)
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
        self.actionAnnotator = QAction(AnimalPoseTracker)
        self.actionAnnotator.setObjectName(u"actionAnnotator")
        self.actionInferencer = QAction(AnimalPoseTracker)
        self.actionInferencer.setObjectName(u"actionInferencer")
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
        self.actionPublicDatasetsProject = QAction(AnimalPoseTracker)
        self.actionPublicDatasetsProject.setObjectName(u"actionPublicDatasetsProject")
        icon7 = QIcon(QIcon.fromTheme(u"edit-copy"))
        self.actionPublicDatasetsProject.setIcon(icon7)
        self.actionDark = QAction(AnimalPoseTracker)
        self.actionDark.setObjectName(u"actionDark")
        self.actionLight = QAction(AnimalPoseTracker)
        self.actionLight.setObjectName(u"actionLight")
        self.actionCreateNewProject = QAction(AnimalPoseTracker)
        self.actionCreateNewProject.setObjectName(u"actionCreateNewProject")
        icon8 = QIcon(QIcon.fromTheme(u"document-new"))
        self.actionCreateNewProject.setIcon(icon8)
        self.AnimalPoseTrackerLayout = QWidget(AnimalPoseTracker)
        self.AnimalPoseTrackerLayout.setObjectName(u"AnimalPoseTrackerLayout")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.AnimalPoseTrackerLayout.sizePolicy().hasHeightForWidth())
        self.AnimalPoseTrackerLayout.setSizePolicy(sizePolicy1)
        self.AnimalPoseTrackerLayout.setAutoFillBackground(True)
        self.AnimalPoseTrackerVLayout = QVBoxLayout(self.AnimalPoseTrackerLayout)
        self.AnimalPoseTrackerVLayout.setObjectName(u"AnimalPoseTrackerVLayout")
        self.AnimalPoseTrackerPage = QStackedWidget(self.AnimalPoseTrackerLayout)
        self.AnimalPoseTrackerPage.setObjectName(u"AnimalPoseTrackerPage")
        self.MainPage = QWidget()
        self.MainPage.setObjectName(u"MainPage")
        self.MainPageLayout = QVBoxLayout(self.MainPage)
        self.MainPageLayout.setObjectName(u"MainPageLayout")
        self.WelcomeLabel = QLabel(self.MainPage)
        self.WelcomeLabel.setObjectName(u"WelcomeLabel")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.WelcomeLabel.sizePolicy().hasHeightForWidth())
        self.WelcomeLabel.setSizePolicy(sizePolicy2)
        font1 = QFont()
        font1.setFamilies([u"Times New Roman"])
        font1.setPointSize(24)
        self.WelcomeLabel.setFont(font1)
        self.WelcomeLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.MainPageLayout.addWidget(self.WelcomeLabel)

        self.LogoLayout = QHBoxLayout()
        self.LogoLayout.setObjectName(u"LogoLayout")
        self.horizontalSpacer2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.LogoLayout.addItem(self.horizontalSpacer2)

        self.Logo = QLabel(self.MainPage)
        self.Logo.setObjectName(u"Logo")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.Logo.sizePolicy().hasHeightForWidth())
        self.Logo.setSizePolicy(sizePolicy3)
        self.Logo.setMaximumSize(QSize(200, 200))
        self.Logo.setTextFormat(Qt.TextFormat.AutoText)
        self.Logo.setPixmap(QPixmap(LOGO_PATH_TRANSPARENT))
        self.Logo.setScaledContents(True)

        self.LogoLayout.addWidget(self.Logo)

        self.horizontalSpacer1 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.LogoLayout.addItem(self.horizontalSpacer1)


        self.MainPageLayout.addLayout(self.LogoLayout)

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

        self.MainPageLayout.addWidget(self.Introdution)

        self.Note = QLabel(self.MainPage)
        self.Note.setObjectName(u"Note")
        sizePolicy2.setHeightForWidth(self.Note.sizePolicy().hasHeightForWidth())
        self.Note.setSizePolicy(sizePolicy2)
        self.Note.setFont(font2)
        self.Note.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.Note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.Note.setWordWrap(True)

        self.MainPageLayout.addWidget(self.Note)

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


        self.MainPageLayout.addLayout(self.ToolsButtonLayout)

        self.AnimalPoseTrackerPage.addWidget(self.MainPage)
        self.ConfigurePage = QWidget()
        self.ConfigurePage.setObjectName(u"ConfigurePage")
        self.ConfigurePageLayout = QVBoxLayout(self.ConfigurePage)
        self.ConfigurePageLayout.setObjectName(u"ConfigurePageLayout")
        self.ConfigureTabPage = QTabWidget(self.ConfigurePage)
        self.ConfigureTabPage.setObjectName(u"ConfigureTabPage")
        sizePolicy1.setHeightForWidth(self.ConfigureTabPage.sizePolicy().hasHeightForWidth())
        self.ConfigureTabPage.setSizePolicy(sizePolicy1)
        self.ConfigureTabPage.setTabBarAutoHide(False)
        self.ManageConfigure = QWidget()
        self.ManageConfigure.setObjectName(u"ManageConfigure")
        self.ManageConfigureLayout = QVBoxLayout(self.ManageConfigure)
        self.ManageConfigureLayout.setObjectName(u"ManageConfigureLayout")
        self.ConfigurePageTitle = QLabel(self.ManageConfigure)
        self.ConfigurePageTitle.setObjectName(u"ConfigurePageTitle")
        sizePolicy.setHeightForWidth(self.ConfigurePageTitle.sizePolicy().hasHeightForWidth())
        self.ConfigurePageTitle.setSizePolicy(sizePolicy)
        self.ConfigurePageTitle.setMaximumSize(QSize(236, 36))
        font3 = QFont()
        font3.setFamilies([u"Times New Roman"])
        font3.setBold(True)
        self.ConfigurePageTitle.setFont(font3)
        self.ConfigurePageTitle.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.ManageConfigureLayout.addWidget(self.ConfigurePageTitle)

        self.ConfigurePageLine = QFrame(self.ManageConfigure)
        self.ConfigurePageLine.setObjectName(u"ConfigurePageLine")
        self.ConfigurePageLine.setFrameShape(QFrame.Shape.HLine)
        self.ConfigurePageLine.setFrameShadow(QFrame.Shadow.Sunken)

        self.ManageConfigureLayout.addWidget(self.ConfigurePageLine)

        self.ConfigureFilePathLayout = QHBoxLayout()
        self.ConfigureFilePathLayout.setObjectName(u"ConfigureFilePathLayout")
        self.ConfigureFilePathLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.ConfigureFilePathLabel = QLabel(self.ManageConfigure)
        self.ConfigureFilePathLabel.setObjectName(u"ConfigureFilePathLabel")
        sizePolicy.setHeightForWidth(self.ConfigureFilePathLabel.sizePolicy().hasHeightForWidth())
        self.ConfigureFilePathLabel.setSizePolicy(sizePolicy)

        self.ConfigureFilePathLayout.addWidget(self.ConfigureFilePathLabel)

        self.ConfigureFilePathDisplay = QLineEdit(self.ManageConfigure)
        self.ConfigureFilePathDisplay.setObjectName(u"ConfigureFilePathDisplay")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.ConfigureFilePathDisplay.sizePolicy().hasHeightForWidth())
        self.ConfigureFilePathDisplay.setSizePolicy(sizePolicy4)

        self.ConfigureFilePathLayout.addWidget(self.ConfigureFilePathDisplay)

        self.ConfigureFilePathBrowser = QPushButton(self.ManageConfigure)
        self.ConfigureFilePathBrowser.setObjectName(u"ConfigureFilePathBrowser")
        sizePolicy.setHeightForWidth(self.ConfigureFilePathBrowser.sizePolicy().hasHeightForWidth())
        self.ConfigureFilePathBrowser.setSizePolicy(sizePolicy)
        icon9 = QIcon(QIcon.fromTheme(u"folder-open"))
        self.ConfigureFilePathBrowser.setIcon(icon9)

        self.ConfigureFilePathLayout.addWidget(self.ConfigureFilePathBrowser)

        self.ConfigureFileEdit = QPushButton(self.ManageConfigure)
        self.ConfigureFileEdit.setObjectName(u"ConfigureFileEdit")
        sizePolicy.setHeightForWidth(self.ConfigureFileEdit.sizePolicy().hasHeightForWidth())
        self.ConfigureFileEdit.setSizePolicy(sizePolicy)
        self.ConfigureFileEdit.setIcon(icon1)

        self.ConfigureFilePathLayout.addWidget(self.ConfigureFileEdit)

        self.ConfigureFilePathLayout.setStretch(0, 1)

        self.ManageConfigureLayout.addLayout(self.ConfigureFilePathLayout)

        self.ConfigureFileGroup = QGroupBox(self.ManageConfigure)
        self.ConfigureFileGroup.setObjectName(u"ConfigureFileGroup")
        self.ConfigureFileGroupVLayout = QVBoxLayout(self.ConfigureFileGroup)
        self.ConfigureFileGroupVLayout.setObjectName(u"ConfigureFileGroupVLayout")
        self.ConfigureFileGroupLayout = QVBoxLayout()
        self.ConfigureFileGroupLayout.setObjectName(u"ConfigureFileGroupLayout")
        self.ConfigureFile = QTreeWidget(self.ConfigureFileGroup)
        QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(self.ConfigureFile)
        __qtreewidgetitem = QTreeWidgetItem(self.ConfigureFile)
        __qtreewidgetitem1 = QTreeWidgetItem(__qtreewidgetitem)
        QTreeWidgetItem(__qtreewidgetitem1)
        __qtreewidgetitem2 = QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(__qtreewidgetitem2)
        __qtreewidgetitem3 = QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(__qtreewidgetitem3)
        __qtreewidgetitem4 = QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(__qtreewidgetitem4)
        QTreeWidgetItem(self.ConfigureFile)
        __qtreewidgetitem5 = QTreeWidgetItem(self.ConfigureFile)
        __qtreewidgetitem6 = QTreeWidgetItem(__qtreewidgetitem5)
        QTreeWidgetItem(__qtreewidgetitem6)
        QTreeWidgetItem(__qtreewidgetitem6)
        __qtreewidgetitem7 = QTreeWidgetItem(self.ConfigureFile)
        QTreeWidgetItem(__qtreewidgetitem7)
        QTreeWidgetItem(__qtreewidgetitem7)
        self.ConfigureFile.setObjectName(u"ConfigureFile")

        self.ConfigureFileGroupLayout.addWidget(self.ConfigureFile)

        self.ConfigureFileToolsLayout = QHBoxLayout()
        self.ConfigureFileToolsLayout.setObjectName(u"ConfigureFileToolsLayout")
        self.SaveConfigureFile = QPushButton(self.ConfigureFileGroup)
        self.SaveConfigureFile.setObjectName(u"SaveConfigureFile")

        self.ConfigureFileToolsLayout.addWidget(self.SaveConfigureFile)

        self.CannelConfigureFile = QPushButton(self.ConfigureFileGroup)
        self.CannelConfigureFile.setObjectName(u"CannelConfigureFile")

        self.ConfigureFileToolsLayout.addWidget(self.CannelConfigureFile)


        self.ConfigureFileGroupLayout.addLayout(self.ConfigureFileToolsLayout)


        self.ConfigureFileGroupVLayout.addLayout(self.ConfigureFileGroupLayout)


        self.ManageConfigureLayout.addWidget(self.ConfigureFileGroup)

        self.ConfigureTabPage.addTab(self.ManageConfigure, "")
        self.ExtractLabelFrames = QWidget()
        self.ExtractLabelFrames.setObjectName(u"ExtractLabelFrames")
        sizePolicy4.setHeightForWidth(self.ExtractLabelFrames.sizePolicy().hasHeightForWidth())
        self.ExtractLabelFrames.setSizePolicy(sizePolicy4)
        self.ExtractLabelFramesLayout = QVBoxLayout(self.ExtractLabelFrames)
        self.ExtractLabelFramesLayout.setObjectName(u"ExtractLabelFramesLayout")
        self.ExtractLabelFramesLayout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.ExtractLabelFrameLabel = QLabel(self.ExtractLabelFrames)
        self.ExtractLabelFrameLabel.setObjectName(u"ExtractLabelFrameLabel")
        sizePolicy.setHeightForWidth(self.ExtractLabelFrameLabel.sizePolicy().hasHeightForWidth())
        self.ExtractLabelFrameLabel.setSizePolicy(sizePolicy)
        self.ExtractLabelFrameLabel.setMaximumSize(QSize(500, 14))
        self.ExtractLabelFrameLabel.setFont(font3)
        self.ExtractLabelFrameLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.ExtractLabelFramesLayout.addWidget(self.ExtractLabelFrameLabel)

        self.ExtractLabelFrameLine = QFrame(self.ExtractLabelFrames)
        self.ExtractLabelFrameLine.setObjectName(u"ExtractLabelFrameLine")
        self.ExtractLabelFrameLine.setFrameShape(QFrame.Shape.HLine)
        self.ExtractLabelFrameLine.setFrameShadow(QFrame.Shadow.Sunken)

        self.ExtractLabelFramesLayout.addWidget(self.ExtractLabelFrameLine)

        self.ExtractFramesGroupLayout = QGroupBox(self.ExtractLabelFrames)
        self.ExtractFramesGroupLayout.setObjectName(u"ExtractFramesGroupLayout")
        sizePolicy2.setHeightForWidth(self.ExtractFramesGroupLayout.sizePolicy().hasHeightForWidth())
        self.ExtractFramesGroupLayout.setSizePolicy(sizePolicy2)
        self.ExtractFramesGroupVLayout = QVBoxLayout(self.ExtractFramesGroupLayout)
        self.ExtractFramesGroupVLayout.setObjectName(u"ExtractFramesGroupVLayout")
        self.ExtractFramesGroupVLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.ExtractFramesLabel = QLabel(self.ExtractFramesGroupLayout)
        self.ExtractFramesLabel.setObjectName(u"ExtractFramesLabel")
        sizePolicy.setHeightForWidth(self.ExtractFramesLabel.sizePolicy().hasHeightForWidth())
        self.ExtractFramesLabel.setSizePolicy(sizePolicy)

        self.ExtractFramesGroupVLayout.addWidget(self.ExtractFramesLabel)

        self.ExtractMethodLayout = QHBoxLayout()
        self.ExtractMethodLayout.setObjectName(u"ExtractMethodLayout")
        self.ExtractionMethodLabel = QLabel(self.ExtractFramesGroupLayout)
        self.ExtractionMethodLabel.setObjectName(u"ExtractionMethodLabel")
        sizePolicy.setHeightForWidth(self.ExtractionMethodLabel.sizePolicy().hasHeightForWidth())
        self.ExtractionMethodLabel.setSizePolicy(sizePolicy)

        self.ExtractMethodLayout.addWidget(self.ExtractionMethodLabel)

        self.ExtractionMethodSelection = QComboBox(self.ExtractFramesGroupLayout)
        self.ExtractionMethodSelection.addItem("")
        self.ExtractionMethodSelection.addItem("")
        self.ExtractionMethodSelection.addItem("")
        self.ExtractionMethodSelection.setObjectName(u"ExtractionMethodSelection")

        self.ExtractMethodLayout.addWidget(self.ExtractionMethodSelection)


        self.ExtractFramesGroupVLayout.addLayout(self.ExtractMethodLayout)

        self.ExtractionAlgorithmClusterStepLayout = QHBoxLayout()
        self.ExtractionAlgorithmClusterStepLayout.setObjectName(u"ExtractionAlgorithmClusterStepLayout")
        self.ExtractionAlgorithmLayout = QHBoxLayout()
        self.ExtractionAlgorithmLayout.setObjectName(u"ExtractionAlgorithmLayout")
        self.ExtractionAlgorithmLabel = QLabel(self.ExtractFramesGroupLayout)
        self.ExtractionAlgorithmLabel.setObjectName(u"ExtractionAlgorithmLabel")
        sizePolicy.setHeightForWidth(self.ExtractionAlgorithmLabel.sizePolicy().hasHeightForWidth())
        self.ExtractionAlgorithmLabel.setSizePolicy(sizePolicy)

        self.ExtractionAlgorithmLayout.addWidget(self.ExtractionAlgorithmLabel)

        self.ExtractionAlgorithmSelection = QComboBox(self.ExtractFramesGroupLayout)
        self.ExtractionAlgorithmSelection.addItem("")
        self.ExtractionAlgorithmSelection.addItem("")
        self.ExtractionAlgorithmSelection.addItem("")
        self.ExtractionAlgorithmSelection.setObjectName(u"ExtractionAlgorithmSelection")

        self.ExtractionAlgorithmLayout.addWidget(self.ExtractionAlgorithmSelection)


        self.ExtractionAlgorithmClusterStepLayout.addLayout(self.ExtractionAlgorithmLayout)

        self.ClusterStepLayout = QHBoxLayout()
        self.ClusterStepLayout.setObjectName(u"ClusterStepLayout")
        self.ClusterStepLabel = QLabel(self.ExtractFramesGroupLayout)
        self.ClusterStepLabel.setObjectName(u"ClusterStepLabel")
        sizePolicy.setHeightForWidth(self.ClusterStepLabel.sizePolicy().hasHeightForWidth())
        self.ClusterStepLabel.setSizePolicy(sizePolicy)

        self.ClusterStepLayout.addWidget(self.ClusterStepLabel)

        self.ClusterStepSetup = QSpinBox(self.ExtractFramesGroupLayout)
        self.ClusterStepSetup.setObjectName(u"ClusterStepSetup")
        sizePolicy2.setHeightForWidth(self.ClusterStepSetup.sizePolicy().hasHeightForWidth())
        self.ClusterStepSetup.setSizePolicy(sizePolicy2)
        self.ClusterStepSetup.setMinimum(1)
        self.ClusterStepSetup.setMaximum(1000)
        self.ClusterStepSetup.setValue(25)

        self.ClusterStepLayout.addWidget(self.ClusterStepSetup)


        self.ExtractionAlgorithmClusterStepLayout.addLayout(self.ClusterStepLayout)

        self.SampleIntervalLayout = QHBoxLayout()
        self.SampleIntervalLayout.setObjectName(u"SampleIntervalLayout")
        self.SampleIntervalLabel = QLabel(self.ExtractFramesGroupLayout)
        self.SampleIntervalLabel.setObjectName(u"SampleIntervalLabel")
        sizePolicy.setHeightForWidth(self.SampleIntervalLabel.sizePolicy().hasHeightForWidth())
        self.SampleIntervalLabel.setSizePolicy(sizePolicy)

        self.SampleIntervalLayout.addWidget(self.SampleIntervalLabel)

        self.SampleIntervalSetup = QSpinBox(self.ExtractFramesGroupLayout)
        self.SampleIntervalSetup.setObjectName(u"SampleIntervalSetup")
        sizePolicy2.setHeightForWidth(self.SampleIntervalSetup.sizePolicy().hasHeightForWidth())
        self.SampleIntervalSetup.setSizePolicy(sizePolicy2)
        self.SampleIntervalSetup.setMinimum(1)
        self.SampleIntervalSetup.setMaximum(1000)

        self.SampleIntervalLayout.addWidget(self.SampleIntervalSetup)


        self.ExtractionAlgorithmClusterStepLayout.addLayout(self.SampleIntervalLayout)


        self.ExtractFramesGroupVLayout.addLayout(self.ExtractionAlgorithmClusterStepLayout)

        self.SelectionVideosImagesLayout = QHBoxLayout()
        self.SelectionVideosImagesLayout.setObjectName(u"SelectionVideosImagesLayout")
        self.SelectionVideosImages = QPushButton(self.ExtractFramesGroupLayout)
        self.SelectionVideosImages.setObjectName(u"SelectionVideosImages")
        sizePolicy2.setHeightForWidth(self.SelectionVideosImages.sizePolicy().hasHeightForWidth())
        self.SelectionVideosImages.setSizePolicy(sizePolicy2)

        self.SelectionVideosImagesLayout.addWidget(self.SelectionVideosImages)

        self.SelectionVideosImagesLabel = QLabel(self.ExtractFramesGroupLayout)
        self.SelectionVideosImagesLabel.setObjectName(u"SelectionVideosImagesLabel")
        sizePolicy2.setHeightForWidth(self.SelectionVideosImagesLabel.sizePolicy().hasHeightForWidth())
        self.SelectionVideosImagesLabel.setSizePolicy(sizePolicy2)

        self.SelectionVideosImagesLayout.addWidget(self.SelectionVideosImagesLabel)


        self.ExtractFramesGroupVLayout.addLayout(self.SelectionVideosImagesLayout)

        self.ClearVideosImagesLayout = QHBoxLayout()
        self.ClearVideosImagesLayout.setObjectName(u"ClearVideosImagesLayout")
        self.ClearVideosImages = QPushButton(self.ExtractFramesGroupLayout)
        self.ClearVideosImages.setObjectName(u"ClearVideosImages")
        sizePolicy2.setHeightForWidth(self.ClearVideosImages.sizePolicy().hasHeightForWidth())
        self.ClearVideosImages.setSizePolicy(sizePolicy2)

        self.ClearVideosImagesLayout.addWidget(self.ClearVideosImages)

        self.ExtracFrames = QPushButton(self.ExtractFramesGroupLayout)
        self.ExtracFrames.setObjectName(u"ExtracFrames")
        sizePolicy2.setHeightForWidth(self.ExtracFrames.sizePolicy().hasHeightForWidth())
        self.ExtracFrames.setSizePolicy(sizePolicy2)

        self.ClearVideosImagesLayout.addWidget(self.ExtracFrames)


        self.ExtractFramesGroupVLayout.addLayout(self.ClearVideosImagesLayout)


        self.ExtractLabelFramesLayout.addWidget(self.ExtractFramesGroupLayout)

        self.LabelFramesGroupLayout = QGroupBox(self.ExtractLabelFrames)
        self.LabelFramesGroupLayout.setObjectName(u"LabelFramesGroupLayout")
        sizePolicy2.setHeightForWidth(self.LabelFramesGroupLayout.sizePolicy().hasHeightForWidth())
        self.LabelFramesGroupLayout.setSizePolicy(sizePolicy2)
        self.LabelFramesGroupVLayout = QVBoxLayout(self.LabelFramesGroupLayout)
        self.LabelFramesGroupVLayout.setObjectName(u"LabelFramesGroupVLayout")
        self.LabelFramesGroupVLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.LabelFrameLayout = QVBoxLayout()
        self.LabelFrameLayout.setObjectName(u"LabelFrameLayout")
        self.LabelFramesLabel = QLabel(self.LabelFramesGroupLayout)
        self.LabelFramesLabel.setObjectName(u"LabelFramesLabel")
        sizePolicy.setHeightForWidth(self.LabelFramesLabel.sizePolicy().hasHeightForWidth())
        self.LabelFramesLabel.setSizePolicy(sizePolicy)

        self.LabelFrameLayout.addWidget(self.LabelFramesLabel)

        self.SaveLayout = QHBoxLayout()
        self.SaveLayout.setObjectName(u"SaveLayout")
        self.SaveYOLO = QCheckBox(self.LabelFramesGroupLayout)
        self.SaveYOLO.setObjectName(u"SaveYOLO")
        sizePolicy2.setHeightForWidth(self.SaveYOLO.sizePolicy().hasHeightForWidth())
        self.SaveYOLO.setSizePolicy(sizePolicy2)

        self.SaveLayout.addWidget(self.SaveYOLO)

        self.SaveCOCO = QCheckBox(self.LabelFramesGroupLayout)
        self.SaveCOCO.setObjectName(u"SaveCOCO")
        sizePolicy2.setHeightForWidth(self.SaveCOCO.sizePolicy().hasHeightForWidth())
        self.SaveCOCO.setSizePolicy(sizePolicy2)

        self.SaveLayout.addWidget(self.SaveCOCO)


        self.LabelFrameLayout.addLayout(self.SaveLayout)

        self.StartLabelFrames = QPushButton(self.LabelFramesGroupLayout)
        self.StartLabelFrames.setObjectName(u"StartLabelFrames")
        sizePolicy2.setHeightForWidth(self.StartLabelFrames.sizePolicy().hasHeightForWidth())
        self.StartLabelFrames.setSizePolicy(sizePolicy2)

        self.LabelFrameLayout.addWidget(self.StartLabelFrames)

        self.StartCreateDatasets = QPushButton(self.LabelFramesGroupLayout)
        self.StartCreateDatasets.setObjectName(u"StartCreateDatasets")

        self.LabelFrameLayout.addWidget(self.StartCreateDatasets)

        self.CheckDatasets = QPushButton(self.LabelFramesGroupLayout)
        self.CheckDatasets.setObjectName(u"CheckDatasets")
        sizePolicy2.setHeightForWidth(self.CheckDatasets.sizePolicy().hasHeightForWidth())
        self.CheckDatasets.setSizePolicy(sizePolicy2)

        self.LabelFrameLayout.addWidget(self.CheckDatasets)


        self.LabelFramesGroupVLayout.addLayout(self.LabelFrameLayout)


        self.ExtractLabelFramesLayout.addWidget(self.LabelFramesGroupLayout)

        self.ConfigureTabPage.addTab(self.ExtractLabelFrames, "")
        self.TrainingConfigure = QWidget()
        self.TrainingConfigure.setObjectName(u"TrainingConfigure")
        self.TrainingConfigureLayout = QVBoxLayout(self.TrainingConfigure)
        self.TrainingConfigureLayout.setObjectName(u"TrainingConfigureLayout")
        self.TrainingConfigureLabel = QLabel(self.TrainingConfigure)
        self.TrainingConfigureLabel.setObjectName(u"TrainingConfigureLabel")
        self.TrainingConfigureLabel.setFont(font3)
        self.TrainingConfigureLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.TrainingConfigureLayout.addWidget(self.TrainingConfigureLabel)

        self.TrainingConfigureLine = QFrame(self.TrainingConfigure)
        self.TrainingConfigureLine.setObjectName(u"TrainingConfigureLine")
        self.TrainingConfigureLine.setFrameShape(QFrame.Shape.HLine)
        self.TrainingConfigureLine.setFrameShadow(QFrame.Shadow.Sunken)

        self.TrainingConfigureLayout.addWidget(self.TrainingConfigureLine)

        self.TrainingConfigureGroup = QGroupBox(self.TrainingConfigure)
        self.TrainingConfigureGroup.setObjectName(u"TrainingConfigureGroup")
        self.TrainingConfigureGroupVLayout = QVBoxLayout(self.TrainingConfigureGroup)
        self.TrainingConfigureGroupVLayout.setObjectName(u"TrainingConfigureGroupVLayout")
        self.TrainingConfigureGroupLayout = QVBoxLayout()
        self.TrainingConfigureGroupLayout.setObjectName(u"TrainingConfigureGroupLayout")
        self.TrainingConfigureToolsLayout = QHBoxLayout()
        self.TrainingConfigureToolsLayout.setObjectName(u"TrainingConfigureToolsLayout")
        self.EditTrainingParameters = QPushButton(self.TrainingConfigureGroup)
        self.EditTrainingParameters.setObjectName(u"EditTrainingParameters")

        self.TrainingConfigureToolsLayout.addWidget(self.EditTrainingParameters)

        self.StartTrain = QPushButton(self.TrainingConfigureGroup)
        self.StartTrain.setObjectName(u"StartTrain")

        self.TrainingConfigureToolsLayout.addWidget(self.StartTrain)

        self.EndTrain = QPushButton(self.TrainingConfigureGroup)
        self.EndTrain.setObjectName(u"EndTrain")

        self.TrainingConfigureToolsLayout.addWidget(self.EndTrain)

        self.ResumeTrain = QPushButton(self.TrainingConfigureGroup)
        self.ResumeTrain.setObjectName(u"ResumeTrain")

        self.TrainingConfigureToolsLayout.addWidget(self.ResumeTrain)


        self.TrainingConfigureGroupLayout.addLayout(self.TrainingConfigureToolsLayout)

        self.TrainingConfigureEdit = QTreeWidget(self.TrainingConfigureGroup)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        QTreeWidgetItem(self.TrainingConfigureEdit)
        self.TrainingConfigureEdit.setObjectName(u"TrainingConfigureEdit")

        self.TrainingConfigureGroupLayout.addWidget(self.TrainingConfigureEdit)


        self.TrainingConfigureGroupVLayout.addLayout(self.TrainingConfigureGroupLayout)


        self.TrainingConfigureLayout.addWidget(self.TrainingConfigureGroup)

        self.ConfigureTabPage.addTab(self.TrainingConfigure, "")
        self.EvaluateModel = QWidget()
        self.EvaluateModel.setObjectName(u"EvaluateModel")
        self.EvaluateModelLayout = QVBoxLayout(self.EvaluateModel)
        self.EvaluateModelLayout.setObjectName(u"EvaluateModelLayout")
        self.EvaluateModelLabel = QLabel(self.EvaluateModel)
        self.EvaluateModelLabel.setObjectName(u"EvaluateModelLabel")
        self.EvaluateModelLabel.setFont(font3)
        self.EvaluateModelLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.EvaluateModelLayout.addWidget(self.EvaluateModelLabel)

        self.EvaluateModelLine = QFrame(self.EvaluateModel)
        self.EvaluateModelLine.setObjectName(u"EvaluateModelLine")
        self.EvaluateModelLine.setFrameShape(QFrame.Shape.HLine)
        self.EvaluateModelLine.setFrameShadow(QFrame.Shadow.Sunken)

        self.EvaluateModelLayout.addWidget(self.EvaluateModelLine)

        self.EvaluateModelGroup = QGroupBox(self.EvaluateModel)
        self.EvaluateModelGroup.setObjectName(u"EvaluateModelGroup")
        self.EvaluateModelGroupVLayout = QVBoxLayout(self.EvaluateModelGroup)
        self.EvaluateModelGroupVLayout.setObjectName(u"EvaluateModelGroupVLayout")
        self.EvaluateModelGroupVLayout.setContentsMargins(9, 9, 9, 9)
        self.EvaluateModelGroupLayout = QVBoxLayout()
        self.EvaluateModelGroupLayout.setObjectName(u"EvaluateModelGroupLayout")
        self.EvaluateModelToolsLayout = QHBoxLayout()
        self.EvaluateModelToolsLayout.setObjectName(u"EvaluateModelToolsLayout")
        self.EditEvaluationParameters = QPushButton(self.EvaluateModelGroup)
        self.EditEvaluationParameters.setObjectName(u"EditEvaluationParameters")

        self.EvaluateModelToolsLayout.addWidget(self.EditEvaluationParameters)

        self.StartEvaluate = QPushButton(self.EvaluateModelGroup)
        self.StartEvaluate.setObjectName(u"StartEvaluate")

        self.EvaluateModelToolsLayout.addWidget(self.StartEvaluate)

        self.EndEvaluate = QPushButton(self.EvaluateModelGroup)
        self.EndEvaluate.setObjectName(u"EndEvaluate")

        self.EvaluateModelToolsLayout.addWidget(self.EndEvaluate)


        self.EvaluateModelGroupLayout.addLayout(self.EvaluateModelToolsLayout)

        self.EvaluateConfigure = QTreeWidget(self.EvaluateModelGroup)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        QTreeWidgetItem(self.EvaluateConfigure)
        self.EvaluateConfigure.setObjectName(u"EvaluateConfigure")

        self.EvaluateModelGroupLayout.addWidget(self.EvaluateConfigure)


        self.EvaluateModelGroupVLayout.addLayout(self.EvaluateModelGroupLayout)


        self.EvaluateModelLayout.addWidget(self.EvaluateModelGroup)

        self.ConfigureTabPage.addTab(self.EvaluateModel, "")
        self.InferenceSource = QWidget()
        self.InferenceSource.setObjectName(u"InferenceSource")
        self.InferenceSourceLayout = QVBoxLayout(self.InferenceSource)
        self.InferenceSourceLayout.setObjectName(u"InferenceSourceLayout")
        self.InferenceSourceLabel = QLabel(self.InferenceSource)
        self.InferenceSourceLabel.setObjectName(u"InferenceSourceLabel")
        self.InferenceSourceLabel.setFont(font3)
        self.InferenceSourceLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.InferenceSourceLayout.addWidget(self.InferenceSourceLabel)

        self.InferenceSourceLine = QFrame(self.InferenceSource)
        self.InferenceSourceLine.setObjectName(u"InferenceSourceLine")
        self.InferenceSourceLine.setFrameShape(QFrame.Shape.HLine)
        self.InferenceSourceLine.setFrameShadow(QFrame.Shadow.Sunken)

        self.InferenceSourceLayout.addWidget(self.InferenceSourceLine)

        self.InferenceSourceGroup = QGroupBox(self.InferenceSource)
        self.InferenceSourceGroup.setObjectName(u"InferenceSourceGroup")
        self.InferenceSourceGroupLayout = QVBoxLayout(self.InferenceSourceGroup)
        self.InferenceSourceGroupLayout.setObjectName(u"InferenceSourceGroupLayout")
        self.InferenceConfigureGroupLayout = QVBoxLayout()
        self.InferenceConfigureGroupLayout.setObjectName(u"InferenceConfigureGroupLayout")
        self.InferenceSourceToolsLayout = QHBoxLayout()
        self.InferenceSourceToolsLayout.setObjectName(u"InferenceSourceToolsLayout")
        self.EditInferenceParameters = QPushButton(self.InferenceSourceGroup)
        self.EditInferenceParameters.setObjectName(u"EditInferenceParameters")

        self.InferenceSourceToolsLayout.addWidget(self.EditInferenceParameters)

        self.SelectionSource = QPushButton(self.InferenceSourceGroup)
        self.SelectionSource.setObjectName(u"SelectionSource")

        self.InferenceSourceToolsLayout.addWidget(self.SelectionSource)

        self.StartInference = QPushButton(self.InferenceSourceGroup)
        self.StartInference.setObjectName(u"StartInference")

        self.InferenceSourceToolsLayout.addWidget(self.StartInference)

        self.EndInference = QPushButton(self.InferenceSourceGroup)
        self.EndInference.setObjectName(u"EndInference")

        self.InferenceSourceToolsLayout.addWidget(self.EndInference)


        self.InferenceConfigureGroupLayout.addLayout(self.InferenceSourceToolsLayout)

        self.InferenceConfigure = QTreeWidget(self.InferenceSourceGroup)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        QTreeWidgetItem(self.InferenceConfigure)
        self.InferenceConfigure.setObjectName(u"InferenceConfigure")

        self.InferenceConfigureGroupLayout.addWidget(self.InferenceConfigure)


        self.InferenceSourceGroupLayout.addLayout(self.InferenceConfigureGroupLayout)


        self.InferenceSourceLayout.addWidget(self.InferenceSourceGroup)

        self.ConfigureTabPage.addTab(self.InferenceSource, "")
        self.ExportModel = QWidget()
        self.ExportModel.setObjectName(u"ExportModel")
        self.ExportModelLayout = QVBoxLayout(self.ExportModel)
        self.ExportModelLayout.setObjectName(u"ExportModelLayout")
        self.ExportModelLabel = QLabel(self.ExportModel)
        self.ExportModelLabel.setObjectName(u"ExportModelLabel")
        self.ExportModelLabel.setFont(font3)
        self.ExportModelLabel.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.ExportModelLayout.addWidget(self.ExportModelLabel)

        self.ExportModelLine = QFrame(self.ExportModel)
        self.ExportModelLine.setObjectName(u"ExportModelLine")
        self.ExportModelLine.setFrameShape(QFrame.Shape.HLine)
        self.ExportModelLine.setFrameShadow(QFrame.Shadow.Sunken)

        self.ExportModelLayout.addWidget(self.ExportModelLine)

        self.ExportConfigureGroup = QGroupBox(self.ExportModel)
        self.ExportConfigureGroup.setObjectName(u"ExportConfigureGroup")
        self.ExportConfigureGroupVLayout = QVBoxLayout(self.ExportConfigureGroup)
        self.ExportConfigureGroupVLayout.setObjectName(u"ExportConfigureGroupVLayout")
        self.ExportConfigureGroupLayout = QVBoxLayout()
        self.ExportConfigureGroupLayout.setObjectName(u"ExportConfigureGroupLayout")
        self.ExportConfigureToolsLayout = QHBoxLayout()
        self.ExportConfigureToolsLayout.setObjectName(u"ExportConfigureToolsLayout")
        self.EditExportParameters = QPushButton(self.ExportConfigureGroup)
        self.EditExportParameters.setObjectName(u"EditExportParameters")

        self.ExportConfigureToolsLayout.addWidget(self.EditExportParameters)

        self.StartModelWeights = QPushButton(self.ExportConfigureGroup)
        self.StartModelWeights.setObjectName(u"StartModelWeights")

        self.ExportConfigureToolsLayout.addWidget(self.StartModelWeights)

        self.StartExport = QPushButton(self.ExportConfigureGroup)
        self.StartExport.setObjectName(u"StartExport")

        self.ExportConfigureToolsLayout.addWidget(self.StartExport)


        self.ExportConfigureGroupLayout.addLayout(self.ExportConfigureToolsLayout)

        self.ExportConfigure = QTreeWidget(self.ExportConfigureGroup)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        QTreeWidgetItem(self.ExportConfigure)
        self.ExportConfigure.setObjectName(u"ExportConfigure")

        self.ExportConfigureGroupLayout.addWidget(self.ExportConfigure)


        self.ExportConfigureGroupVLayout.addLayout(self.ExportConfigureGroupLayout)


        self.ExportModelLayout.addWidget(self.ExportConfigureGroup)

        self.ConfigureTabPage.addTab(self.ExportModel, "")

        self.ConfigurePageLayout.addWidget(self.ConfigureTabPage)

        self.AnimalPoseTrackerPage.addWidget(self.ConfigurePage)

        self.AnimalPoseTrackerVLayout.addWidget(self.AnimalPoseTrackerPage)

        AnimalPoseTracker.setCentralWidget(self.AnimalPoseTrackerLayout)
        self.menubar = QMenuBar(AnimalPoseTracker)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 33))
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
        self.menuAnimalPoseTracker.addAction(self.actionCreateNewProject)
        self.menuAnimalPoseTracker.addAction(self.actionLoadProject)
        self.menuAnimalPoseTracker.addAction(self.actionOpenRecent)
        self.menuAnimalPoseTracker.addAction(self.actionSave)
        self.menuAnimalPoseTracker.addAction(self.actionExit)
        self.menuTools.addAction(self.actionAnnotator)
        self.menuTools.addAction(self.actionInferencer)
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

        self.AnimalPoseTrackerPage.setCurrentIndex(0)
        self.ConfigureTabPage.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(AnimalPoseTracker)
    # setupUi

    def retranslateUi(self, AnimalPoseTracker):
        AnimalPoseTracker.setWindowTitle(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker", None))
        self.actionLoadProject.setText(QCoreApplication.translate("AnimalPoseTracker", u"Open Old Project Ctrl+O", None))
        self.actionOpenRecent.setText(QCoreApplication.translate("AnimalPoseTracker", u"Open Recent", None))
        self.actionSave.setText(QCoreApplication.translate("AnimalPoseTracker", u"Save", None))
        self.actionExit.setText(QCoreApplication.translate("AnimalPoseTracker", u"Exit", None))
        self.actionAnnotator.setText(QCoreApplication.translate("AnimalPoseTracker", u"Annotator", None))
        self.actionInferencer.setText(QCoreApplication.translate("AnimalPoseTracker", u"Inferencer", None))
        self.actionDoc.setText(QCoreApplication.translate("AnimalPoseTracker", u"Doc", None))
        self.actionCheckUpdated.setText(QCoreApplication.translate("AnimalPoseTracker", u"Check for Updating", None))
        self.actionHelp.setText(QCoreApplication.translate("AnimalPoseTracker", u"Help", None))
        self.actionPublicDatasetsProject.setText(QCoreApplication.translate("AnimalPoseTracker", u"Public Datasets Project", None))
        self.actionDark.setText(QCoreApplication.translate("AnimalPoseTracker", u"Dark", None))
        self.actionLight.setText(QCoreApplication.translate("AnimalPoseTracker", u"Light", None))
        self.actionCreateNewProject.setText(QCoreApplication.translate("AnimalPoseTracker", u"New Project Ctrl+N", None))
        self.WelcomeLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"Welcome to the AnimalPoseTracker!", None))
        self.Logo.setText("")
        self.Introdution.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker is an open source tool for animal pose estimation with deep learning. Wu X. | https://wux024.github.io", None))
        self.Note.setText(QCoreApplication.translate("AnimalPoseTracker", u"To get started, you can create a new project, load an existing project, or use opensource public datasets.", None))
        self.CreateNewProjectButton.setText(QCoreApplication.translate("AnimalPoseTracker", u"Create New Project", None))
        self.LoadProjectButton.setText(QCoreApplication.translate("AnimalPoseTracker", u"Load Project", None))
        self.PublicDatasetsProjectButton.setText(QCoreApplication.translate("AnimalPoseTracker", u"Public Datasets Project", None))
        self.ConfigurePageTitle.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker-Manage Configure", None))
        self.ConfigureFilePathLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"Configure File Path:", None))
        self.ConfigureFilePathBrowser.setText("")
        self.ConfigureFileEdit.setText("")
        self.ConfigureFileGroup.setTitle("")
        ___qtreewidgetitem = self.ConfigureFile.headerItem()
        ___qtreewidgetitem.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"Value", None));
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"Parameter", None));

        __sortingEnabled = self.ConfigureFile.isSortingEnabled()
        self.ConfigureFile.setSortingEnabled(False)
        ___qtreewidgetitem1 = self.ConfigureFile.topLevelItem(0)
        ___qtreewidgetitem1.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"mouse", None));
        ___qtreewidgetitem1.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"experiment", None));
        ___qtreewidgetitem2 = self.ConfigureFile.topLevelItem(1)
        ___qtreewidgetitem2.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"adam", None));
        ___qtreewidgetitem2.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"experimenter", None));
        ___qtreewidgetitem3 = self.ConfigureFile.topLevelItem(2)
        ___qtreewidgetitem3.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"20250406", None));
        ___qtreewidgetitem3.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"date", None));
        ___qtreewidgetitem4 = self.ConfigureFile.topLevelItem(3)
        ___qtreewidgetitem4.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"project_path", None));
        ___qtreewidgetitem5 = self.ConfigureFile.topLevelItem(4)
        ___qtreewidgetitem5.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"model_type", None));
        ___qtreewidgetitem6 = self.ConfigureFile.topLevelItem(5)
        ___qtreewidgetitem6.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"model_scale", None));
        ___qtreewidgetitem7 = self.ConfigureFile.topLevelItem(6)
        ___qtreewidgetitem7.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"10", None));
        ___qtreewidgetitem7.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"videos", None));
        ___qtreewidgetitem8 = ___qtreewidgetitem7.child(0)
        ___qtreewidgetitem8.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"path1", None));
        ___qtreewidgetitem9 = ___qtreewidgetitem8.child(0)
        ___qtreewidgetitem9.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"resolution", None));
        ___qtreewidgetitem10 = self.ConfigureFile.topLevelItem(7)
        ___qtreewidgetitem10.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"10", None));
        ___qtreewidgetitem10.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"images", None));
        ___qtreewidgetitem11 = ___qtreewidgetitem10.child(0)
        ___qtreewidgetitem11.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"resolution", None));
        ___qtreewidgetitem12 = self.ConfigureFile.topLevelItem(8)
        ___qtreewidgetitem12.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"12", None));
        ___qtreewidgetitem12.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"keypoints", None));
        ___qtreewidgetitem13 = ___qtreewidgetitem12.child(0)
        ___qtreewidgetitem13.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"nose", None));
        ___qtreewidgetitem13.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"0", None));
        ___qtreewidgetitem14 = self.ConfigureFile.topLevelItem(9)
        ___qtreewidgetitem14.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"3", None));
        ___qtreewidgetitem14.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"classes", None));
        ___qtreewidgetitem15 = ___qtreewidgetitem14.child(0)
        ___qtreewidgetitem15.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"mouse", None));
        ___qtreewidgetitem15.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"0", None));
        ___qtreewidgetitem16 = self.ConfigureFile.topLevelItem(10)
        ___qtreewidgetitem16.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"1", None));
        ___qtreewidgetitem16.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"visible", None));
        ___qtreewidgetitem17 = self.ConfigureFile.topLevelItem(11)
        ___qtreewidgetitem17.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"skeleton", None));
        ___qtreewidgetitem18 = ___qtreewidgetitem17.child(0)
        ___qtreewidgetitem18.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"0", None));
        ___qtreewidgetitem19 = ___qtreewidgetitem18.child(0)
        ___qtreewidgetitem19.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"nose", None));
        ___qtreewidgetitem19.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"0", None));
        ___qtreewidgetitem20 = ___qtreewidgetitem18.child(1)
        ___qtreewidgetitem20.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"left_eye", None));
        ___qtreewidgetitem20.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"1", None));
        ___qtreewidgetitem21 = self.ConfigureFile.topLevelItem(12)
        ___qtreewidgetitem21.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"flip", None));
        ___qtreewidgetitem22 = ___qtreewidgetitem21.child(0)
        ___qtreewidgetitem22.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"1", None));
        ___qtreewidgetitem22.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"0", None));
        ___qtreewidgetitem23 = ___qtreewidgetitem21.child(1)
        ___qtreewidgetitem23.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0", None));
        ___qtreewidgetitem23.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"1", None));
        self.ConfigureFile.setSortingEnabled(__sortingEnabled)

        self.SaveConfigureFile.setText(QCoreApplication.translate("AnimalPoseTracker", u"Save", None))
        self.CannelConfigureFile.setText(QCoreApplication.translate("AnimalPoseTracker", u"Cannel", None))
        self.ConfigureTabPage.setTabText(self.ConfigureTabPage.indexOf(self.ManageConfigure), QCoreApplication.translate("AnimalPoseTracker", u"Manage Configure", None))
        self.ExtractLabelFrameLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker-Extract and Label Frames", None))
        self.ExtractFramesGroupLayout.setTitle("")
        self.ExtractFramesLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"Extract Frames", None))
        self.ExtractionMethodLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"Extraction method", None))
        self.ExtractionMethodSelection.setItemText(0, QCoreApplication.translate("AnimalPoseTracker", u"auto", None))
        self.ExtractionMethodSelection.setItemText(1, QCoreApplication.translate("AnimalPoseTracker", u"manual", None))
        self.ExtractionMethodSelection.setItemText(2, QCoreApplication.translate("AnimalPoseTracker", u"all", None))

        self.ExtractionAlgorithmLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"Extraction algorithm", None))
        self.ExtractionAlgorithmSelection.setItemText(0, QCoreApplication.translate("AnimalPoseTracker", u"kmeas", None))
        self.ExtractionAlgorithmSelection.setItemText(1, QCoreApplication.translate("AnimalPoseTracker", u"uniform", None))
        self.ExtractionAlgorithmSelection.setItemText(2, QCoreApplication.translate("AnimalPoseTracker", u"None", None))

        self.ClusterStepLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"Cluster step", None))
        self.SampleIntervalLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"Sample Interval", None))
        self.SelectionVideosImages.setText(QCoreApplication.translate("AnimalPoseTracker", u"Selection videos or images", None))
        self.SelectionVideosImagesLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"50 videos or images selected", None))
        self.ClearVideosImages.setText(QCoreApplication.translate("AnimalPoseTracker", u"Clear selection", None))
        self.ExtracFrames.setText(QCoreApplication.translate("AnimalPoseTracker", u"Extract Frames", None))
        self.LabelFramesGroupLayout.setTitle("")
        self.LabelFramesLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"Label Frames", None))
        self.SaveYOLO.setText(QCoreApplication.translate("AnimalPoseTracker", u"Save YOLO Format", None))
        self.SaveCOCO.setText(QCoreApplication.translate("AnimalPoseTracker", u"Save COCO Format", None))
        self.StartLabelFrames.setText(QCoreApplication.translate("AnimalPoseTracker", u"Start Label Frames", None))
        self.StartCreateDatasets.setText(QCoreApplication.translate("AnimalPoseTracker", u"Start Create Datasets", None))
        self.CheckDatasets.setText(QCoreApplication.translate("AnimalPoseTracker", u"Check Datasets", None))
        self.ConfigureTabPage.setTabText(self.ConfigureTabPage.indexOf(self.ExtractLabelFrames), QCoreApplication.translate("AnimalPoseTracker", u"Extract and Label Frames", None))
        self.TrainingConfigureLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker-Training Configure", None))
        self.TrainingConfigureGroup.setTitle("")
        self.EditTrainingParameters.setText(QCoreApplication.translate("AnimalPoseTracker", u"Edit training parameters", None))
        self.StartTrain.setText(QCoreApplication.translate("AnimalPoseTracker", u"Start Train", None))
        self.EndTrain.setText(QCoreApplication.translate("AnimalPoseTracker", u"End Train", None))
        self.ResumeTrain.setText(QCoreApplication.translate("AnimalPoseTracker", u"Resume Train", None))
        ___qtreewidgetitem24 = self.TrainingConfigureEdit.headerItem()
        ___qtreewidgetitem24.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"Value", None));
        ___qtreewidgetitem24.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"Parameter", None));

        __sortingEnabled1 = self.TrainingConfigureEdit.isSortingEnabled()
        self.TrainingConfigureEdit.setSortingEnabled(False)
        ___qtreewidgetitem25 = self.TrainingConfigureEdit.topLevelItem(0)
        ___qtreewidgetitem25.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"model", None));
        ___qtreewidgetitem26 = self.TrainingConfigureEdit.topLevelItem(1)
        ___qtreewidgetitem26.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"data", None));
        ___qtreewidgetitem27 = self.TrainingConfigureEdit.topLevelItem(2)
        ___qtreewidgetitem27.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"1000", None));
        ___qtreewidgetitem27.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"epochs", None));
        ___qtreewidgetitem28 = self.TrainingConfigureEdit.topLevelItem(3)
        ___qtreewidgetitem28.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"None", None));
        ___qtreewidgetitem28.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"time", None));
        ___qtreewidgetitem29 = self.TrainingConfigureEdit.topLevelItem(4)
        ___qtreewidgetitem29.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"300", None));
        ___qtreewidgetitem29.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"patience", None));
        ___qtreewidgetitem30 = self.TrainingConfigureEdit.topLevelItem(5)
        ___qtreewidgetitem30.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"16", None));
        ___qtreewidgetitem30.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"batch", None));
        ___qtreewidgetitem31 = self.TrainingConfigureEdit.topLevelItem(6)
        ___qtreewidgetitem31.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"640", None));
        ___qtreewidgetitem31.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"imgsz", None));
        ___qtreewidgetitem32 = self.TrainingConfigureEdit.topLevelItem(7)
        ___qtreewidgetitem32.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"True", None));
        ___qtreewidgetitem32.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save", None));
        ___qtreewidgetitem33 = self.TrainingConfigureEdit.topLevelItem(8)
        ___qtreewidgetitem33.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"-1", None));
        ___qtreewidgetitem33.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_period", None));
        ___qtreewidgetitem34 = self.TrainingConfigureEdit.topLevelItem(9)
        ___qtreewidgetitem34.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem34.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"cache", None));
        ___qtreewidgetitem35 = self.TrainingConfigureEdit.topLevelItem(10)
        ___qtreewidgetitem35.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"None", None));
        ___qtreewidgetitem35.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"device", None));
        ___qtreewidgetitem36 = self.TrainingConfigureEdit.topLevelItem(11)
        ___qtreewidgetitem36.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"8", None));
        ___qtreewidgetitem36.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"workers", None));
        ___qtreewidgetitem37 = self.TrainingConfigureEdit.topLevelItem(12)
        ___qtreewidgetitem37.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"None", None));
        ___qtreewidgetitem37.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"project", None));
        ___qtreewidgetitem38 = self.TrainingConfigureEdit.topLevelItem(13)
        ___qtreewidgetitem38.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"None", None));
        ___qtreewidgetitem38.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"name", None));
        ___qtreewidgetitem39 = self.TrainingConfigureEdit.topLevelItem(14)
        ___qtreewidgetitem39.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"True", None));
        ___qtreewidgetitem39.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"pretrained", None));
        ___qtreewidgetitem40 = self.TrainingConfigureEdit.topLevelItem(15)
        ___qtreewidgetitem40.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"auto", None));
        ___qtreewidgetitem40.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"optimizer", None));
        ___qtreewidgetitem41 = self.TrainingConfigureEdit.topLevelItem(16)
        ___qtreewidgetitem41.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"date", None));
        ___qtreewidgetitem41.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"seed", None));
        ___qtreewidgetitem42 = self.TrainingConfigureEdit.topLevelItem(17)
        ___qtreewidgetitem42.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem42.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"cos_lr", None));
        ___qtreewidgetitem43 = self.TrainingConfigureEdit.topLevelItem(18)
        ___qtreewidgetitem43.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"10", None));
        ___qtreewidgetitem43.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"close_mosaic", None));
        ___qtreewidgetitem44 = self.TrainingConfigureEdit.topLevelItem(19)
        ___qtreewidgetitem44.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem44.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"resume", None));
        ___qtreewidgetitem45 = self.TrainingConfigureEdit.topLevelItem(20)
        ___qtreewidgetitem45.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"True", None));
        ___qtreewidgetitem45.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"amp", None));
        ___qtreewidgetitem46 = self.TrainingConfigureEdit.topLevelItem(21)
        ___qtreewidgetitem46.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"1.0", None));
        ___qtreewidgetitem46.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"fraction", None));
        ___qtreewidgetitem47 = self.TrainingConfigureEdit.topLevelItem(22)
        ___qtreewidgetitem47.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"false", None));
        ___qtreewidgetitem47.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"profile", None));
        ___qtreewidgetitem48 = self.TrainingConfigureEdit.topLevelItem(23)
        ___qtreewidgetitem48.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.01", None));
        ___qtreewidgetitem48.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"lr0", None));
        ___qtreewidgetitem49 = self.TrainingConfigureEdit.topLevelItem(24)
        ___qtreewidgetitem49.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.01", None));
        ___qtreewidgetitem49.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"lrf", None));
        ___qtreewidgetitem50 = self.TrainingConfigureEdit.topLevelItem(25)
        ___qtreewidgetitem50.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.937", None));
        ___qtreewidgetitem50.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"momentum", None));
        ___qtreewidgetitem51 = self.TrainingConfigureEdit.topLevelItem(26)
        ___qtreewidgetitem51.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.0005", None));
        ___qtreewidgetitem51.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"weight_decay", None));
        ___qtreewidgetitem52 = self.TrainingConfigureEdit.topLevelItem(27)
        ___qtreewidgetitem52.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"3", None));
        ___qtreewidgetitem52.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"warmup_epochs", None));
        ___qtreewidgetitem53 = self.TrainingConfigureEdit.topLevelItem(28)
        ___qtreewidgetitem53.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.8", None));
        ___qtreewidgetitem53.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"warmup_momentum", None));
        ___qtreewidgetitem54 = self.TrainingConfigureEdit.topLevelItem(29)
        ___qtreewidgetitem54.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.1", None));
        ___qtreewidgetitem54.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"warmup_bias_lr", None));
        ___qtreewidgetitem55 = self.TrainingConfigureEdit.topLevelItem(30)
        ___qtreewidgetitem55.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"7.5", None));
        ___qtreewidgetitem55.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"box", None));
        ___qtreewidgetitem56 = self.TrainingConfigureEdit.topLevelItem(31)
        ___qtreewidgetitem56.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.5", None));
        ___qtreewidgetitem56.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"cls", None));
        ___qtreewidgetitem57 = self.TrainingConfigureEdit.topLevelItem(32)
        ___qtreewidgetitem57.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"1.5", None));
        ___qtreewidgetitem57.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"dfl", None));
        ___qtreewidgetitem58 = self.TrainingConfigureEdit.topLevelItem(33)
        ___qtreewidgetitem58.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"40.0", None));
        ___qtreewidgetitem58.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"pose", None));
        ___qtreewidgetitem59 = self.TrainingConfigureEdit.topLevelItem(34)
        ___qtreewidgetitem59.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"2.0", None));
        ___qtreewidgetitem59.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"kobj", None));
        ___qtreewidgetitem60 = self.TrainingConfigureEdit.topLevelItem(35)
        ___qtreewidgetitem60.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"64", None));
        ___qtreewidgetitem60.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"nbs", None));
        ___qtreewidgetitem61 = self.TrainingConfigureEdit.topLevelItem(36)
        ___qtreewidgetitem61.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"True", None));
        ___qtreewidgetitem61.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"overlap_mask", None));
        ___qtreewidgetitem62 = self.TrainingConfigureEdit.topLevelItem(37)
        ___qtreewidgetitem62.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"4", None));
        ___qtreewidgetitem62.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"mask_ratio", None));
        ___qtreewidgetitem63 = self.TrainingConfigureEdit.topLevelItem(38)
        ___qtreewidgetitem63.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.0", None));
        ___qtreewidgetitem63.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"dropout", None));
        ___qtreewidgetitem64 = self.TrainingConfigureEdit.topLevelItem(39)
        ___qtreewidgetitem64.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"True", None));
        ___qtreewidgetitem64.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"val", None));
        ___qtreewidgetitem65 = self.TrainingConfigureEdit.topLevelItem(40)
        ___qtreewidgetitem65.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem65.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"plots", None));
        self.TrainingConfigureEdit.setSortingEnabled(__sortingEnabled1)

        self.ConfigureTabPage.setTabText(self.ConfigureTabPage.indexOf(self.TrainingConfigure), QCoreApplication.translate("AnimalPoseTracker", u"Training Configure", None))
        self.EvaluateModelLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker-Evaluate Model", None))
        self.EvaluateModelGroup.setTitle("")
        self.EditEvaluationParameters.setText(QCoreApplication.translate("AnimalPoseTracker", u"Edit evaluation parameters", None))
        self.StartEvaluate.setText(QCoreApplication.translate("AnimalPoseTracker", u"Start evaluate", None))
        self.EndEvaluate.setText(QCoreApplication.translate("AnimalPoseTracker", u"End evaluate", None))
        ___qtreewidgetitem66 = self.EvaluateConfigure.headerItem()
        ___qtreewidgetitem66.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"Value", None));
        ___qtreewidgetitem66.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"Parameter", None));

        __sortingEnabled2 = self.EvaluateConfigure.isSortingEnabled()
        self.EvaluateConfigure.setSortingEnabled(False)
        ___qtreewidgetitem67 = self.EvaluateConfigure.topLevelItem(0)
        ___qtreewidgetitem67.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"data", None));
        ___qtreewidgetitem68 = self.EvaluateConfigure.topLevelItem(1)
        ___qtreewidgetitem68.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"16", None));
        ___qtreewidgetitem68.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"batch", None));
        ___qtreewidgetitem69 = self.EvaluateConfigure.topLevelItem(2)
        ___qtreewidgetitem69.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"640", None));
        ___qtreewidgetitem69.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"imgsz", None));
        ___qtreewidgetitem70 = self.EvaluateConfigure.topLevelItem(3)
        ___qtreewidgetitem70.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem70.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_json", None));
        ___qtreewidgetitem71 = self.EvaluateConfigure.topLevelItem(4)
        ___qtreewidgetitem71.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem71.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_hybrid", None));
        ___qtreewidgetitem72 = self.EvaluateConfigure.topLevelItem(5)
        ___qtreewidgetitem72.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.001", None));
        ___qtreewidgetitem72.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"conf", None));
        ___qtreewidgetitem73 = self.EvaluateConfigure.topLevelItem(6)
        ___qtreewidgetitem73.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"0.6", None));
        ___qtreewidgetitem73.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"iou", None));
        ___qtreewidgetitem74 = self.EvaluateConfigure.topLevelItem(7)
        ___qtreewidgetitem74.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"300", None));
        ___qtreewidgetitem74.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"max_det", None));
        ___qtreewidgetitem75 = self.EvaluateConfigure.topLevelItem(8)
        ___qtreewidgetitem75.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"True", None));
        ___qtreewidgetitem75.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"half", None));
        ___qtreewidgetitem76 = self.EvaluateConfigure.topLevelItem(9)
        ___qtreewidgetitem76.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"None", None));
        ___qtreewidgetitem76.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"device", None));
        ___qtreewidgetitem77 = self.EvaluateConfigure.topLevelItem(10)
        ___qtreewidgetitem77.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem77.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"dnn", None));
        ___qtreewidgetitem78 = self.EvaluateConfigure.topLevelItem(11)
        ___qtreewidgetitem78.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem78.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"plots", None));
        ___qtreewidgetitem79 = self.EvaluateConfigure.topLevelItem(12)
        ___qtreewidgetitem79.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"True", None));
        ___qtreewidgetitem79.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"rect", None));
        ___qtreewidgetitem80 = self.EvaluateConfigure.topLevelItem(13)
        ___qtreewidgetitem80.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"val", None));
        ___qtreewidgetitem80.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"split", None));
        ___qtreewidgetitem81 = self.EvaluateConfigure.topLevelItem(14)
        ___qtreewidgetitem81.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"None", None));
        ___qtreewidgetitem81.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"project", None));
        ___qtreewidgetitem82 = self.EvaluateConfigure.topLevelItem(15)
        ___qtreewidgetitem82.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"None", None));
        ___qtreewidgetitem82.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"name", None));
        ___qtreewidgetitem83 = self.EvaluateConfigure.topLevelItem(16)
        ___qtreewidgetitem83.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"false", None));
        ___qtreewidgetitem83.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"profile", None));
        ___qtreewidgetitem84 = self.EvaluateConfigure.topLevelItem(17)
        ___qtreewidgetitem84.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem84.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"verbose", None));
        ___qtreewidgetitem85 = self.EvaluateConfigure.topLevelItem(18)
        ___qtreewidgetitem85.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem85.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_txt", None));
        ___qtreewidgetitem86 = self.EvaluateConfigure.topLevelItem(19)
        ___qtreewidgetitem86.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem86.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_conf", None));
        ___qtreewidgetitem87 = self.EvaluateConfigure.topLevelItem(20)
        ___qtreewidgetitem87.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem87.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_crop", None));
        ___qtreewidgetitem88 = self.EvaluateConfigure.topLevelItem(21)
        ___qtreewidgetitem88.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"8", None));
        ___qtreewidgetitem88.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"workers", None));
        ___qtreewidgetitem89 = self.EvaluateConfigure.topLevelItem(22)
        ___qtreewidgetitem89.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem89.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"augment", None));
        ___qtreewidgetitem90 = self.EvaluateConfigure.topLevelItem(23)
        ___qtreewidgetitem90.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem90.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"agnostic_nms", None));
        ___qtreewidgetitem91 = self.EvaluateConfigure.topLevelItem(24)
        ___qtreewidgetitem91.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"False", None));
        ___qtreewidgetitem91.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"single_cls", None));
        self.EvaluateConfigure.setSortingEnabled(__sortingEnabled2)

        self.ConfigureTabPage.setTabText(self.ConfigureTabPage.indexOf(self.EvaluateModel), QCoreApplication.translate("AnimalPoseTracker", u"Evaluate Model", None))
        self.InferenceSourceLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker-Inference Source", None))
        self.InferenceSourceGroup.setTitle("")
        self.EditInferenceParameters.setText(QCoreApplication.translate("AnimalPoseTracker", u"Edit inference parameters", None))
        self.SelectionSource.setText(QCoreApplication.translate("AnimalPoseTracker", u"Selection Source", None))
        self.StartInference.setText(QCoreApplication.translate("AnimalPoseTracker", u"Start Inference", None))
        self.EndInference.setText(QCoreApplication.translate("AnimalPoseTracker", u"End Inference", None))
        ___qtreewidgetitem92 = self.InferenceConfigure.headerItem()
        ___qtreewidgetitem92.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"Value", None));
        ___qtreewidgetitem92.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"Parameter", None));

        __sortingEnabled3 = self.InferenceConfigure.isSortingEnabled()
        self.InferenceConfigure.setSortingEnabled(False)
        ___qtreewidgetitem93 = self.InferenceConfigure.topLevelItem(0)
        ___qtreewidgetitem93.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"source", None));
        ___qtreewidgetitem94 = self.InferenceConfigure.topLevelItem(1)
        ___qtreewidgetitem94.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"conf", None));
        ___qtreewidgetitem95 = self.InferenceConfigure.topLevelItem(2)
        ___qtreewidgetitem95.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"iou", None));
        ___qtreewidgetitem96 = self.InferenceConfigure.topLevelItem(3)
        ___qtreewidgetitem96.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"imgsz", None));
        ___qtreewidgetitem97 = self.InferenceConfigure.topLevelItem(4)
        ___qtreewidgetitem97.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"half", None));
        ___qtreewidgetitem98 = self.InferenceConfigure.topLevelItem(5)
        ___qtreewidgetitem98.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"device", None));
        ___qtreewidgetitem99 = self.InferenceConfigure.topLevelItem(6)
        ___qtreewidgetitem99.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"batch", None));
        ___qtreewidgetitem100 = self.InferenceConfigure.topLevelItem(7)
        ___qtreewidgetitem100.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"max_det", None));
        ___qtreewidgetitem101 = self.InferenceConfigure.topLevelItem(8)
        ___qtreewidgetitem101.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"vid_stride", None));
        ___qtreewidgetitem102 = self.InferenceConfigure.topLevelItem(9)
        ___qtreewidgetitem102.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"stream_buffer", None));
        ___qtreewidgetitem103 = self.InferenceConfigure.topLevelItem(10)
        ___qtreewidgetitem103.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"visualize", None));
        ___qtreewidgetitem104 = self.InferenceConfigure.topLevelItem(11)
        ___qtreewidgetitem104.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"augment", None));
        ___qtreewidgetitem105 = self.InferenceConfigure.topLevelItem(12)
        ___qtreewidgetitem105.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"agnostic_nms", None));
        ___qtreewidgetitem106 = self.InferenceConfigure.topLevelItem(13)
        ___qtreewidgetitem106.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"classes", None));
        ___qtreewidgetitem107 = self.InferenceConfigure.topLevelItem(14)
        ___qtreewidgetitem107.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"retina_masks", None));
        ___qtreewidgetitem108 = self.InferenceConfigure.topLevelItem(15)
        ___qtreewidgetitem108.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"embed", None));
        ___qtreewidgetitem109 = self.InferenceConfigure.topLevelItem(16)
        ___qtreewidgetitem109.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"project", None));
        ___qtreewidgetitem110 = self.InferenceConfigure.topLevelItem(17)
        ___qtreewidgetitem110.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"name", None));
        ___qtreewidgetitem111 = self.InferenceConfigure.topLevelItem(18)
        ___qtreewidgetitem111.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"stream", None));
        ___qtreewidgetitem112 = self.InferenceConfigure.topLevelItem(19)
        ___qtreewidgetitem112.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"verbose", None));
        ___qtreewidgetitem113 = self.InferenceConfigure.topLevelItem(20)
        ___qtreewidgetitem113.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"show", None));
        ___qtreewidgetitem114 = self.InferenceConfigure.topLevelItem(21)
        ___qtreewidgetitem114.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save", None));
        ___qtreewidgetitem115 = self.InferenceConfigure.topLevelItem(22)
        ___qtreewidgetitem115.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_frame", None));
        ___qtreewidgetitem116 = self.InferenceConfigure.topLevelItem(23)
        ___qtreewidgetitem116.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_txt", None));
        ___qtreewidgetitem117 = self.InferenceConfigure.topLevelItem(24)
        ___qtreewidgetitem117.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_conf", None));
        ___qtreewidgetitem118 = self.InferenceConfigure.topLevelItem(25)
        ___qtreewidgetitem118.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"save_crop", None));
        ___qtreewidgetitem119 = self.InferenceConfigure.topLevelItem(26)
        ___qtreewidgetitem119.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"show_label", None));
        ___qtreewidgetitem120 = self.InferenceConfigure.topLevelItem(27)
        ___qtreewidgetitem120.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"show_conf", None));
        ___qtreewidgetitem121 = self.InferenceConfigure.topLevelItem(28)
        ___qtreewidgetitem121.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"show_boxes", None));
        ___qtreewidgetitem122 = self.InferenceConfigure.topLevelItem(29)
        ___qtreewidgetitem122.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"line_width", None));
        ___qtreewidgetitem123 = self.InferenceConfigure.topLevelItem(30)
        ___qtreewidgetitem123.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"font_size", None));
        ___qtreewidgetitem124 = self.InferenceConfigure.topLevelItem(31)
        ___qtreewidgetitem124.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"font", None));
        ___qtreewidgetitem125 = self.InferenceConfigure.topLevelItem(32)
        ___qtreewidgetitem125.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"pil", None));
        ___qtreewidgetitem126 = self.InferenceConfigure.topLevelItem(33)
        ___qtreewidgetitem126.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"kpt_radius", None));
        ___qtreewidgetitem127 = self.InferenceConfigure.topLevelItem(34)
        ___qtreewidgetitem127.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"kpt_line", None));
        ___qtreewidgetitem128 = self.InferenceConfigure.topLevelItem(35)
        ___qtreewidgetitem128.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"masks", None));
        ___qtreewidgetitem129 = self.InferenceConfigure.topLevelItem(36)
        ___qtreewidgetitem129.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"probs", None));
        ___qtreewidgetitem130 = self.InferenceConfigure.topLevelItem(37)
        ___qtreewidgetitem130.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"filename", None));
        ___qtreewidgetitem131 = self.InferenceConfigure.topLevelItem(38)
        ___qtreewidgetitem131.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"color_mode", None));
        ___qtreewidgetitem132 = self.InferenceConfigure.topLevelItem(39)
        ___qtreewidgetitem132.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"txt_color", None));
        self.InferenceConfigure.setSortingEnabled(__sortingEnabled3)

        self.ConfigureTabPage.setTabText(self.ConfigureTabPage.indexOf(self.InferenceSource), QCoreApplication.translate("AnimalPoseTracker", u"Inference Source", None))
        self.ExportModelLabel.setText(QCoreApplication.translate("AnimalPoseTracker", u"AnimalPoseTracker-Export Model", None))
        self.ExportConfigureGroup.setTitle("")
        self.EditExportParameters.setText(QCoreApplication.translate("AnimalPoseTracker", u"Edit export parameters", None))
        self.StartModelWeights.setText(QCoreApplication.translate("AnimalPoseTracker", u"Selection model weights ", None))
        self.StartExport.setText(QCoreApplication.translate("AnimalPoseTracker", u"Start Export", None))
        ___qtreewidgetitem133 = self.ExportConfigure.headerItem()
        ___qtreewidgetitem133.setText(1, QCoreApplication.translate("AnimalPoseTracker", u"Value", None));
        ___qtreewidgetitem133.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"Parameter", None));

        __sortingEnabled4 = self.ExportConfigure.isSortingEnabled()
        self.ExportConfigure.setSortingEnabled(False)
        ___qtreewidgetitem134 = self.ExportConfigure.topLevelItem(0)
        ___qtreewidgetitem134.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"format", None));
        ___qtreewidgetitem135 = self.ExportConfigure.topLevelItem(1)
        ___qtreewidgetitem135.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"imgsz", None));
        ___qtreewidgetitem136 = self.ExportConfigure.topLevelItem(2)
        ___qtreewidgetitem136.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"keras", None));
        ___qtreewidgetitem137 = self.ExportConfigure.topLevelItem(3)
        ___qtreewidgetitem137.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"optimize", None));
        ___qtreewidgetitem138 = self.ExportConfigure.topLevelItem(4)
        ___qtreewidgetitem138.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"half", None));
        ___qtreewidgetitem139 = self.ExportConfigure.topLevelItem(5)
        ___qtreewidgetitem139.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"int8", None));
        ___qtreewidgetitem140 = self.ExportConfigure.topLevelItem(6)
        ___qtreewidgetitem140.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"dynamic", None));
        ___qtreewidgetitem141 = self.ExportConfigure.topLevelItem(7)
        ___qtreewidgetitem141.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"simplify", None));
        ___qtreewidgetitem142 = self.ExportConfigure.topLevelItem(8)
        ___qtreewidgetitem142.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"opset", None));
        ___qtreewidgetitem143 = self.ExportConfigure.topLevelItem(9)
        ___qtreewidgetitem143.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"workspace", None));
        ___qtreewidgetitem144 = self.ExportConfigure.topLevelItem(10)
        ___qtreewidgetitem144.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"nms", None));
        ___qtreewidgetitem145 = self.ExportConfigure.topLevelItem(11)
        ___qtreewidgetitem145.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"batch", None));
        ___qtreewidgetitem146 = self.ExportConfigure.topLevelItem(12)
        ___qtreewidgetitem146.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"device", None));
        ___qtreewidgetitem147 = self.ExportConfigure.topLevelItem(13)
        ___qtreewidgetitem147.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"data", None));
        ___qtreewidgetitem148 = self.ExportConfigure.topLevelItem(14)
        ___qtreewidgetitem148.setText(0, QCoreApplication.translate("AnimalPoseTracker", u"fraction", None));
        self.ExportConfigure.setSortingEnabled(__sortingEnabled4)

        self.ConfigureTabPage.setTabText(self.ConfigureTabPage.indexOf(self.ExportModel), QCoreApplication.translate("AnimalPoseTracker", u"Export Model", None))
        self.menuAnimalPoseTracker.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"File", None))
        self.menuTools.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"Tools", None))
        self.menuHelp.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"Help", None))
        self.menuView.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"View", None))
        self.menuTheme.setTitle(QCoreApplication.translate("AnimalPoseTracker", u"Theme", None))
        self.toolBar.setWindowTitle(QCoreApplication.translate("AnimalPoseTracker", u"toolBar", None))
    # retranslateUi

