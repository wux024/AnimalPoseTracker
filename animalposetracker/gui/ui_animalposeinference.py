# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'animalposeinferenceiMxJEY.ui'
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
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QGroupBox, QHBoxLayout, QLabel, QLayout,
    QPushButton, QRadioButton, QSizePolicy, QSlider,
    QSpinBox, QVBoxLayout, QWidget)

from animalposetracker.gui import LOGO_SMALL_PATH

class Ui_AnimalPoseInference(object):
    def setupUi(self, AnimalPoseInference):
        if not AnimalPoseInference.objectName():
            AnimalPoseInference.setObjectName(u"AnimalPoseInference")
        AnimalPoseInference.resize(1321, 845)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AnimalPoseInference.sizePolicy().hasHeightForWidth())
        AnimalPoseInference.setSizePolicy(sizePolicy)
        AnimalPoseInference.setAcceptDrops(False)
        icon = QIcon()
        icon.addFile(LOGO_SMALL_PATH, QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        AnimalPoseInference.setWindowIcon(icon)
        self.verticalLayout = QVBoxLayout(AnimalPoseInference)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.BaseConfigure = QHBoxLayout()
        self.BaseConfigure.setObjectName(u"BaseConfigure")
        self.BaseToolsGroup = QGroupBox(AnimalPoseInference)
        self.BaseToolsGroup.setObjectName(u"BaseToolsGroup")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.BaseToolsGroup.sizePolicy().hasHeightForWidth())
        self.BaseToolsGroup.setSizePolicy(sizePolicy1)
        self.BaseToolsGroupLayout = QVBoxLayout(self.BaseToolsGroup)
        self.BaseToolsGroupLayout.setObjectName(u"BaseToolsGroupLayout")
        self.BaseToolsLayout = QVBoxLayout()
        self.BaseToolsLayout.setObjectName(u"BaseToolsLayout")
        self.BaseConfigureLayout = QHBoxLayout()
        self.BaseConfigureLayout.setObjectName(u"BaseConfigureLayout")
        self.SelectConfigure = QPushButton(self.BaseToolsGroup)
        self.SelectConfigure.setObjectName(u"SelectConfigure")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.SelectConfigure.sizePolicy().hasHeightForWidth())
        self.SelectConfigure.setSizePolicy(sizePolicy2)
        self.SelectConfigure.setMinimumSize(QSize(0, 25))
        self.SelectConfigure.setMaximumSize(QSize(16777215, 25))

        self.BaseConfigureLayout.addWidget(self.SelectConfigure)

        self.SelectWeights = QPushButton(self.BaseToolsGroup)
        self.SelectWeights.setObjectName(u"SelectWeights")
        sizePolicy2.setHeightForWidth(self.SelectWeights.sizePolicy().hasHeightForWidth())
        self.SelectWeights.setSizePolicy(sizePolicy2)
        self.SelectWeights.setMinimumSize(QSize(0, 25))
        self.SelectWeights.setMaximumSize(QSize(16777215, 25))

        self.BaseConfigureLayout.addWidget(self.SelectWeights)

        self.CameraORVideos = QRadioButton(self.BaseToolsGroup)
        self.CameraORVideos.setObjectName(u"CameraORVideos")
        sizePolicy2.setHeightForWidth(self.CameraORVideos.sizePolicy().hasHeightForWidth())
        self.CameraORVideos.setSizePolicy(sizePolicy2)

        self.BaseConfigureLayout.addWidget(self.CameraORVideos)


        self.BaseToolsLayout.addLayout(self.BaseConfigureLayout)

        self.CameraVideosLayout = QHBoxLayout()
        self.CameraVideosLayout.setObjectName(u"CameraVideosLayout")
        self.CameraVideosSelection = QComboBox(self.BaseToolsGroup)
        self.CameraVideosSelection.addItem("")
        self.CameraVideosSelection.addItem("")
        self.CameraVideosSelection.setObjectName(u"CameraVideosSelection")
        sizePolicy2.setHeightForWidth(self.CameraVideosSelection.sizePolicy().hasHeightForWidth())
        self.CameraVideosSelection.setSizePolicy(sizePolicy2)
        self.CameraVideosSelection.setMinimumSize(QSize(0, 25))
        self.CameraVideosSelection.setMaximumSize(QSize(16777215, 25))

        self.CameraVideosLayout.addWidget(self.CameraVideosSelection)

        self.CheckCameraVideosConnect = QPushButton(self.BaseToolsGroup)
        self.CheckCameraVideosConnect.setObjectName(u"CheckCameraVideosConnect")
        sizePolicy2.setHeightForWidth(self.CheckCameraVideosConnect.sizePolicy().hasHeightForWidth())
        self.CheckCameraVideosConnect.setSizePolicy(sizePolicy2)
        self.CheckCameraVideosConnect.setMinimumSize(QSize(0, 25))
        self.CheckCameraVideosConnect.setMaximumSize(QSize(16777215, 25))

        self.CameraVideosLayout.addWidget(self.CheckCameraVideosConnect)


        self.BaseToolsLayout.addLayout(self.CameraVideosLayout)

        self.CameraParametersLayout = QHBoxLayout()
        self.CameraParametersLayout.setObjectName(u"CameraParametersLayout")
        self.WidthLayout = QHBoxLayout()
        self.WidthLayout.setObjectName(u"WidthLayout")
        self.WidthLabel = QLabel(self.BaseToolsGroup)
        self.WidthLabel.setObjectName(u"WidthLabel")

        self.WidthLayout.addWidget(self.WidthLabel)

        self.WidthSetup = QSpinBox(self.BaseToolsGroup)
        self.WidthSetup.setObjectName(u"WidthSetup")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.WidthSetup.sizePolicy().hasHeightForWidth())
        self.WidthSetup.setSizePolicy(sizePolicy3)

        self.WidthLayout.addWidget(self.WidthSetup)


        self.CameraParametersLayout.addLayout(self.WidthLayout)

        self.HeightLayout = QHBoxLayout()
        self.HeightLayout.setObjectName(u"HeightLayout")
        self.HeightLabel = QLabel(self.BaseToolsGroup)
        self.HeightLabel.setObjectName(u"HeightLabel")

        self.HeightLayout.addWidget(self.HeightLabel)

        self.HeightSetup = QSpinBox(self.BaseToolsGroup)
        self.HeightSetup.setObjectName(u"HeightSetup")
        sizePolicy3.setHeightForWidth(self.HeightSetup.sizePolicy().hasHeightForWidth())
        self.HeightSetup.setSizePolicy(sizePolicy3)

        self.HeightLayout.addWidget(self.HeightSetup)


        self.CameraParametersLayout.addLayout(self.HeightLayout)

        self.FPSLayout = QHBoxLayout()
        self.FPSLayout.setObjectName(u"FPSLayout")
        self.FPSLabel = QLabel(self.BaseToolsGroup)
        self.FPSLabel.setObjectName(u"FPSLabel")

        self.FPSLayout.addWidget(self.FPSLabel)

        self.FPSSetup = QSpinBox(self.BaseToolsGroup)
        self.FPSSetup.setObjectName(u"FPSSetup")
        sizePolicy3.setHeightForWidth(self.FPSSetup.sizePolicy().hasHeightForWidth())
        self.FPSSetup.setSizePolicy(sizePolicy3)

        self.FPSLayout.addWidget(self.FPSSetup)


        self.CameraParametersLayout.addLayout(self.FPSLayout)


        self.BaseToolsLayout.addLayout(self.CameraParametersLayout)

        self.EngineDeviceLayout = QHBoxLayout()
        self.EngineDeviceLayout.setObjectName(u"EngineDeviceLayout")
        self.ModelBitsLayout = QHBoxLayout()
        self.ModelBitsLayout.setObjectName(u"ModelBitsLayout")
        self.ModelBitsLabel = QLabel(self.BaseToolsGroup)
        self.ModelBitsLabel.setObjectName(u"ModelBitsLabel")

        self.ModelBitsLayout.addWidget(self.ModelBitsLabel)

        self.ModelBits = QComboBox(self.BaseToolsGroup)
        self.ModelBits.addItem("")
        self.ModelBits.addItem("")
        self.ModelBits.addItem("")
        self.ModelBits.setObjectName(u"ModelBits")
        sizePolicy3.setHeightForWidth(self.ModelBits.sizePolicy().hasHeightForWidth())
        self.ModelBits.setSizePolicy(sizePolicy3)
        self.ModelBits.setMinimumSize(QSize(0, 25))
        self.ModelBits.setMaximumSize(QSize(16777215, 25))

        self.ModelBitsLayout.addWidget(self.ModelBits)


        self.EngineDeviceLayout.addLayout(self.ModelBitsLayout)

        self.EngineLayout = QHBoxLayout()
        self.EngineLayout.setObjectName(u"EngineLayout")
        self.EngineLabel = QLabel(self.BaseToolsGroup)
        self.EngineLabel.setObjectName(u"EngineLabel")

        self.EngineLayout.addWidget(self.EngineLabel)

        self.EngineSelection = QComboBox(self.BaseToolsGroup)
        self.EngineSelection.addItem("")
        self.EngineSelection.addItem("")
        self.EngineSelection.addItem("")
        self.EngineSelection.addItem("")
        self.EngineSelection.addItem("")
        self.EngineSelection.addItem("")
        self.EngineSelection.setObjectName(u"EngineSelection")
        sizePolicy3.setHeightForWidth(self.EngineSelection.sizePolicy().hasHeightForWidth())
        self.EngineSelection.setSizePolicy(sizePolicy3)
        self.EngineSelection.setMinimumSize(QSize(0, 25))
        self.EngineSelection.setMaximumSize(QSize(16777215, 25))

        self.EngineLayout.addWidget(self.EngineSelection)


        self.EngineDeviceLayout.addLayout(self.EngineLayout)

        self.DeviceLayout = QHBoxLayout()
        self.DeviceLayout.setObjectName(u"DeviceLayout")
        self.DeviceLabel = QLabel(self.BaseToolsGroup)
        self.DeviceLabel.setObjectName(u"DeviceLabel")

        self.DeviceLayout.addWidget(self.DeviceLabel)

        self.DeviceSelection = QComboBox(self.BaseToolsGroup)
        self.DeviceSelection.addItem("")
        self.DeviceSelection.addItem("")
        self.DeviceSelection.addItem("")
        self.DeviceSelection.addItem("")
        self.DeviceSelection.addItem("")
        self.DeviceSelection.addItem("")
        self.DeviceSelection.addItem("")
        self.DeviceSelection.addItem("")
        self.DeviceSelection.setObjectName(u"DeviceSelection")
        sizePolicy3.setHeightForWidth(self.DeviceSelection.sizePolicy().hasHeightForWidth())
        self.DeviceSelection.setSizePolicy(sizePolicy3)
        self.DeviceSelection.setMinimumSize(QSize(0, 25))
        self.DeviceSelection.setMaximumSize(QSize(16777215, 25))

        self.DeviceLayout.addWidget(self.DeviceSelection)


        self.EngineDeviceLayout.addLayout(self.DeviceLayout)


        self.BaseToolsLayout.addLayout(self.EngineDeviceLayout)

        self.StartEndLayout = QHBoxLayout()
        self.StartEndLayout.setObjectName(u"StartEndLayout")
        self.Start = QPushButton(self.BaseToolsGroup)
        self.Start.setObjectName(u"Start")
        self.Start.setMinimumSize(QSize(0, 25))
        self.Start.setMaximumSize(QSize(16777215, 25))

        self.StartEndLayout.addWidget(self.Start)

        self.End = QPushButton(self.BaseToolsGroup)
        self.End.setObjectName(u"End")
        self.End.setMinimumSize(QSize(0, 25))
        self.End.setMaximumSize(QSize(16777215, 25))

        self.StartEndLayout.addWidget(self.End)


        self.BaseToolsLayout.addLayout(self.StartEndLayout)


        self.BaseToolsGroupLayout.addLayout(self.BaseToolsLayout)

        self.PrintInformation = QLabel(self.BaseToolsGroup)
        self.PrintInformation.setObjectName(u"PrintInformation")
        sizePolicy2.setHeightForWidth(self.PrintInformation.sizePolicy().hasHeightForWidth())
        self.PrintInformation.setSizePolicy(sizePolicy2)
        self.PrintInformation.setMinimumSize(QSize(0, 25))
        self.PrintInformation.setMaximumSize(QSize(16777215, 25))

        self.BaseToolsGroupLayout.addWidget(self.PrintInformation)


        self.BaseConfigure.addWidget(self.BaseToolsGroup)

        self.ShowToolsGroup = QGroupBox(AnimalPoseInference)
        self.ShowToolsGroup.setObjectName(u"ShowToolsGroup")
        sizePolicy1.setHeightForWidth(self.ShowToolsGroup.sizePolicy().hasHeightForWidth())
        self.ShowToolsGroup.setSizePolicy(sizePolicy1)
        self.ShowToolsGroupVLayout = QVBoxLayout(self.ShowToolsGroup)
        self.ShowToolsGroupVLayout.setObjectName(u"ShowToolsGroupVLayout")
        self.SaveIoUConfLayout = QHBoxLayout()
        self.SaveIoUConfLayout.setObjectName(u"SaveIoUConfLayout")
        self.Save = QCheckBox(self.ShowToolsGroup)
        self.Save.setObjectName(u"Save")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.Save.sizePolicy().hasHeightForWidth())
        self.Save.setSizePolicy(sizePolicy4)
        self.Save.setMinimumSize(QSize(120, 25))
        self.Save.setMaximumSize(QSize(120, 25))

        self.SaveIoUConfLayout.addWidget(self.Save)

        self.IoULayout = QHBoxLayout()
        self.IoULayout.setObjectName(u"IoULayout")
        self.IoULabel = QLabel(self.ShowToolsGroup)
        self.IoULabel.setObjectName(u"IoULabel")
        sizePolicy4.setHeightForWidth(self.IoULabel.sizePolicy().hasHeightForWidth())
        self.IoULabel.setSizePolicy(sizePolicy4)

        self.IoULayout.addWidget(self.IoULabel)

        self.IoU = QDoubleSpinBox(self.ShowToolsGroup)
        self.IoU.setObjectName(u"IoU")
        self.IoU.setMaximum(1.000000000000000)
        self.IoU.setSingleStep(0.010000000000000)
        self.IoU.setValue(0.700000000000000)

        self.IoULayout.addWidget(self.IoU)


        self.SaveIoUConfLayout.addLayout(self.IoULayout)

        self.ConfLayout = QHBoxLayout()
        self.ConfLayout.setObjectName(u"ConfLayout")
        self.ConfLabel = QLabel(self.ShowToolsGroup)
        self.ConfLabel.setObjectName(u"ConfLabel")
        sizePolicy4.setHeightForWidth(self.ConfLabel.sizePolicy().hasHeightForWidth())
        self.ConfLabel.setSizePolicy(sizePolicy4)

        self.ConfLayout.addWidget(self.ConfLabel)

        self.Conf = QDoubleSpinBox(self.ShowToolsGroup)
        self.Conf.setObjectName(u"Conf")
        self.Conf.setMaximum(1.000000000000000)
        self.Conf.setSingleStep(0.010000000000000)
        self.Conf.setValue(0.250000000000000)

        self.ConfLayout.addWidget(self.Conf)


        self.SaveIoUConfLayout.addLayout(self.ConfLayout)


        self.ShowToolsGroupVLayout.addLayout(self.SaveIoUConfLayout)

        self.ShowLayout = QHBoxLayout()
        self.ShowLayout.setSpacing(0)
        self.ShowLayout.setObjectName(u"ShowLayout")
        self.ShowLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.Show = QCheckBox(self.ShowToolsGroup)
        self.Show.setObjectName(u"Show")
        sizePolicy4.setHeightForWidth(self.Show.sizePolicy().hasHeightForWidth())
        self.Show.setSizePolicy(sizePolicy4)
        self.Show.setMinimumSize(QSize(120, 25))
        self.Show.setMaximumSize(QSize(120, 25))

        self.ShowLayout.addWidget(self.Show)

        self.Backgroud = QCheckBox(self.ShowToolsGroup)
        self.Backgroud.setObjectName(u"Backgroud")
        self.Backgroud.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.Backgroud.sizePolicy().hasHeightForWidth())
        self.Backgroud.setSizePolicy(sizePolicy2)
        self.Backgroud.setMinimumSize(QSize(120, 25))
        self.Backgroud.setTristate(True)

        self.ShowLayout.addWidget(self.Backgroud)


        self.ShowToolsGroupVLayout.addLayout(self.ShowLayout)

        self.ShowClassesLayout = QHBoxLayout()
        self.ShowClassesLayout.setObjectName(u"ShowClassesLayout")
        self.ShowClasses = QCheckBox(self.ShowToolsGroup)
        self.ShowClasses.setObjectName(u"ShowClasses")
        sizePolicy4.setHeightForWidth(self.ShowClasses.sizePolicy().hasHeightForWidth())
        self.ShowClasses.setSizePolicy(sizePolicy4)
        self.ShowClasses.setMinimumSize(QSize(120, 25))
        self.ShowClasses.setMaximumSize(QSize(120, 25))

        self.ShowClassesLayout.addWidget(self.ShowClasses)

        self.Classes = QLabel(self.ShowToolsGroup)
        self.Classes.setObjectName(u"Classes")
        self.Classes.setMinimumSize(QSize(0, 25))
        self.Classes.setMaximumSize(QSize(16777215, 25))

        self.ShowClassesLayout.addWidget(self.Classes)


        self.ShowToolsGroupVLayout.addLayout(self.ShowClassesLayout)

        self.ShowKeypointsLayout = QHBoxLayout()
        self.ShowKeypointsLayout.setObjectName(u"ShowKeypointsLayout")
        self.ShowKeypoints = QCheckBox(self.ShowToolsGroup)
        self.ShowKeypoints.setObjectName(u"ShowKeypoints")
        sizePolicy4.setHeightForWidth(self.ShowKeypoints.sizePolicy().hasHeightForWidth())
        self.ShowKeypoints.setSizePolicy(sizePolicy4)
        self.ShowKeypoints.setMinimumSize(QSize(120, 25))
        self.ShowKeypoints.setMaximumSize(QSize(120, 25))

        self.ShowKeypointsLayout.addWidget(self.ShowKeypoints)

        self.ShowKeypointsRadius = QSlider(self.ShowToolsGroup)
        self.ShowKeypointsRadius.setObjectName(u"ShowKeypointsRadius")
        self.ShowKeypointsRadius.setMinimum(1)
        self.ShowKeypointsRadius.setMaximum(20)
        self.ShowKeypointsRadius.setPageStep(2)
        self.ShowKeypointsRadius.setSliderPosition(6)
        self.ShowKeypointsRadius.setOrientation(Qt.Orientation.Horizontal)
        self.ShowKeypointsRadius.setInvertedAppearance(False)
        self.ShowKeypointsRadius.setInvertedControls(False)
        self.ShowKeypointsRadius.setTickPosition(QSlider.TickPosition.NoTicks)

        self.ShowKeypointsLayout.addWidget(self.ShowKeypointsRadius)


        self.ShowToolsGroupVLayout.addLayout(self.ShowKeypointsLayout)

        self.ShowSkeletonsLayout = QHBoxLayout()
        self.ShowSkeletonsLayout.setObjectName(u"ShowSkeletonsLayout")
        self.ShowSkeletons = QCheckBox(self.ShowToolsGroup)
        self.ShowSkeletons.setObjectName(u"ShowSkeletons")
        sizePolicy4.setHeightForWidth(self.ShowSkeletons.sizePolicy().hasHeightForWidth())
        self.ShowSkeletons.setSizePolicy(sizePolicy4)
        self.ShowSkeletons.setMinimumSize(QSize(120, 25))
        self.ShowSkeletons.setMaximumSize(QSize(120, 25))

        self.ShowSkeletonsLayout.addWidget(self.ShowSkeletons)

        self.ShowSkeletonsLineWidth = QSlider(self.ShowToolsGroup)
        self.ShowSkeletonsLineWidth.setObjectName(u"ShowSkeletonsLineWidth")
        self.ShowSkeletonsLineWidth.setMinimum(1)
        self.ShowSkeletonsLineWidth.setMaximum(10)
        self.ShowSkeletonsLineWidth.setPageStep(1)
        self.ShowSkeletonsLineWidth.setValue(2)
        self.ShowSkeletonsLineWidth.setOrientation(Qt.Orientation.Horizontal)

        self.ShowSkeletonsLayout.addWidget(self.ShowSkeletonsLineWidth)


        self.ShowToolsGroupVLayout.addLayout(self.ShowSkeletonsLayout)

        self.ShowBBoxLayout = QHBoxLayout()
        self.ShowBBoxLayout.setObjectName(u"ShowBBoxLayout")
        self.ShowBBox = QCheckBox(self.ShowToolsGroup)
        self.ShowBBox.setObjectName(u"ShowBBox")
        sizePolicy4.setHeightForWidth(self.ShowBBox.sizePolicy().hasHeightForWidth())
        self.ShowBBox.setSizePolicy(sizePolicy4)
        self.ShowBBox.setMinimumSize(QSize(120, 25))
        self.ShowBBox.setMaximumSize(QSize(120, 25))

        self.ShowBBoxLayout.addWidget(self.ShowBBox)

        self.ShowBBoxWidth = QSlider(self.ShowToolsGroup)
        self.ShowBBoxWidth.setObjectName(u"ShowBBoxWidth")
        self.ShowBBoxWidth.setMinimum(1)
        self.ShowBBoxWidth.setMaximum(10)
        self.ShowBBoxWidth.setPageStep(1)
        self.ShowBBoxWidth.setValue(2)
        self.ShowBBoxWidth.setOrientation(Qt.Orientation.Horizontal)

        self.ShowBBoxLayout.addWidget(self.ShowBBoxWidth)


        self.ShowToolsGroupVLayout.addLayout(self.ShowBBoxLayout)


        self.BaseConfigure.addWidget(self.ShowToolsGroup)


        self.verticalLayout.addLayout(self.BaseConfigure)

        self.DisplayGroup = QGroupBox(AnimalPoseInference)
        self.DisplayGroup.setObjectName(u"DisplayGroup")
        sizePolicy.setHeightForWidth(self.DisplayGroup.sizePolicy().hasHeightForWidth())
        self.DisplayGroup.setSizePolicy(sizePolicy)
        self.DisplayGroup.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.DisplayGroupHLayout = QHBoxLayout(self.DisplayGroup)
        self.DisplayGroupHLayout.setObjectName(u"DisplayGroupHLayout")
        self.Display = QLabel(self.DisplayGroup)
        self.Display.setObjectName(u"Display")
        sizePolicy.setHeightForWidth(self.Display.sizePolicy().hasHeightForWidth())
        self.Display.setSizePolicy(sizePolicy)
        self.Display.setMinimumSize(QSize(100, 100))
        self.Display.setMaximumSize(QSize(1000, 1000))
        self.Display.setToolTipDuration(0)
        self.Display.setScaledContents(True)
        self.Display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.DisplayGroupHLayout.addWidget(self.Display)


        self.verticalLayout.addWidget(self.DisplayGroup)


        self.retranslateUi(AnimalPoseInference)

        QMetaObject.connectSlotsByName(AnimalPoseInference)
    # setupUi

    def retranslateUi(self, AnimalPoseInference):
        AnimalPoseInference.setWindowTitle(QCoreApplication.translate("AnimalPoseInference", u"AnimalPoseInference", None))
        self.BaseToolsGroup.setTitle("")
        self.SelectConfigure.setText(QCoreApplication.translate("AnimalPoseInference", u"Select Configure", None))
        self.SelectWeights.setText(QCoreApplication.translate("AnimalPoseInference", u"Select Weights", None))
        self.CameraORVideos.setText(QCoreApplication.translate("AnimalPoseInference", u"Use Video", None))
        self.CameraVideosSelection.setItemText(0, QCoreApplication.translate("AnimalPoseInference", u"...", None))
        self.CameraVideosSelection.setItemText(1, QCoreApplication.translate("AnimalPoseInference", u"Select video file...", None))

        self.CheckCameraVideosConnect.setText(QCoreApplication.translate("AnimalPoseInference", u"Preview Video", None))
        self.WidthLabel.setText(QCoreApplication.translate("AnimalPoseInference", u"Width", None))
        self.HeightLabel.setText(QCoreApplication.translate("AnimalPoseInference", u"Height", None))
        self.FPSLabel.setText(QCoreApplication.translate("AnimalPoseInference", u"FPS", None))
        self.ModelBitsLabel.setText(QCoreApplication.translate("AnimalPoseInference", u"Model Bits", None))
        self.ModelBits.setItemText(0, QCoreApplication.translate("AnimalPoseInference", u"FP32", None))
        self.ModelBits.setItemText(1, QCoreApplication.translate("AnimalPoseInference", u"FP16", None))
        self.ModelBits.setItemText(2, QCoreApplication.translate("AnimalPoseInference", u"INT8", None))

        self.EngineLabel.setText(QCoreApplication.translate("AnimalPoseInference", u"Engine", None))
        self.EngineSelection.setItemText(0, QCoreApplication.translate("AnimalPoseInference", u"OpenCV", None))
        self.EngineSelection.setItemText(1, QCoreApplication.translate("AnimalPoseInference", u"ONNX", None))
        self.EngineSelection.setItemText(2, QCoreApplication.translate("AnimalPoseInference", u"OpenVINO", None))
        self.EngineSelection.setItemText(3, QCoreApplication.translate("AnimalPoseInference", u"TensorRT", None))
        self.EngineSelection.setItemText(4, QCoreApplication.translate("AnimalPoseInference", u"CoreML", None))
        self.EngineSelection.setItemText(5, QCoreApplication.translate("AnimalPoseInference", u"CANN", None))

        self.DeviceLabel.setText(QCoreApplication.translate("AnimalPoseInference", u"Inference Device", None))
        self.DeviceSelection.setItemText(0, QCoreApplication.translate("AnimalPoseInference", u"Intel CPU", None))
        self.DeviceSelection.setItemText(1, QCoreApplication.translate("AnimalPoseInference", u"Intel GPU", None))
        self.DeviceSelection.setItemText(2, QCoreApplication.translate("AnimalPoseInference", u"Intel NPU", None))
        self.DeviceSelection.setItemText(3, QCoreApplication.translate("AnimalPoseInference", u"NVIDIA GPU", None))
        self.DeviceSelection.setItemText(4, QCoreApplication.translate("AnimalPoseInference", u"AMD CPU", None))
        self.DeviceSelection.setItemText(5, QCoreApplication.translate("AnimalPoseInference", u"AMD GPU", None))
        self.DeviceSelection.setItemText(6, QCoreApplication.translate("AnimalPoseInference", u"ARM CPU", None))
        self.DeviceSelection.setItemText(7, QCoreApplication.translate("AnimalPoseInference", u"Ascend NPU", None))

        self.Start.setText(QCoreApplication.translate("AnimalPoseInference", u"Start", None))
        self.End.setText(QCoreApplication.translate("AnimalPoseInference", u"End", None))
        self.PrintInformation.setText(QCoreApplication.translate("AnimalPoseInference", u"Some print information", None))
        self.ShowToolsGroup.setTitle("")
        self.Save.setText(QCoreApplication.translate("AnimalPoseInference", u"Save", None))
        self.IoULabel.setText(QCoreApplication.translate("AnimalPoseInference", u"IoU", None))
        self.ConfLabel.setText(QCoreApplication.translate("AnimalPoseInference", u"Conf", None))
        self.Show.setText(QCoreApplication.translate("AnimalPoseInference", u"Show", None))
        self.Backgroud.setText(QCoreApplication.translate("AnimalPoseInference", u"Original", None))
        self.ShowClasses.setText(QCoreApplication.translate("AnimalPoseInference", u"Show Classes", None))
        self.Classes.setText("")
        self.ShowKeypoints.setText(QCoreApplication.translate("AnimalPoseInference", u"Show Keypoints", None))
        self.ShowSkeletons.setText(QCoreApplication.translate("AnimalPoseInference", u"Show Skeletons", None))
        self.ShowBBox.setText(QCoreApplication.translate("AnimalPoseInference", u"Show BBox", None))
        self.DisplayGroup.setTitle("")
        self.Display.setText("")
    # retranslateUi

