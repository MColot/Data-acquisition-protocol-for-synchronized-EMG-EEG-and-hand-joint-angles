# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'remoteControl.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(966, 452)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.button_signs = QtWidgets.QPushButton(self.centralwidget)
        self.button_signs.setMinimumSize(QtCore.QSize(0, 50))
        self.button_signs.setObjectName("button_signs")
        self.verticalLayout.addWidget(self.button_signs)
        self.button_freeMoves = QtWidgets.QPushButton(self.centralwidget)
        self.button_freeMoves.setMinimumSize(QtCore.QSize(0, 50))
        self.button_freeMoves.setObjectName("button_freeMoves")
        self.verticalLayout.addWidget(self.button_freeMoves)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.label_curEx = QtWidgets.QLabel(self.centralwidget)
        self.label_curEx.setAlignment(QtCore.Qt.AlignCenter)
        self.label_curEx.setObjectName("label_curEx")
        self.verticalLayout.addWidget(self.label_curEx)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.button_syncUTC = QtWidgets.QPushButton(self.centralwidget)
        self.button_syncUTC.setMinimumSize(QtCore.QSize(0, 50))
        self.button_syncUTC.setObjectName("button_syncUTC")
        self.verticalLayout.addWidget(self.button_syncUTC)
        self.button_utc = QtWidgets.QPushButton(self.centralwidget)
        self.button_utc.setMinimumSize(QtCore.QSize(0, 50))
        self.button_utc.setObjectName("button_utc")
        self.verticalLayout.addWidget(self.button_utc)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_recordName = QtWidgets.QLabel(self.centralwidget)
        self.label_recordName.setScaledContents(False)
        self.label_recordName.setAlignment(QtCore.Qt.AlignCenter)
        self.label_recordName.setObjectName("label_recordName")
        self.verticalLayout_2.addWidget(self.label_recordName)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem2)
        self.label_status = QtWidgets.QLabel(self.centralwidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.label_status.setPalette(palette)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_status.setFont(font)
        self.label_status.setAlignment(QtCore.Qt.AlignCenter)
        self.label_status.setObjectName("label_status")
        self.verticalLayout_2.addWidget(self.label_status)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem3)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setScaledContents(False)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.button_startQuest = QtWidgets.QPushButton(self.centralwidget)
        self.button_startQuest.setMinimumSize(QtCore.QSize(0, 50))
        self.button_startQuest.setObjectName("button_startQuest")
        self.horizontalLayout_2.addWidget(self.button_startQuest)
        self.button_startEmgEeg = QtWidgets.QPushButton(self.centralwidget)
        self.button_startEmgEeg.setMinimumSize(QtCore.QSize(0, 50))
        self.button_startEmgEeg.setObjectName("button_startEmgEeg")
        self.horizontalLayout_2.addWidget(self.button_startEmgEeg)
        self.button_start = QtWidgets.QPushButton(self.centralwidget)
        self.button_start.setMinimumSize(QtCore.QSize(0, 50))
        self.button_start.setObjectName("button_start")
        self.horizontalLayout_2.addWidget(self.button_start)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem4)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setScaledContents(False)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.button_stop = QtWidgets.QPushButton(self.centralwidget)
        self.button_stop.setMinimumSize(QtCore.QSize(0, 50))
        self.button_stop.setObjectName("button_stop")
        self.horizontalLayout_3.addWidget(self.button_stop)
        self.button_reset = QtWidgets.QPushButton(self.centralwidget)
        self.button_reset.setMinimumSize(QtCore.QSize(0, 50))
        self.button_reset.setObjectName("button_reset")
        self.horizontalLayout_3.addWidget(self.button_reset)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem5)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 966, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Quest-EMG-EEG Remote Control"))
        self.label.setText(_translate("MainWindow", "VR exercices"))
        self.button_signs.setText(_translate("MainWindow", "Signs"))
        self.button_freeMoves.setText(_translate("MainWindow", "Free moves"))
        self.label_curEx.setText(_translate("MainWindow", "Currently displaying: Exercice"))
        self.button_syncUTC.setText(_translate("MainWindow", "Sync UTC"))
        self.button_utc.setText(_translate("MainWindow", "UTC"))
        self.label_recordName.setText(_translate("MainWindow", "Record Name"))
        self.label_status.setText(_translate("MainWindow", "Status: Not recording"))
        self.label_3.setText(_translate("MainWindow", "Start"))
        self.button_startQuest.setText(_translate("MainWindow", "Quest"))
        self.button_startEmgEeg.setText(_translate("MainWindow", "EMG-EEG"))
        self.button_start.setText(_translate("MainWindow", "Everything"))
        self.label_4.setText(_translate("MainWindow", "Stop"))
        self.button_stop.setText(_translate("MainWindow", "Stop recording"))
        self.button_reset.setText(_translate("MainWindow", "Reset"))
