# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'remoteControlNameUi.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_NameWidget(object):
    def setupUi(self, NameWidget):
        NameWidget.setObjectName("NameWidget")
        NameWidget.resize(424, 104)
        NameWidget.setMinimumSize(QtCore.QSize(424, 104))
        NameWidget.setMaximumSize(QtCore.QSize(424, 104))
        self.verticalLayout = QtWidgets.QVBoxLayout(NameWidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(NameWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.lineEdit_name = QtWidgets.QLineEdit(NameWidget)
        self.lineEdit_name.setObjectName("lineEdit_name")
        self.verticalLayout.addWidget(self.lineEdit_name)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.button_cancel = QtWidgets.QPushButton(NameWidget)
        self.button_cancel.setObjectName("button_cancel")
        self.horizontalLayout.addWidget(self.button_cancel)
        self.button_start = QtWidgets.QPushButton(NameWidget)
        self.button_start.setMinimumSize(QtCore.QSize(200, 28))
        self.button_start.setMaximumSize(QtCore.QSize(200, 28))
        self.button_start.setObjectName("button_start")
        self.horizontalLayout.addWidget(self.button_start)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(NameWidget)
        QtCore.QMetaObject.connectSlotsByName(NameWidget)

    def retranslateUi(self, NameWidget):
        _translate = QtCore.QCoreApplication.translate
        NameWidget.setWindowTitle(_translate("NameWidget", "Form"))
        self.label.setText(_translate("NameWidget", "Name of the record : "))
        self.button_cancel.setText(_translate("NameWidget", "Cancel"))
        self.button_start.setText(_translate("NameWidget", "Start"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    NameWidget = QtWidgets.QWidget()
    ui = Ui_NameWidget()
    ui.setupUi(NameWidget)
    NameWidget.show()
    sys.exit(app.exec_())
