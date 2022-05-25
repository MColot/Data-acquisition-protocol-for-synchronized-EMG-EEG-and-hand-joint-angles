# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'remoteControlUTCwidgetUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1000, 400)
        Form.setMinimumSize(QtCore.QSize(1500, 400))
        Form.setMaximumSize(QtCore.QSize(1500, 400))
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_utc = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.label_utc.setFont(font)
        self.label_utc.setObjectName("labe_utc")
        self.verticalLayout.addWidget(self.label_utc)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.button_minus1 = QtWidgets.QPushButton(Form)
        self.button_minus1.setMinimumSize(QtCore.QSize(0, 50))
        self.button_minus1.setObjectName("button_minus1")
        self.horizontalLayout.addWidget(self.button_minus1)
        self.button_plus1 = QtWidgets.QPushButton(Form)
        self.button_plus1.setMinimumSize(QtCore.QSize(0, 50))
        self.button_plus1.setObjectName("button_plus1")
        self.horizontalLayout.addWidget(self.button_plus1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.button_minus2 = QtWidgets.QPushButton(Form)
        self.button_minus2.setMinimumSize(QtCore.QSize(0, 50))
        self.button_minus2.setObjectName("button_minus2")
        self.horizontalLayout_2.addWidget(self.button_minus2)
        self.button_plus2 = QtWidgets.QPushButton(Form)
        self.button_plus2.setMinimumSize(QtCore.QSize(0, 50))
        self.button_plus2.setObjectName("button_plus2")
        self.horizontalLayout_2.addWidget(self.button_plus2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.button_validate = QtWidgets.QPushButton(Form)
        self.button_validate.setMinimumSize(QtCore.QSize(0, 50))
        self.button_validate.setObjectName("button_validate")
        self.horizontalLayout_3.addWidget(self.button_validate)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "UTC"))
        self.label_utc.setText(_translate("Form", "123456789 offset=0.012"))
        self.button_minus1.setText(_translate("Form", "-0.01"))
        self.button_plus1.setText(_translate("Form", "+0.01"))
        self.button_minus2.setText(_translate("Form", "-0.001"))
        self.button_plus2.setText(_translate("Form", "+0.001"))
        self.button_validate.setText(_translate("Form", "validate"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
