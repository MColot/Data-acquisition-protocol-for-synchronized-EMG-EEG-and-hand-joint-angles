import sys

from PyQt5.QtCore import QTimer
from ui.remoteControlUI import *
from ui.remoteControlSignWidgetUI import *
from ui.remoteControlFreemoveWidgetUI import *
from ui.remoteControlNameWidgetUI import *
from ui.remoteControlUTCwidgetUI import *
from remoteControl import *

selectedFreemove = 0
freemoveButtons = None
currentDisplay = "Nothing"
recordName = ""
recordType = 0
recOn = False


def startRecording():
    global recordName, recOn
    name = uiName.lineEdit_name.text()
    if recordType == 0:  # Quest only
        recordQuest(name)
    elif recordType == 1:  # EMG-EEG only
        recordEMG(name)
    elif recordType == 2:  # Quest + EMG-EEG
        recordQuest(name)
        recordEMG(name)
    elif recordType == 3:  # Quest + EMG-EEG + signs
        execAndUpdate(lambda: signLangInstruction(uiSigns.lineEdit_rep.text(), uiSigns.lineEdit_gestures.text()))
        recordQuest(name)
        recordEMG(name)
        SignsWidget.hide()
    elif recordType == 4:  # Quest + EMG-EEG + freemoves
        execAndUpdate(lambda: freeMoveInstruction(selectedFreemove))
        recordQuest(name)
        recordEMG(name)
        FreemoveWidget.hide()
    NameWidget.hide()
    recordName = name
    recOn = True
    updateStatus()


def stop(r=False):
    global recordName, recOn
    if (r):
        reset()
    else:
        stopRecording()
    recordName = ""
    recOn = False
    updateStatus()


def displayNameWidget(recType):
    global recordType
    recordType = recType
    NameWidget.show()


def selectFreemove(index):
    global selectedFreemove
    selectedFreemove = index
    for i in range(6):
        font = QtGui.QFont()
        font.setBold(i == index)
        font.setUnderline(i == index)
        font.setWeight(100 if i == index else 25)
        font.setPointSize(12 if i == index else 8)
        freemoveButtons[i].setFont(font)


def updateStatus():
    uiMain.label_recordName.setText(f"record name: {recordName}")
    uiMain.label_status.setText(
        f"Status \nQuest:{' ' if recOn and recordType != 1 else ' not '}recording \nEMG-EEG:{' ' if recOn and recordType != 0 else ' not '}recording")
    uiMain.label_curEx.setText(f"Currently displaying: {currentDisplay}")


def execAndUpdate(fun):
    global currentDisplay
    currentDisplay = fun()
    updateStatus()


def utc(utcWidget, timer):
    displayUtc()
    utcWidget.show()
    timer.start(0)

def stopUtc(utcWidget, timer):
    timer.stop()
    utcWidget.hide()



def updateUtc(uiUtc):
    global manualTimeOffset
    uiUtc.label_utc.setText(f"{round(getTime(), 3)}, offset = {getTimeOffset()}")


def updateTimeOffset(v):
    global manualTimeOffset
    setTimeOffset(round(getTimeOffset() + v, 3))


def linkButtons(uiMain, uiSigns, uiFreemove, uiName, uiUtc, nameWidget, SignsWidget, freemoveWidget, utcWidget):
    global freemoveButtons, stopUpdateUtc
    freemoveButtons = (
        uiFreemove.button_freemove0, uiFreemove.button_freemove1, uiFreemove.button_freemove2,
        uiFreemove.button_freemove3,
        uiFreemove.button_freemove4, uiFreemove.button_freemove5)

    uiMain.button_signs.clicked.connect(lambda: SignsWidget.show())
    uiMain.button_freeMoves.clicked.connect(lambda: freemoveWidget.show())
    uiSigns.button_cancel.clicked.connect(lambda: SignsWidget.hide())
    uiFreemove.button_cancel.clicked.connect(lambda: freemoveWidget.hide())
    uiSigns.button_send.clicked.connect(lambda: execAndUpdate(
        lambda: signLangInstruction(uiSigns.lineEdit_rep.text(), uiSigns.lineEdit_gestures.text())))
    freemoveButtons[0].clicked.connect(lambda: selectFreemove(0))
    freemoveButtons[1].clicked.connect(lambda: selectFreemove(1))
    freemoveButtons[2].clicked.connect(lambda: selectFreemove(2))
    freemoveButtons[3].clicked.connect(lambda: selectFreemove(3))
    freemoveButtons[4].clicked.connect(lambda: selectFreemove(4))
    freemoveButtons[5].clicked.connect(lambda: selectFreemove(5))
    uiFreemove.button_send.clicked.connect(lambda: execAndUpdate(lambda: freeMoveInstruction(selectedFreemove)))
    uiMain.button_startQuest.clicked.connect(lambda: displayNameWidget(0))
    uiMain.button_startEmgEeg.clicked.connect(lambda: displayNameWidget(1))
    uiMain.button_start.clicked.connect(lambda: displayNameWidget(2))
    uiName.button_cancel.clicked.connect(lambda: nameWidget.hide())
    uiName.button_start.clicked.connect(lambda: startRecording())
    uiMain.button_stop.clicked.connect(lambda: stop(False))
    uiMain.button_reset.clicked.connect(lambda: stop(True))
    uiSigns.button_sendRecord.clicked.connect(lambda: displayNameWidget(3))
    uiFreemove.button_sendRecord.clicked.connect(lambda: displayNameWidget(4))
    uiMain.button_syncUTC.clicked.connect(lambda: syncUtc())

    timer = QTimer(utcWidget)
    timer.timeout.connect(lambda: updateUtc(uiUTC))
    uiMain.button_utc.clicked.connect(lambda: utc(utcWidget, timer))
    uiUtc.button_validate.clicked.connect(lambda: stopUtc(utcWidget, timer))
    uiUtc.button_plus1.clicked.connect(lambda: updateTimeOffset(0.01))
    uiUtc.button_minus1.clicked.connect(lambda: updateTimeOffset(-0.01))
    uiUtc.button_plus2.clicked.connect(lambda: updateTimeOffset(0.001))
    uiUtc.button_minus2.clicked.connect(lambda: updateTimeOffset(-0.001))



    updateStatus()
    selectFreemove(0)





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()
    uiMain = Ui_MainWindow()
    uiMain.setupUi(MainWindow)

    SignsWidget = QtWidgets.QWidget()
    uiSigns = Ui_SignsWidget()
    uiSigns.setupUi(SignsWidget)

    FreemoveWidget = QtWidgets.QWidget()
    uiFreemove = Ui_FreemoveWidget()
    uiFreemove.setupUi(FreemoveWidget)

    NameWidget = QtWidgets.QWidget()
    uiName = Ui_NameWidget()
    uiName.setupUi(NameWidget)

    utcWidget = QtWidgets.QWidget()
    uiUTC = Ui_Form()
    uiUTC.setupUi(utcWidget)

    linkButtons(uiMain, uiSigns, uiFreemove, uiName, uiUTC, NameWidget, SignsWidget, FreemoveWidget, utcWidget)

    MainWindow.show()
    sys.exit(app.exec_())
