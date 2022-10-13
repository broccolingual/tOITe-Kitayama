import pyautogui
import cv2
import time
from functions.setGUI import setGUI
from functions.common import PlaySound, CheckComplete
from functions.DesignLayout import make_fullimage_layout


# 処理の辞書割り当て ======================================================
def updateDictProc_QR(dictProc):
	dictProc_this = {
		"QR_Q"			: procQR_Q,
		"QR_CORRECT"	: procQR_Correct,
		"QR_WRONG"		: procQR_Wrong,
	}
	return dict(dictProc, **dictProc_this)


# レイアウト設定・辞書割り当て =============================================
def updateDictWindow_QR(dictWindow):
	layoutQR_Correct = make_fullimage_layout("images/correct.png", "QR_CORRECT")
	layoutQR_Wrong = make_fullimage_layout("images/wrong.png", "QR_WRONG")

	dictLayout = {
		"QR_Q"			: "None",
		"QR_CORRECT"	: layoutQR_Correct,
		"QR_WRONG"		: layoutQR_Wrong,
	}
	dictWindow_this = setGUI(dictLayout)

	return dict(dictWindow, **dictWindow_this)


# Qr_Qモード処理 ======================================================
def procQR_Q(dictArgument):
	event = dictArgument["Event"]
	cState = dictArgument["State"]
	cCtrlCard = dictArgument["CtrlCard"]
	proc = dictArgument["ImageProc"]

	isFound = proc.execute()
	cv2.waitKey(1)
	
	if isFound == True:
		PlaySound("sound/correct.wav")
		cCtrlCard.write_result("image", "T")
		proc.closeWindows()
		sStartTime = cState.updateState("QR_CORRECT")
		dictArgument["Start time"] = sStartTime
	elif time.time() - dictArgument["Start time"] > 5:
		PlaySound("sound/wrong.wav")
		proc.closeWindows()
		sStartTime = cState.updateState("QR_WRONG")
		dictArgument["Start time"] = sStartTime


# Qr_correctモード処理 ======================================================
def procQR_Correct(dictArgument):
	event = dictArgument["Event"]
	cState = dictArgument["State"]
	cCtrlCard = dictArgument["CtrlCard"]

	if event == "QR_CORRECT":
		cState.dictWindow["SELECT_GAME"]["画像"].update(disabled=True)

		if CheckComplete(cCtrlCard):
			cCtrlCard.write_result("complete", "T")
			sStartTime = cState.updateState("ENDING")
			dictArgument["Start time"] = sStartTime
		else:
			sStartTime = cState.updateState("SELECT_GAME")
			dictArgument["Start time"] = sStartTime


# Qr_wrongモード処理 ======================================================
def procQR_Wrong(dictArgument):
	event = dictArgument["Event"]
	cState = dictArgument["State"]

	if event == "QR_WRONG":
		sStartTime = cState.updateState("SELECT_GAME")
		dictArgument["Start time"] = sStartTime