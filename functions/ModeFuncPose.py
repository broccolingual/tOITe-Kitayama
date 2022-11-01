import cv2
from functions.setGUI import setGUI
from functions.DesignLayout import make_fullimage_layout


# 処理の辞書割り当て ======================================================
def updateDictProc_Pose(dictProc):
    dictProc_this = {
        "POSE_Q": procPose_Q,
        "POSE_IMGPROC": procPose_ImageProc,
        "POSE_CORRECT": procPose_correct,
        "POSE_WRONG": procPose_wrong,
    }
    return dict(dictProc, **dictProc_this)


# レイアウト設定・辞書割り当て =============================================
def updateDictWindow_Pose(dictWindow):
    layoutPose_Q = make_fullimage_layout("images/tutorial1.png", "POSE_Q")
    layoutPose_Correct = make_fullimage_layout(
        "images/clear.png", "POSE_CORRECT")
    layoutPose_Wrong = make_fullimage_layout(
        "images/gameover.png", "POSE_WRONG")

    dictLayout = {
        "POSE_Q": layoutPose_Q,
        "POSE_IMGPROC": "None",
        "POSE_CORRECT": layoutPose_Correct,
        "POSE_WRONG": layoutPose_Wrong
    }
    dictWindow_this = setGUI(dictLayout)

    return dict(dictWindow, **dictWindow_this)


# POSE_Qモード処理 ==============================================
def procPose_Q(dictArgument):
    event = dictArgument["Event"]
    cState = dictArgument["State"]
    proc = dictArgument["ImageProc"]
    cAudioOut = dictArgument["AudioOut"]

    if event == "POSE_Q":
        cAudioOut.playSoundAsync("sound/do_attack.wav")
        dictArgument["Start time"] = cState.updateState("POSE_IMGPROC")
        proc.createWindows()


# POSE_IMGPROCモード処理 ======================================================
def procPose_ImageProc(dictArgument):
    cState = dictArgument["State"]
    sFrame = dictArgument["Frame"]
    proc = dictArgument["ImageProc"]
    cCtrlCard = dictArgument["CtrlCard"]
    cAudioOut = dictArgument["AudioOut"]
    cLogger = dictArgument["Logger"]

    isFound = proc.execute()
    cv2.waitKey(1)

    # cLogger.logDebug("isFound : ", isFound)
    # print("sCountFound",sCountFound)
    # print("sFrame",sFrame)

    if isFound is True:
        cAudioOut.playSoundAsync("sound/clear.wav")
        cCtrlCard.write_result("pose", "T")
        dictArgument["Start time"] = cState.updateState("POSE_CORRECT")
        proc.closeWindows()
    elif isFound is False:
        cAudioOut.playSoundAsync("sound/gameover.wav")
        cCtrlCard.write_result("pose", "T")
        dictArgument["Start time"] = cState.updateState("POSE_WRONG")
        proc.closeWindows()


# POSE_CORRECTモード処理　======================================================
def procPose_correct(dictArgument):
    event = dictArgument["Event"]
    cState = dictArgument["State"]
    cCtrlCard = dictArgument["CtrlCard"]

    if event == "POSE_CORRECT":
        cCtrlCard.write_result("pose", "T")
        if cCtrlCard.checkComplete():
            dictArgument["Start time"] = cState.updateState("ENDING")
        else:
            dictArgument["Start time"] = cState.updateState("SELECT_GAME")
            cState.dictWindow["SELECT_GAME"]["プレイ"].update(disabled=True)


# POSE_WRONGモード処理　======================================================
def procPose_wrong(dictArgument):
    event = dictArgument["Event"]
    cState = dictArgument["State"]
    cCtrlCard = dictArgument["CtrlCard"]

    if event == "POSE_WRONG":
        cCtrlCard.write_result("pose", "T")
        if cCtrlCard.checkComplete():
            dictArgument["Start time"] = cState.updateState("ENDING")
        else:
            dictArgument["Start time"] = cState.updateState("SELECT_GAME")
            cState.dictWindow["SELECT_GAME"]["プレイ"].update(disabled=True)
