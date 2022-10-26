import cv2
import math

import numpy as np
import mediapipe as mp

from Classes.ClsImageProcess import ClsImageProcess
from Classes.ClsAudioOut import ClsAudioOut
from Classes.ClsLogger import ClsLogger
from functions.common import PlaySound


class ClsImageProcessPose(ClsImageProcess):
    def initProcess(self):
        self.isROIdefined = False
        self.ratioROI = 0.6
        self.frameCnt = 0
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # set audio out
        self.cLogger = ClsLogger()
        self.cAudioOut = ClsAudioOut(self.cLogger)

        # store past data
        self.pastFrameNum = 30
        self.pastLandmarks = [] * self.pastFrameNum
        self.pastPoses = [] * self.pastFrameNum
        self.previousPoseID = None

        # store enemy/player data
        self.clear = False
        self.gameover = False
        self.enemyHP = 100
        self.playerHP = 3

        # phase settings
        self.attackPhase = True
        self.phaseCnt = 0
        self.attackPhaseMax = 10
        self.defenceCheckPhase = False
        self.defensePhaseMax = 5
        self.defensePatternIDs = []
        self.initDefencePattern()

        # set overlay
        self.imOverlayEnemy = self.loadOverlayImage("./images/bg_conv.png")
        self.imOverlayEnemyRage = self.changeHue(self.imOverlayEnemy, 180)
        self.imOverlayMaskEnemy = self.makeOverlayMask(self.imOverlayEnemy)
        self.setOverlayCenter(self.imOverlayEnemy, self.imOverlayMaskEnemy)

    def loadOverlayImage(self, path: str) -> cv2.Mat:
        return cv2.imread(path, -1)

    def makeOverlayMask(self, imOverlay: cv2.Mat) -> cv2.Mat:
        imOverlayMask = imOverlay[:, :, 3]
        imOverlayMask = cv2.cvtColor(
            imOverlayMask, cv2.COLOR_GRAY2BGR)
        return imOverlayMask / 255

    def setOverlayCenter(self, imOverlay: cv2.Mat, imOverlayMask: cv2.Mat,
                         width: int = 1024, height: int = 600, dy: int = 0):
        imOverlay = imOverlay[:, :, :3]
        h, w = imOverlayMask.shape[0], imOverlayMask.shape[1]
        self.window.setEnableOverlay(
            True, int(width / 2 - w / 2), int(height / 2 - h / 2) + dy)
        self.window.setOverlayImage(
            imOverlay, imOverlayMask)

    def setRatioROI(self, ratioROI: float):
        self.ratioROI = ratioROI

    def defineROI(self, img: cv2.Mat):
        width = int(img.shape[1] * self.ratioROI)
        self.leftPosROI = int((img.shape[1] - width) / 2)
        self.rightPosROI = img.shape[1] - self.leftPosROI
        self.isROIdefined = True

    def drawCircle(self, x: int, y: int, r: int):
        cv2.circle(self.imSensor, (x, y), int(r), (255, 0, 0),
                   1, lineType=cv2.LINE_8, shift=0)

    def putText(self, text: str, x: int, y: int):
        cv2.putText(
            self.imSensor, text,
            (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 1)

    def calcDegree(self, x1: int, y1: int, x2: int, y2: int) -> float:
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        vec = b - a
        radian = np.arctan2(vec[0], vec[1])
        return np.rad2deg(radian)

    def calcDistance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        return np.linalg.norm(a-b)

    def judgeBodyDegree(self, vPoints: list, LR: str) -> bool:
        """
        LR : "left" or "right" の 文字列を入れる
        """
        # define point
        left_shoulder = vPoints[11]
        right_shoulder = vPoints[12]

        # judge logic
        shoulder_deg = self.calcDegree(right_shoulder[0], right_shoulder[1],
                                       left_shoulder[0], left_shoulder[1])

        if LR == "left" and shoulder_deg < 75:
            return True
        elif LR == "right" and shoulder_deg > 105:
            return True
        return False

    def judgePunch(self, vPoints: list, LR: str) -> bool:
        """
        LR : "left" or "right" の 文字列を入れる
        """
        # define point
        left_shoulder = vPoints[11]
        right_shoulder = vPoints[12]
        left_elbow = vPoints[13]
        right_elbow = vPoints[14]
        left_wrist = vPoints[15]
        right_wrist = vPoints[16]
        left_hip = vPoints[23]
        right_hip = vPoints[24]

        # judge logic
        body_height = self.calcDistance(
            (left_shoulder[0] + right_shoulder[0]) /
            2, (left_shoulder[1] + right_shoulder[1]) / 2,
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2)

        if (LR == "left" and
            (left_elbow[0] - left_wrist[0]) ** 2 + (left_elbow[1] - left_wrist[1]) ** 2 < (body_height / 3) ** 2) and \
                left_wrist[1] < left_elbow[1]:
            return True
        elif (LR == "right" and
              (right_elbow[0] - right_wrist[0]) ** 2 + (right_elbow[1] - right_wrist[1]) ** 2 < (body_height / 3) ** 2) and \
                right_wrist[1] < right_elbow[1]:
            return True
        return False

    def judgeGuard(self, vPoints: list) -> bool:
        # define point
        left_shoulder = vPoints[11]
        right_shoulder = vPoints[12]
        left_elbow = vPoints[13]
        left_wrist = vPoints[15]
        right_elbow = vPoints[14]
        right_wrist = vPoints[16]

        # judge logic
        ra = np.array([right_wrist[0] - right_elbow[0],
                       right_wrist[1] - right_elbow[1]])
        rb = np.array([right_shoulder[0] - right_elbow[0],
                       right_shoulder[1] - right_elbow[1]])
        right_cos = np.inner(ra, rb) / \
            (np.linalg.norm(ra) * np.linalg.norm(rb))

        la = np.array([left_wrist[0] - left_elbow[0],
                       left_wrist[1] - left_elbow[1]])
        lb = np.array([left_shoulder[0] - left_elbow[0],
                       left_shoulder[1] - left_elbow[1]])
        left_cos = np.inner(la, lb) / \
            (np.linalg.norm(la) * np.linalg.norm(lb))

        if (0.9 > right_cos > 0.2) & (0.9 > left_cos > 0.2) & \
                (right_wrist[0] > left_wrist[0]):
            return True
        return False

    def judgeHeal(self, vPoints: list) -> bool:
        # define point
        left_shoulder = vPoints[11]
        right_shoulder = vPoints[12]
        left_wrist = vPoints[15]
        right_wrist = vPoints[16]
        left_thumb = vPoints[21]
        right_thumb = vPoints[22]

        # judge logic
        length1 = abs(left_shoulder[0]-right_shoulder[0]) / 3
        length2 = abs(left_shoulder[0]-right_shoulder[0]) / 5

        if ((right_wrist[0] - left_wrist[0]) ** 2 + (right_wrist[1] - left_wrist[1]) ** 2 < length1 ** 2) and \
                ((right_thumb[0] - left_thumb[0]) ** 2 + (right_thumb[1] - left_thumb[1]) ** 2 < length2 ** 2):
            return True
        return False

    def judgeUpperPunch(self, vPoints: list, LR: str) -> bool:
        # define point
        nose = vPoints[0]
        left_shoulder = vPoints[11]
        right_shoulder = vPoints[12]
        left_elbow = vPoints[13]
        left_wrist = vPoints[15]
        right_elbow = vPoints[14]
        right_wrist = vPoints[16]
        left_thumb = vPoints[21]
        right_thumb = vPoints[22]

        # judge logic
        ra = np.array([right_wrist[0] - right_elbow[0],
                       right_wrist[1] - right_elbow[1]])
        rb = np.array([right_shoulder[0] - right_elbow[0],
                       right_shoulder[1] - right_elbow[1]])
        right_cos = np.inner(ra, rb) / (np.linalg.norm(ra)
                                        * np.linalg.norm(rb))

        la = np.array([left_wrist[0] - left_elbow[0],
                       left_wrist[1] - left_elbow[1]])
        lb = np.array([left_shoulder[0] - left_elbow[0],
                       left_shoulder[1] - left_elbow[1]])
        left_cos = np.inner(la, lb) / \
            (np.linalg.norm(la) * np.linalg.norm(lb))

        if (LR == "left"
                and ((left_thumb[1] < nose[1]) & (-0.86 < left_cos < 0.86))):
            return True
        elif (LR == "right"
              and ((right_thumb[1] < nose[1]) & (-0.86 < right_cos < 0.86))):
            return True
        return False

    def judgeAvoidUnder(self, vPoints: list) -> bool:
        # define point
        left_shoulder = vPoints[11]
        right_shoulder = vPoints[12]
        left_hip = vPoints[23]
        right_hip = vPoints[24]
        left_knee = vPoints[25]
        right_knee = vPoints[26]

        # judge logic
        body_height = self.calcDistance(
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2)

        if (((left_knee[0] - left_hip[0]) ** 2 + (left_knee[1] - left_hip[1]) ** 2 < (body_height * 2 / 3) ** 2) and
                ((right_knee[0] - right_hip[0]) ** 2 + (right_knee[1] - right_hip[1]) ** 2 < (body_height * 2 / 3) ** 2)):
            return True
        return False

    def judgePose(self, pose_id, past_frame=5):
        for poses in self.pastPoses[:past_frame]:
            if poses[pose_id] is True:
                return True
        return False

    @staticmethod
    def changeHue(imOrig: cv2.Mat, hue: int) -> cv2.Mat:
        """
        imOrig: 変換対象の画像
        hue: 色相の値(0~359)
         - 赤: 180
        """
        imHSV = cv2.cvtColor(
            imOrig, cv2.COLOR_BGR2HSV)
        imHSV[:, :, (0)] = hue
        return cv2.cvtColor(
            imHSV, cv2.COLOR_HSV2BGR)

    @staticmethod
    def incrementHue(imOrig: cv2.Mat, dHue: int) -> cv2.Mat:
        """
        imOrig: 変換対象の画像
        dhue: 加算する色相の値(0~359)
        """
        imHSV = cv2.cvtColor(
            imOrig, cv2.COLOR_BGR2HSV)
        imHSV[:, :, (0)] = imHSV[:, :, (0)] + dHue
        return cv2.cvtColor(
            imHSV, cv2.COLOR_HSV2BGR)

    def initDefencePattern(self):
        self.defensePatternIDs = np.random.randint(
            5, 9, (1, self.defensePhaseMax)).tolist()[0]

    def process(self):
        debug = False

        # set ROI
        if self.isROIdefined is False:
            self.defineROI(self.imSensor)
        imROI = self.imSensor[:, self.leftPosROI:self.rightPosROI]

        # get landmarks
        imROI = cv2.cvtColor(imROI, cv2.COLOR_BGR2RGB)
        imROI.flags.writeable = False
        results = self.pose.process(imROI)
        imROI.flags.writeable = True
        imROI = cv2.cvtColor(imROI, cv2.COLOR_RGB2BGR)

        if not results.pose_landmarks:
            self.imProcessed = self.imSensor
            self.frameCnt += 1
            return 0

        # initialize currentPose array
        currentPose = [False, False, False,
                       False, False, False, False, False, False]

        # calc coord (ROI + landmarks)
        # vPoint[n][0]: x, vPoint[n][1]: y, vPoint[n][2]: visibility
        vPoints = [(int(landmark.x*imROI.shape[1]+self.leftPosROI),
                    int(landmark.y*imROI.shape[0]), landmark.visibility)
                   for landmark in results.pose_landmarks.landmark]

        # add landmarks to past landmarks records
        self.pastLandmarks.insert(0, vPoints)

        # delete past landmarks
        if len(self.pastLandmarks) >= self.pastFrameNum:
            del self.pastLandmarks[-1]

        # draw landmarks (debug is true only)
        # if debug:
        #     self.mp_drawing.draw_landmarks(
        #         self.imSensor, results.pose_landmarks,
        #         self.mp_pose.POSE_CONNECTIONS,
        #         connection_drawing_spec=self.mp_drawing.DrawingSpec(
        #             thickness=2, circle_radius=10))

        if self.attackPhase:
            # judge punch
            if self.judgePunch(vPoints, "left"):  # id: 0
                currentPose[0] = True
            if self.judgePunch(vPoints, "right"):  # id: 1
                currentPose[1] = True

            # judge upper punch
            if self.judgeUpperPunch(vPoints, "left"):  # id: 2
                currentPose[2] = True
            if self.judgeUpperPunch(vPoints, "right"):  # id: 3
                currentPose[3] = True

            # judge heal
            if self.judgeHeal(vPoints):  # id: 4
                currentPose[4] = True

            if debug:
                cv2.putText(self.imSensor, "Attack Phase", (10, 60),
                            cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
        else:
            # judge shoulder degree
            if self.judgeBodyDegree(vPoints, "left"):  # id: 5
                currentPose[5] = True
            if self.judgeBodyDegree(vPoints, "right"):  # id: 6
                currentPose[6] = True

            # judge guard
            if self.judgeGuard(vPoints):  # id: 7
                currentPose[7] = True

            # judge avoid under
            if self.judgeAvoidUnder(vPoints):  # id: 8
                currentPose[8] = True

            if debug:
                cv2.putText(self.imSensor, "Defence Phase", (10, 60),
                            cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)

        # Attach phase
        if self.attackPhase and self.frameCnt % 5 == 0:
            # punch (R)
            if self.judgePose(0) and self.previousPoseID not in (0, 2):
                self.previousPoseID = 0
                self.enemyHP -= 2
                self.phaseCnt += 1
            # punch (L)
            elif self.judgePose(1) and self.previousPoseID not in (1, 3):
                self.previousPoseID = 1
                self.enemyHP -= 2
                self.phaseCnt += 1
            # upper punch (R)
            elif self.judgePose(2) and self.previousPoseID != 2:
                self.previousPoseID = 2
                self.enemyHP -= 3
                self.phaseCnt += 1
            # upper punch (L)
            elif self.judgePose(3) and self.previousPoseID != 3:
                self.previousPoseID = 3
                self.enemyHP -= 3
                self.phaseCnt += 1
            # heal (hidden)
            elif self.judgePose(4) and self.previousPoseID != 4:
                self.previousPoseID = 4
                self.playerHP += 1

            # exit attach phase process
            if self.phaseCnt == self.attackPhaseMax:
                self.cAudioOut.playSoundAsync("sound/do_defence.wav")
                self.phaseCnt = 0
                self.attackPhase = False
                self.setOverlayCenter(
                    self.imOverlayEnemyRage, self.imOverlayMaskEnemy, dy=0)

        # Guard phase
        if not self.attackPhase and self.frameCnt % 50 == 0:
            currentDefencePose = self.defensePatternIDs[self.phaseCnt]
            if self.defenceCheckPhase is False:
                if currentDefencePose == 5:
                    self.cAudioOut.playSoundAsync("sound/dodge_right.wav")
                elif currentDefencePose == 6:
                    self.cAudioOut.playSoundAsync("sound/dodge_left.wav")
                elif currentDefencePose == 7:
                    self.cAudioOut.playSoundAsync("sound/guard.wav")
                elif currentDefencePose == 8:
                    self.cAudioOut.playSoundAsync("sound/dodge_under.wav")
                self.defenceCheckPhase = True
            else:
                if self.judgePose(5, past_frame=15) and currentDefencePose == 5:  # right
                    self.cAudioOut.playSoundAsync("sound/correct_24.wav")
                    self.previousPoseID = 5
                elif self.judgePose(6, past_frame=15) and currentDefencePose == 6:  # left
                    self.cAudioOut.playSoundAsync("sound/correct_24.wav")
                    self.previousPoseID = 6
                elif self.judgePose(7, past_frame=15) and currentDefencePose == 7:  # guard
                    self.cAudioOut.playSoundAsync("sound/correct_24.wav")
                    self.previousPoseID = 7
                elif self.judgePose(8, past_frame=15) and currentDefencePose == 8:  # avoid under
                    self.cAudioOut.playSoundAsync("sound/correct_24.wav")
                    self.previousPoseID = 8
                else:
                    self.cAudioOut.playSoundAsync("sound/wrong_24.wav")
                    self.playerHP -= 1
                self.phaseCnt += 1
                self.defenceCheckPhase = False

            # exit attach phase process
            if self.phaseCnt == self.defensePhaseMax:
                self.cAudioOut.playSoundAsync("sound/do_attack.wav")
                self.phaseCnt = 0
                self.attackPhase = True
                self.initDefencePattern()
                self.setOverlayCenter(
                    self.imOverlayEnemy, self.imOverlayMaskEnemy, dy=0)

        # draw enemy hp
        cv2.putText(self.imSensor, str(self.enemyHP), (10, 30),
                    cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        # draw player hp
        cv2.putText(self.imSensor, str(self.playerHP), (320 - 30, 30),
                    cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        if debug:
            poseName = ""
            if self.previousPoseID == 0:
                poseName = "P (R)"
            elif self.previousPoseID == 1:
                poseName = "P (L)"
            elif self.previousPoseID == 2:
                poseName = "UP (R)"
            elif self.previousPoseID == 3:
                poseName = "UP (L)"
            elif self.previousPoseID == 4:
                poseName = "HE"
            elif self.previousPoseID == 5:
                poseName = "A (R)"
            elif self.previousPoseID == 6:
                poseName = "A (L)"
            elif self.previousPoseID == 7:
                poseName = "G"
            elif self.previousPoseID == 8:
                poseName = "A (UN)"
            cv2.putText(self.imSensor, poseName, (320 - 60, 50),
                        cv2.FONT_ITALIC, 0.5, (0, 0, 255), 1)

            left_shoulder = vPoints[11]
            right_shoulder = vPoints[12]
            shoulder_deg = self.calcDegree(right_shoulder[0], right_shoulder[1],
                                           left_shoulder[0], left_shoulder[1])
            cv2.putText(self.imSensor, str(round(shoulder_deg, 1)), (320 - 60, 70),
                        cv2.FONT_ITALIC, 0.5, (0, 0, 255), 1)

        # add pose
        self.pastPoses.insert(0, currentPose)

        # delete past pose
        if len(self.pastPoses) >= self.pastFrameNum:
            del self.pastPoses[-1]

        # judge clear
        if self.enemyHP <= 0:
            self.clear = True

        # judge gameover
        if self.playerHP <= 0:
            self.gameover = True

        # clear/initialize process
        if self.clear is True:
            self.clear = False
            self.enemyHP = 100
            self.playerHP = 3
            self.frameCnt = 0
            return True

        # gameover/initialize process
        if self.gameover is True:
            self.gameover = False
            self.enemyHP = 100
            self.playerHP = 3
            self.frameCnt = 0
            return False

        self.imProcessed = self.imSensor
        self.frameCnt += 1

        return 0


if __name__ == '__main__':
    CProc = ClsImageProcessPose
    import os

    if os.name == 'nt':
        strPlatform = 'WIN'
    else:
        strPlatform = 'JETSON'

    sCameraNumber = 0
    sSensoright_wristidth = 320
    sSensorHeight = 180
    sMonitoright_wristidth = 1024
    sMonitorHeight = 600
    tpleft_wristindowName = ('full',)
    sFlipMode = 1

    proc = CProc(
        strPlatform,
        sCameraNumber,
        sSensoright_wristidth,
        sSensorHeight,
        sMonitoright_wristidth,
        sMonitorHeight,
        tpleft_wristindowName,
        sFlipMode)

    proc.createWindows()

    while True:
        proc.execute()
        sKey = cv2.waitKey(1) & 0xFF
        if sKey == ord('q'):
            del proc
            break
