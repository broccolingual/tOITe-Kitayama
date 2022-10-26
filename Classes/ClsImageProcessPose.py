import cv2
import math
import mediapipe as mp

from Classes.ClsImageProcess import ClsImageProcess
from functions.common import PlaySound


class ClsImageProcessPose(ClsImageProcess):
    def initProcess(self):
        self.isROIdefined = False
        self.ratioROI = 0.6
        self.end = False
        self.frameCnt = 0
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # store past data
        self.pastFrameNum = 30
        self.pastLandmarks = [] * self.pastFrameNum
        self.pastPoses = [] * self.pastFrameNum
        self.previousPoseID = None

        # store enemy/player data
        self.enemyHP = 100
        self.playerHP = 3

        # set overlay
        self.imOverlayEnemy = self.loadOverlayImage("./images/enemy.png")
        self.imOverlayMaskEnemy = self.makeOverlayMask(self.imOverlayEnemy)
        self.setOverlayCenter(self.imOverlayEnemy, self.imOverlayMaskEnemy)

    def loadOverlayImage(self, path):
        return cv2.imread(path, -1)

    def makeOverlayMask(self, imOverlay):
        imOverlayMask = imOverlay[:, :, 3]
        imOverlayMask = cv2.cvtColor(
            imOverlayMask, cv2.COLOR_GRAY2BGR)
        return imOverlayMask / 255

    def setOverlayCenter(self, imOverlay, imOverlayMask, width=1024, height=600, dy=0):
        imOverlay = imOverlay[:, :, :3]
        h, w = imOverlayMask.shape[0], imOverlayMask.shape[1]
        self.window.setEnableOverlay(
            True, int(width / 2 - w / 2), int(height / 2 - h / 2) + dy)
        self.window.setOverlayImage(
            imOverlay, imOverlayMask)

    def setRatioROI(self, ratioROI):
        self.ratioROI = ratioROI

    def defineROI(self, img):
        width = int(img.shape[1] * self.ratioROI)
        self.leftPosROI = int((img.shape[1] - width) / 2)
        self.rightPosROI = img.shape[1] - self.leftPosROI
        self.isROIdefined = True

    def drawCircle(self, x, y, r):
        cv2.circle(self.imSensor, (x, y), int(r), (255, 0, 0),
                   1, lineType=cv2.LINE_8, shift=0)

    def putText(self, text, x, y):
        cv2.putText(
            self.imSensor, text,
            (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 1)

    def calcDegree(self, x1, y1, x2, y2):
        radian = math.atan2(y2-y1, x2-x1)
        return radian * 180 / math.pi

    def calcDistance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

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
        if LR == "left" and shoulder_deg > 15:
            return True
        elif LR == "right" and shoulder_deg < -15:
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
                (left_elbow[0] - left_wrist[0]) ** 2 + (left_elbow[1] - left_wrist[1]) ** 2 < (body_height / 3) ** 2) and left_wrist[1] < left_elbow[1]:
            return True
        elif (LR == "right" and
              (right_elbow[0] - right_wrist[0]) ** 2 + (right_elbow[1] - right_wrist[1]) ** 2 < (body_height / 3) ** 2) and right_wrist[1] < right_elbow[1]:
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
        # TODO: ZeroDivision Errorの修正
        r_cos = ((right_wrist[0] - right_elbow[0]) * (right_shoulder[0] - right_elbow[0]) +
                 (right_wrist[1] - right_elbow[1]) * (right_shoulder[1] - right_elbow[1])) / (math.sqrt((right_wrist[0] - right_elbow[0]) ** 2 + (right_wrist[1] - right_elbow[1]) ** 2) * math.sqrt((right_shoulder[0] - right_elbow[0]) ** 2 + (right_shoulder[1] - right_elbow[1]) ** 2))
        l_cos = ((left_wrist[0] - left_elbow[0]) * (left_shoulder[0] - left_elbow[0]) +
                 (left_wrist[1] - left_elbow[1]) * (left_shoulder[1] - left_elbow[1])) / (math.sqrt((left_wrist[0] - left_elbow[0]) ** 2 + (left_wrist[1] - left_elbow[1]) ** 2) * math.sqrt((left_shoulder[0] - left_elbow[0]) ** 2 + (left_shoulder[1] - left_elbow[1]) ** 2))

        if (0.9 > r_cos > 0.2) & (0.9 > l_cos > 0.2) & (right_wrist[0] > left_wrist[0]):
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
        if ((right_wrist[0]-left_wrist[0])**2+(right_wrist[1]-left_wrist[1])**2 < length1**2) and ((right_thumb[0]-left_thumb[0])**2+(right_thumb[1]-left_thumb[1])**2 < length2**2):
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
        r_cos = ((right_wrist[0] - right_elbow[0]) * (right_shoulder[0] - right_elbow[0]) +
                 (right_wrist[1] - right_elbow[1]) * (right_shoulder[1] - right_elbow[1])) / (math.sqrt((right_wrist[0] - right_elbow[0]) ** 2 + (right_wrist[1] - right_elbow[1]) ** 2) * math.sqrt((right_shoulder[0] - right_elbow[0]) ** 2 + (right_shoulder[1] - right_elbow[1]) ** 2))
        l_cos = ((left_wrist[0] - left_elbow[0]) * (left_shoulder[0] - left_elbow[0]) +
                 (left_wrist[1] - left_elbow[1]) * (left_shoulder[1] - left_elbow[1])) / (math.sqrt((left_wrist[0] - left_elbow[0]) ** 2 + (left_wrist[1] - left_elbow[1]) ** 2) * math.sqrt((left_shoulder[0] - left_elbow[0]) ** 2 + (left_shoulder[1] - left_elbow[1]) ** 2))

        if (LR == "left"
                and ((left_thumb[1] < nose[1]) and (-0.86 < l_cos < 0.86))):
            return True
        elif (LR == "right"
              and ((right_thumb[1] < nose[1]) and (-0.86 < r_cos < 0.86))):
            return True
        return False

    def judgeAvoidUnder(self, vPoints: list) -> bool:
        left_shoulder = vPoints[11]
        right_shoulder = vPoints[12]
        left_hip = vPoints[23]
        right_hip = vPoints[24]
        left_knee = vPoints[25]
        right_knee = vPoints[26]
        body_height = self.calcDistance(
            (left_shoulder[0] + right_shoulder[0]) /
            2, (left_shoulder[1] + right_shoulder[1]) / 2,
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2)
        if (((left_knee[0] - left_hip[0]) ** 2 + (left_knee[1] - left_hip[1]) ** 2 < (body_height * 2 / 3) ** 2)
                and ((right_knee[0] - right_hip[0]) ** 2 + (right_knee[1] - right_hip[1]) ** 2 < (body_height * 2 / 3) ** 2)):
            return True
        return False

    def judgePose(self, pose_id, past_frame=5):
        for poses in self.pastPoses[:past_frame]:
            if poses[pose_id] is True:
                return True
        return False

    def changeHue(self, imOrig, hue):
        """
        imOrig: 変換対象の画像
        hue: 色相の値(0~359)
         - 赤: 90, 黄緑: 180, 水色: 270
        """
        imHSV = cv2.cvtColor(
            imOrig, cv2.COLOR_BGR2HSV)
        imHSV[:, :, (0)] = hue
        return cv2.cvtColor(
            imHSV, cv2.COLOR_HSV2BGR)

    def incrementHue(self, imOrig, dHue):
        """
        imOrig: 変換対象の画像
        dhue: 加算する色相の値(0~359)
        """
        imHSV = cv2.cvtColor(
            imOrig, cv2.COLOR_BGR2HSV)
        imHSV[:, :, (0)] = imHSV[:, :, (0)] + dHue
        return cv2.cvtColor(
            imHSV, cv2.COLOR_HSV2BGR)

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

        if results.pose_landmarks:
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
            if debug:
                self.mp_drawing.draw_landmarks(
                    self.imSensor, results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        thickness=2, circle_radius=10))

            # judge shoulder degree
            if self.judgeBodyDegree(vPoints, "left"):  # id: 0
                currentPose[0] = True
            if self.judgeBodyDegree(vPoints, "right"):  # id: 1
                currentPose[1] = True

            # judge punch
            if self.judgePunch(vPoints, "left"):  # id: 2
                currentPose[2] = True
            if self.judgePunch(vPoints, "right"):  # id: 3
                currentPose[3] = True

            # judge guard
            if self.judgeGuard(vPoints):  # id: 4
                currentPose[4] = True

            # judge heal
            if self.judgeHeal(vPoints):  # id: 5
                currentPose[5] = True

            # judge avoid under
            if self.judgeAvoidUnder(vPoints):  # id: 6
                currentPose[6] = True

            # judge upper punch
            if self.judgeUpperPunch(vPoints, "left"):  # id: 7
                currentPose[7] = True
            if self.judgeUpperPunch(vPoints, "right"):  # id: 8
                currentPose[8] = True

            # draw enemy hp
            cv2.putText(self.imSensor, str(self.enemyHP), (10, 40),
                        cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            if self.frameCnt % 5 == 0:
                if self.judgePose(2) and self.previousPoseID != 2:
                    self.enemyHP -= 2
                    self.previousPoseID = 2
                    self.imOverlayEnemy = self.changeHue(
                        self.imOverlayEnemy, 90)
                    self.setOverlayCenter(
                        self.imOverlayEnemy, self.imOverlayMaskEnemy, dy=20)
                elif self.judgePose(3) and self.previousPoseID != 3:
                    self.enemyHP -= 2
                    self.previousPoseID = 3
                    self.imOverlayEnemy = self.changeHue(
                        self.imOverlayEnemy, 90)
                    self.setOverlayCenter(
                        self.imOverlayEnemy, self.imOverlayMaskEnemy, dy=20)
                elif self.judgePose(4) and self.previousPoseID != 4:  # guard
                    PlaySound("./sound/guard.wav")
                    self.previousPoseID = 4
                    self.imOverlayEnemy = self.changeHue(
                        self.imOverlayEnemy, 90)
                elif self.judgePose(5) and self.previousPoseID != 5:  # heal
                    PlaySound("./sound/heal.wav")
                    self.previousPoseID = 5
                    self.imOverlayEnemy = self.changeHue(
                        self.imOverlayEnemy, 90)
                elif self.judgePose(6) and self.previousPoseID != 6:  # avoid under
                    self.previousPoseID = 6
                    self.imOverlayEnemy = self.changeHue(
                        self.imOverlayEnemy, 90)
                elif self.judgePose(7) and self.previousPoseID != 7:  # upper punch (L)
                    self.enemyHP -= 4
                    self.previousPoseID = 7
                    self.imOverlayEnemy = self.changeHue(
                        self.imOverlayEnemy, 180)
                    self.setOverlayCenter(
                        self.imOverlayEnemy, self.imOverlayMaskEnemy, dy=100)
                elif self.judgePose(8) and self.previousPoseID != 8:  # upper punch (R)
                    self.enemyHP -= 4
                    self.previousPoseID = 8
                    self.imOverlayEnemy = self.changeHue(
                        self.imOverlayEnemy, 180)
                    self.setOverlayCenter(
                        self.imOverlayEnemy, self.imOverlayMaskEnemy, dy=100)

            else:
                self.imOverlayEnemy = self.changeHue(
                    self.imOverlayEnemy, 90)
                self.setOverlayCenter(
                    self.imOverlayEnemy, self.imOverlayMaskEnemy)

            # add pose
            self.pastPoses.insert(0, currentPose)

            # delete past pose
            if len(self.pastPoses) >= self.pastFrameNum:
                del self.pastPoses[-1]

            # brake img process loop
            if self.enemyHP <= 0:
                self.end = True

            # end/initialize process
            if self.end is True:
                self.end = False
                self.enemyHP = 100
                self.frameCnt = 0
                return True

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
