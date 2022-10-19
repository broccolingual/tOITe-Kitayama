import cv2
import math
import mediapipe as mp

from Classes.ClsImageProcess import ClsImageProcess


class ClsImageProcessPose(ClsImageProcess):
    def initProcess(self):
        self.isROIdefined = False
        self.ratioROI = 1
        self.end = False
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.pastFrameNum = 60
        self.pastLandmarks = [] * self.pastFrameNum

        # imOverlayOrig_inst = cv2.imread('./images/sign_inst2.png', -1)
        # self.imOverlayMask_inst = imOverlayOrig_inst[:, :, 3]
        # self.imOverlayMask_inst = cv2.cvtColor(
        #     self.imOverlayMask_inst, cv2.COLOR_GRAY2BGR)
        # self.imOverlayMask_inst = self.imOverlayMask_inst / 255
        # self.imOverlayOrig_inst = imOverlayOrig_inst[:, :, :3]
        # self.window.setEnableOverlay(True, 300, 0)
        # self.window.setOverlayImage(self.imOverlayOrig_inst, self.imOverlayMask_inst)

        # imOverlayOrig_correct = cv2.imread(
        #     './images/sign_correct_cyan.png', -1)
        # self.imOverlayMask_correct = imOverlayOrig_correct[:, :, 3]
        # self.imOverlayMask_correct = cv2.cvtColor(
        #     self.imOverlayMask_correct, cv2.COLOR_GRAY2BGR)
        # self.imOverlayMask_correct = self.imOverlayMask_correct / 255
        # self.imOverlayOrig_correct = imOverlayOrig_correct[:, :, :3]
        # self.window.setEnableOverlay(True, 0, 0)
        # self.window.setOverlayImage(self.imOverlayOrig_correct, self.imOverlayMask_inst)

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

        # for debug
        # self.drawCircle(left_wrist[0], left_wrist[1], body_height/3)
        # self.drawCircle(
        #     right_wrist[0], right_wrist[1], body_height/3)

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
        rel_cos = ((right_wrist[0] - right_elbow[0]) * (right_shoulder[0] - right_elbow[0]) +
                   (right_wrist[1] - right_elbow[1]) * (right_shoulder[1] - right_elbow[1])) / (math.sqrt((right_wrist[0] - right_elbow[0]) ** 2 + (right_wrist[1] - right_elbow[1]) ** 2) * math.sqrt((right_shoulder[0] - right_elbow[0]) ** 2 + (right_shoulder[1] - right_elbow[1]) ** 2))
        lel_cos = ((left_wrist[0] - left_elbow[0]) * (left_shoulder[0] - left_elbow[0]) +
                   (left_wrist[1] - left_elbow[1]) * (left_shoulder[1] - left_elbow[1])) / (math.sqrt((left_wrist[0] - left_elbow[0]) ** 2 + (left_wrist[1] - left_elbow[1]) ** 2) * math.sqrt((left_shoulder[0] - left_elbow[0]) ** 2 + (left_shoulder[1] - left_elbow[1]) ** 2))

        if (0.9 > rel_cos > 0.2) & (0.9 > lel_cos > 0.2) & (right_wrist[0] > left_wrist[0]):
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

    def process(self):
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
            # x座標に切り抜いた左側の位置を足し合わす
            vPoints = [(int(landmark.x*imROI.shape[1]+self.leftPosROI),
                        int(landmark.y*imROI.shape[0]))
                       for landmark in results.pose_landmarks.landmark]

            # add past frame
            self.pastLandmarks.insert(0, vPoints)

            # delete past frame
            if len(self.pastLandmarks) >= self.pastFrameNum:
                del self.pastLandmarks[-1]

            # draw landmarks
            self.mp_drawing.draw_landmarks(
                self.imSensor, results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    thickness=2, circle_radius=10))

            # judge shoulder degree
            if self.judgeBodyDegree(vPoints, "left"):
                self.putText("body left", 20, 40)
            elif self.judgeBodyDegree(vPoints, "right"):
                self.putText("body right", 20, 60)

            # judge punch
            if self.judgePunch(vPoints, "left"):
                self.putText("punch left", 20, 80)
            elif self.judgePunch(vPoints, "right"):
                self.putText("punch right", 20, 100)

            # judge guard
            if self.judgeGuard(vPoints):
                self.putText("guard", 20, 120)

            # judge heal
            if self.judgeHeal(vPoints):
                self.putText("heal", 20, 140)

            # end
            if self.end is True:
                self.end = False
                return True

        # 正解の時はreturn Trueする
        self.imProcessed = self.imSensor

        return 0


if __name__ == '__main__':
    CProc = Cleft_shoulderImageProcessPose
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
