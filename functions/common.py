def getDictFlag():
	dictFlag = {
		"tutorial"		: "チュートリアル",
		"true_false"	: "まるばつ",
		"pose"			: "バトル",
		"hand"			: "影絵",
		"ar_labyrinth"	: "AR",
	}

	return dictFlag

	
# ゲームを初期化
def Reset_Game(dictArgument):
	sStartTime = dictArgument["State"].updateState("STANDBY")
	dictArgument["ImageProc"].reset()
	dictArgument["Event"] = None
	dictArgument["Values"] = None
	dictArgument["Frame"] = 0
	dictArgument["Start time"] = sStartTime


# ゲームをクリアしたかを判定
def CheckComplete(cCtrlCard, dictFlag):
	dictSaveData = cCtrlCard.read_result()

	blClear = True
	for key in dictFlag.keys():
		if dictSaveData[key] != "T":
			blClear = False
			break

	return blClear


def CheckTappedArea(vPosition, listArea):
	sTappedArea = -1
	for sAreaNumber in range(len(listArea)):
		sMinX = listArea[sAreaNumber][0]
		sMaxX = listArea[sAreaNumber][0] + listArea[sAreaNumber][2]
		sMinY = listArea[sAreaNumber][1]
		sMaxY = listArea[sAreaNumber][1] + listArea[sAreaNumber][3]

		if(vPosition.x > sMinX and vPosition.x < sMaxX 
		and vPosition.y > sMinY and vPosition.y < sMaxY):
			sTappedArea = sAreaNumber
			break

	return sTappedArea
			