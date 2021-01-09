import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines


def file2matrix(filename):
	fr = open(filename)
	arrayOfLines = fr.readlines()
	numberOfLines = len(arrayOfLines)
	retMat = np.zeros((numberOfLines, 3))
	classLabelVector = []
	index = 0
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split('\t')
		retMat[index, :] = listFromLine[0: 3]
		if listFromLine[-1] == 'didntLike':
			classLabelVector.append(1)
		elif listFromLine[-1] == 'smallDoses':
			classLabelVector.append(2)
		elif listFromLine[-1] == 'largeDoses':
			classLabelVector.append(3)
		index += 1
	return retMat, classLabelVector


def showdatas(datingDataMat, datingLabels):
	fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
	numberOfLabels = len(datingLabels)
	LabelsColors = []
	for i in datingLabels:
		if i == 1:
			LabelsColors.append('black')
		if i == 2:
			LabelsColors.append('orange')
		if i == 3:
			LabelsColors.append('red')

	axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
	axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
	axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
	axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比')
	plt.setp(axs0_title_text, size=9, weight='bold', color='red')
	plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

	axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
	axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激凌公升数')
	axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
	axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激凌公升数')
	plt.setp(axs1_title_text, size=9, weight='bold', color='red')
	plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

	axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
	axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数')
	axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
	axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数')
	plt.setp(axs2_title_text, size=9, weight='bold', color='red')
	plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
	plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

	didntLike = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='didntLike')
	smallDoses = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='smallDoses')
	largeDoses = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='largeDoses')

	axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
	axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
	axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

	plt.show()


if __name__ == '__main__':
	filename = "datingTestSet.txt"
	datingDataMat, datingLabels = file2matrix(filename)
	showdatas(datingDataMat, datingLabels)

	