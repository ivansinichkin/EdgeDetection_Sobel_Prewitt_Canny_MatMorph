import sys
# Импортируем наш интерфейс
from EdgeDetection_ui import *
from conv import *
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap
from PIL import Image, ImageDraw
import numpy as np
import os.path
import cv2


class MyWin():
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_EdgeDetector()
        self.ui.setupUi(self.main_win)

        self.ui.pushButtonSelectImage.clicked.connect(self.selectImage)
        self.ui.pushButtonDetectSobel.clicked.connect(self.detectSobel)
        self.ui.pushButtonDetectPrewitt.clicked.connect(self.detectPrewitt)
        self.ui.pushButtonDetectMatMorph.clicked.connect(self.detectMatMorph)
        self.ui.pushButtonDetectCanny.clicked.connect(self.detectCanny)
        self.ui.pushButtonDetectCannyCV.clicked.connect(self.detectCannyCV)

    def show(self):
        self.main_win.show()

    def selectImage(self):
        path = QFileDialog.getOpenFileName()[0]
        relpath = os.path.relpath(path)
        image = Image.open(relpath)
        image.save("process1.jpg", "JPEG")
        pixmap = QPixmap("process1.jpg")
        myScaledPixmap = pixmap.scaled(320, 320, QtCore.Qt.KeepAspectRatio)
        self.ui.labelOrig.setPixmap(myScaledPixmap)
        self.ui.labelOrig.repaint()
        QApplication.processEvents()

    def detectSobel(self):
        img = cv2.imread('process1.jpg', cv2.IMREAD_GRAYSCALE)
        sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobGx = conv(img, sobelX)
        sobGy = conv(img, sobelY)
        sob_out = np.sqrt(np.power(sobGx, 2) + np.power(sobGy, 2))
        cv2.imwrite('SobelEdges.jpg', sob_out)
        pixmap = QPixmap("SobelEdges.jpg")
        myScaledPixmap = pixmap.scaled(320, 320, QtCore.Qt.KeepAspectRatio)
        self.ui.labelSob.setPixmap(myScaledPixmap)
        self.ui.labelSob.repaint()
        QApplication.processEvents()

    def detectPrewitt(self):
        img = cv2.imread('process1.jpg', cv2.IMREAD_GRAYSCALE)
        prewittX = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewittY = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewittGx = conv(img, prewittX)
        prewittGy = conv(img, prewittY)
        prew_out = np.sqrt(np.power(prewittGx, 2) + np.power(prewittGy, 2))
        cv2.imwrite('PrewittEdges.jpg', prew_out)
        pixmap = QPixmap("PrewittEdges.jpg")
        myScaledPixmap = pixmap.scaled(320, 320, QtCore.Qt.KeepAspectRatio)
        self.ui.labelPrew.setPixmap(myScaledPixmap)
        self.ui.labelPrew.repaint()
        QApplication.processEvents()

    def detectMatMorph(self):
        struct = np.array([[0, 0, 255, 255, 255, 0, 0],
                           [0, 255, 255, 255, 255, 255, 0],
                           [255, 255, 255, 255, 255, 255, 255],
                           [255, 255, 255, 255, 255, 255, 255],
                           [255, 255, 255, 255, 255, 255, 255],
                           [0, 255, 255, 255, 255, 255, 0],
                           [0, 0, 255, 255, 255, 0, 0]]).astype('int')
        img = cv2.imread('process1.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.medianBlur(img, 5)
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        dilation = my_dilation(th, struct)
        erosion = my_erosion(th, struct)
        edge = np.subtract(dilation, erosion)
        cv2.imwrite('matMorphEdges.jpg', edge)
        pixmap = QPixmap("matMorphEdges.jpg")
        myScaledPixmap = pixmap.scaled(320, 320, QtCore.Qt.KeepAspectRatio)
        self.ui.labelMatMorph.setPixmap(myScaledPixmap)
        self.ui.labelMatMorph.repaint()
        QApplication.processEvents()

    def detectCanny(self):
        sigma = float(self.ui.lineEdit_sigma.text())
        downth = float(self.ui.lineEdit_downth.text())
        topth = float(self.ui.lineEdit_topth.text())
        img = cv2.imread('process1.jpg', cv2.IMREAD_GRAYSCALE)

        blur = conv(img, gaussian_kernel(5, sigma))
        grad = gradients(blur)
        supr = non_max_suppression(grad[0], grad[1])
        img_thresh, weak, strong = threshold(supr, downth, topth)
        img_final = hysteresis(img_thresh, weak, strong)

        cv2.imwrite('CannyEdges.jpg', img_final)
        pixmap = QPixmap("CannyEdges.jpg")
        myScaledPixmap = pixmap.scaled(320, 320, QtCore.Qt.KeepAspectRatio)
        self.ui.labelCanny.setPixmap(myScaledPixmap)
        self.ui.labelCanny.repaint()
        QApplication.processEvents()

    def detectCannyCV(self):
        kernel = int(self.ui.lineEdit_kernelCV.text())
        downth = float(self.ui.lineEdit_downthCV.text())
        topth = float(self.ui.lineEdit_topthCV.text())
        img = cv2.imread('process1.jpg', cv2.IMREAD_GRAYSCALE)

        img_blur = cv2.blur(img, (3, 3))
        edges = cv2.Canny(img_blur, downth, topth, apertureSize=5)

        cv2.imwrite('CannyCVEdges.jpg', edges)
        pixmap = QPixmap("CannyCVEdges.jpg")
        myScaledPixmap = pixmap.scaled(320, 320, QtCore.Qt.KeepAspectRatio)
        self.ui.labelCannyCV.setPixmap(myScaledPixmap)
        self.ui.labelCannyCV.repaint()
        QApplication.processEvents()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    sys.exit(app.exec_())