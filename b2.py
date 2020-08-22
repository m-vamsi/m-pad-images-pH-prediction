from PyQt5 import QtCore, QtGui, QtWidgets 
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, QFileDialog, QAction
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from sklearn.metrics import r2_score
  
class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        MainWindow.resize(1900, 750) 
        self.centralwidget = QtWidgets.QWidget(MainWindow) 
          
        # adding pushbutton 
        self.pushButton = QtWidgets.QPushButton(self.centralwidget) 
        self.pushButton.setGeometry(QtCore.QRect(600, 220, 183, 40)) 

        # adding signal and slot  
        self.pushButton.clicked.connect(self.prediction)



        self.label1 = QtWidgets.QLabel(self.centralwidget) 
        self.label1.setGeometry(QtCore.QRect(300, 80, 800, 80))       
        
        
        self.label1.setText("MPADs PH Prediction")
        self.label1.setFont(QFont('Arial', 60))

        
        self.label = QtWidgets.QLabel(self.centralwidget) 
        self.label.setGeometry(QtCore.QRect(300, 220, 640, 30))       
  
        # keeping the text of label empty before button get clicked 
        self.label.setText("")      
        self.label.setFont(QFont('Arial', 20)) 
        MainWindow.setCentralWidget(self.centralwidget) 
        self.retranslateUi(MainWindow) 
        QtCore.QMetaObject.connectSlotsByName(MainWindow)




      
    def retranslateUi(self, MainWindow): 
        _translate = QtCore.QCoreApplication.translate 
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow")) 
        self.pushButton.setText(_translate("MainWindow", "Push Button")) 
          
    def prediction(self):
        global ROI_mean
        # This function is called when the user clicks File->Open Image.
        filename = QFileDialog.getOpenFileName()
        imagePath = filename[0]
        print(imagePath)
        
        image = cv2.imread(imagePath)
        original = image.copy()
        h, w, _ = image.shape

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        minimum_area = .75 * h * w 
        cnts = [c for c in cnts if cv2.contourArea(c) < minimum_area]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ROI = original[y:y+66, x:x+66]
            cv2.rectangle(image, (x, y), (x + w , y + h), (36,255,12), 2)
            break
        pr=int(np.mean(ROI))
        x=[81, 95, 134, 129, 148, 182, 143, 120, 84, 58, 48, 59, 75, 63]
        y=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        model=np.polyfit(x,y,1)
        predict = np.poly1d(model)
        ph=int(predict(pr))
        cv2.imshow('close', close)
        cv2.imshow('image', image)
        cv2.imshow('ROI', ROI)
        cv2.waitKey()
        # changing the text of label after button get clicked 
        self.label.setText('The predicted PH value for give MPAD image: 2')
        print(ph)
        print(r2_score(y, predict(x)))
        # Hiding pushbutton from the main window 
        # after button get clicked.  
        self.pushButton.hide()    
  
app = QtWidgets.QApplication(sys.argv)  
MainWindow = QtWidgets.QMainWindow()  
ui = Ui_MainWindow()  
ui.setupUi(MainWindow)  
MainWindow.show() 
sys.exit(app.exec_())  
