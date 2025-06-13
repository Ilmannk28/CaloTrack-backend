# backend/utils/calories_prediction.py
import sys
from PyQt6 import QtWidgets, QtGui, QtCore
from services.predictor import predict_calories
import cv2

class FoodCalorieDetector(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Food Calorie Detector")
        self.resize(900, 900)
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel("Upload an image and predict calories.")
        layout.addWidget(self.label)

        self.btnLoad = QtWidgets.QPushButton("Load Image")
        self.btnLoad.clicked.connect(self.load_image)
        layout.addWidget(self.btnLoad)

        self.btnPredict = QtWidgets.QPushButton("Predict")
        self.btnPredict.clicked.connect(self.predict)
        self.btnPredict.setEnabled(False)
        layout.addWidget(self.btnPredict)

        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setMinimumSize(400, 300)
        layout.addWidget(self.imageLabel)

        self.resultList = QtWidgets.QListWidget()
        layout.addWidget(self.resultList)

        self.totalLabel = QtWidgets.QLabel("")
        layout.addWidget(self.totalLabel)

    def load_image(self):
        file_dialog = QtWidgets.QFileDialog(self)
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.image_path = file_path
            self.btnPredict.setEnabled(True)

            pixmap = QtGui.QPixmap(file_path)
            self.imageLabel.setPixmap(pixmap.scaled(400, 300, QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def predict(self):
        if not self.image_path:
            return

        try:
            result = predict_calories(self.image_path)
            self.resultList.clear()
            for item in result["predictions"]:
                self.resultList.addItem(f"{item['food']} - {item['calories']} kcal (conf: {item['confidence']})")
            self.totalLabel.setText(f"Total Calories: {result['total_calories']} kcal")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FoodCalorieDetector()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
