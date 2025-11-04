import sys

import imageio
import numpy as np
from PyQt6 import QtGui, QtWidgets, QtCore
from pyqtgraph import PlotWidget
from EDX import Traitement_EDX


import sys
from PyQt6 import QtGui, QtWidgets, QtCore
from pyqtgraph import PlotWidget
import pyqtgraph as pg

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.traitement_EDX = Traitement_EDX()

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Create Load and Save buttons
        self.load_button = QtWidgets.QPushButton('Load')
        self.save_button = QtWidgets.QPushButton('Save')

        # Create pointer, frame buttons and checkbox
        self.pointer_button = QtWidgets.QPushButton('Pointeur')
        self.frame_button = QtWidgets.QPushButton('Hide/Show box')
        self.compute_frame_button = QtWidgets.QPushButton('Compute box')
        self.sum_z = QtWidgets.QPushButton('Sum of z')
        self.x_checkbox = QtWidgets.QCheckBox('x')

        # Create plots
        self.plot_1 = PlotWidget()
        self.plot_2 = PlotWidget()

        # Create dropdown
        self.dropdown = QtWidgets.QComboBox()
        #self.dropdown.addItems([str(i) for i in range(1, 11)])

        # Create text fields
        self.text_field_1 = QtWidgets.QTextEdit()
        self.text_field_2 = QtWidgets.QTextEdit()

        # Créez un nouveau widget pour le graphique XY
        self.plot_3 = PlotWidget()

        # Group buttons system
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.load_button)
        hbox.addWidget(self.save_button)

        # Group buttons tools
        hboxtools = QtWidgets.QVBoxLayout()
        hboxtools.addWidget(self.dropdown)
        hboxtools.addWidget(self.pointer_button)
        hboxtools.addWidget(self.frame_button)
        hboxtools.addWidget(self.compute_frame_button)
        hboxtools.addWidget(self.sum_z)

        # add to layout
        layout.addLayout(hbox, 0, 0, 1, 2)
        layout.addLayout(hboxtools, 1, 1)
        layout.addWidget(self.plot_1, 1, 0)
        layout.addWidget(self.text_field_1, 2, 0)
        #layout.addWidget(self.dropdown, 2, 1)
        layout.addWidget(self.plot_2, 1, 2)
        layout.addWidget(self.text_field_2, 2, 2)
        #layout.addWidget(self.pointer_button, 2, 1)
        #layout.addWidget(self.frame_button, 3, 1)
        layout.addWidget(self.x_checkbox, 3, 2)
        # Ajoute le graphique XY à la position désirée dans la grille

        layout.addWidget(self.plot_3, 4, 0, 1, 3)
        #layout.addWidget(self.sum_z, 4,1)

        # connect boutton
        self.load_button.clicked.connect(self.load_file)
        self.dropdown.currentIndexChanged.connect(self.change_indice_and_display)
        self.plot_1.scene().sigMouseClicked.connect(self.on_plot_click_plot_1)  # connect click avec plot_1
        self.plot_2.scene().sigMouseClicked.connect(self.on_plot_click_plot_2)  # connect click avec plot_1
        self.compute_frame_button.clicked.connect(self.compute_frame)
        self.sum_z.clicked.connect(self.trace_sum_z)

        # -------------------------SELECTION--------------------------
        self.roi_plot_1 = pg.RectROI([0, 0], [100, 100], pen=pg.mkPen('r', width=2))  # ROI initiale
        self.roi_plot_2 = pg.RectROI([0, 0], [100, 100], pen=pg.mkPen('r', width=2))  # ROI initiale
        self.plot_1.addItem(self.roi_plot_1)
        self.plot_2.addItem(self.roi_plot_2)  # Ajouter ROI pour plot_2 aussi

        # Connectez sigRegionChanged à la fonction correspondante
        self.roi_plot_1.sigRegionChangeFinished.connect(self.on_roi1_changed)
        self.roi_plot_2.sigRegionChangeFinished.connect(self.on_roi2_changed)
        self.frame_button.clicked.connect(self.draw_frame)

        self.draw_frame()
        #self.roi.sigRegionChangeFinished.connect(self.roi_moved)  # Connecter le signal

    def load_file(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open .emd file", "",
                                                            "Electron microscopy data files (*.emd)")

        if filePath:
            self.taille_signal = self.traitement_EDX.load(filePath)
            self.dropdown.addItems([str(i) for i in range(1, self.taille_signal-1)])
            self.start_indice = 1
            self.stop_indice = self.taille_signal-1

            # image principale
            # find BF title
            self.BF_img = self.traitement_EDX.find_text_in_info()

            data = self.traitement_EDX.send_image_edx(indice=self.BF_img)
            data_transposed = np.transpose(data)  # transpose data
            self.img = pg.ImageItem(data_transposed)
            self.plot_1.addItem(self.img)
            text = self.traitement_EDX.info(indice=self.BF_img)
            self.text_field_1.setText(text)
            self.text_field_1.setReadOnly(True)

            self.change_indice_and_display()

            # plot sum_z
            data = self.traitement_EDX.sum_z_values(self.stop_indice)
            self.plot_3.clear()  # Clear previous plots
            self.plot_3.plot(data)

    def change_indice_and_display(self):
        #Récuperer la valeur de dropdown
        indice = int(self.dropdown.currentText())
        # image_secondaire
        data = self.traitement_EDX.send_image_edx(indice=indice)
        data_transposed = np.transpose(data)  # transpose data
        self.img = pg.ImageItem(data_transposed)
        self.plot_2.addItem(self.img)
        text = self.traitement_EDX.info(indice=indice)
        self.text_field_2.setText(text)
        self.text_field_2.setReadOnly(True)

    # on clique sur une valeur de plot_1
    def on_plot_click_plot_1(self, event):
        pos = event.pos()
        point = self.plot_1.plotItem.vb.mapSceneToView(pos)
        x = point.x()
        y = point.y()
        data = self.traitement_EDX.x_y_to_spectrum(int(x), int(y), self.stop_indice)
        data = np.array(data)  # Just to ensure data is in numpy array format
        self.plot_3.clear()  # Clear previous plots
        self.plot_3.plot(data)

    def on_plot_click_plot_2(self, event):
        pos = event.pos()
        point = self.plot_2.plotItem.vb.mapSceneToView(pos)
        x = point.x()
        y = point.y()
        data = self.traitement_EDX.x_y_to_spectrum(int(x), int(y), self.stop_indice)
        data = np.array(data)  # Just to ensure data is in numpy array format
        self.plot_3.clear()  # Clear previous plots
        self.plot_3.plot(data)

    def trace_sum_z(self):
        data = self.traitement_EDX.sum_z_values(self.stop_indice)
        self.plot_3.clear()  # Clear previous plots
        self.plot_3.plot(data)

    def draw_frame(self):
        if self.roi_plot_1.isVisible():
            self.roi_plot_1.hide()
            self.roi_plot_2.hide()
        else:
            self.roi_plot_1.show()
            self.roi_plot_2.show()


    # Handle when roi1 change
    def on_roi1_changed(self):
        position = self.roi_plot_1.pos()
        size = self.roi_plot_1.size()
        self.roi_plot_2.setPos(position, update=False)
        self.roi_plot_2.setSize(size, update=False)

    # Handle when roi2 change
    def on_roi2_changed(self):
        position = self.roi_plot_2.pos()
        size = self.roi_plot_2.size()
        self.roi_plot_1.setPos(position, update=False)
        self.roi_plot_1.setSize(size, update=False)

    def compute_frame(self):
        position = self.roi_plot_1.pos()
        size = self.roi_plot_1.size()
        x1 = position.x()
        y1 = position.y()
        x2 = x1 + size.x()
        y2 = y1 + size.y()
        data = self.traitement_EDX.frame_z_spectrum(x1, y1, x2, y2, direction="xy", indice=self.stop_indice)
        self.plot_3.clear()  # Clear previous plots
        self.plot_3.plot(data)

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
