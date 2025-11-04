import sys
import numpy as np
from PyQt6 import QtGui, QtWidgets, QtCore
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from EDX import Traitement_EDX


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.traitement_EDX = Traitement_EDX()

        # Variables d'état
        self.BF_img = None
        self.current_indice = None
        self.axis_info_cache = {}
        self.img_item_plot1 = None
        self.img_item_plot2 = None

        # Setup UI
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        """Initialise l'interface graphique"""
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        # Buttons
        self.load_button = QtWidgets.QPushButton('Load')
        self.save_button = QtWidgets.QPushButton('Save')
        self.pointer_button = QtWidgets.QPushButton('Pointeur')
        self.frame_button = QtWidgets.QPushButton('Hide/Show box')
        self.compute_frame_button = QtWidgets.QPushButton('Compute box')
        self.sum_z = QtWidgets.QPushButton('Sum of z')
        self.x_checkbox = QtWidgets.QCheckBox('x')

        # Plots
        self.plot_1 = PlotWidget()
        self.plot_2 = PlotWidget()
        self.plot_3 = PlotWidget()

        # Dropdown
        self.dropdown = QtWidgets.QComboBox()

        # Text fields
        self.text_field_1 = QtWidgets.QTextEdit()
        self.text_field_1.setReadOnly(True)
        self.text_field_2 = QtWidgets.QTextEdit()
        self.text_field_2.setReadOnly(True)

        # Layouts
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.load_button)
        hbox.addWidget(self.save_button)

        hboxtools = QtWidgets.QVBoxLayout()
        hboxtools.addWidget(self.dropdown)
        hboxtools.addWidget(self.pointer_button)
        hboxtools.addWidget(self.frame_button)
        hboxtools.addWidget(self.compute_frame_button)
        hboxtools.addWidget(self.sum_z)

        # Add to main layout
        layout.addLayout(hbox, 0, 0, 1, 2)
        layout.addLayout(hboxtools, 1, 1)
        layout.addWidget(self.plot_1, 1, 0)
        layout.addWidget(self.text_field_1, 2, 0)
        layout.addWidget(self.plot_2, 1, 2)
        layout.addWidget(self.text_field_2, 2, 2)
        layout.addWidget(self.x_checkbox, 3, 2)
        layout.addWidget(self.plot_3, 4, 0, 1, 3)

        # ROI (initialement cachés)
        self.roi_plot_1 = pg.RectROI([0, 0], [10, 10], pen=pg.mkPen('r', width=2))
        self.roi_plot_2 = pg.RectROI([0, 0], [10, 10], pen=pg.mkPen('r', width=2))
        self.roi_plot_1.hide()
        self.roi_plot_2.hide()
        self.plot_1.addItem(self.roi_plot_1)
        self.plot_2.addItem(self.roi_plot_2)

    def connect_signals(self):
        """Connecte tous les signaux"""
        self.load_button.clicked.connect(self.load_file)
        self.dropdown.currentIndexChanged.connect(self.change_indice_and_display)
        self.plot_1.scene().sigMouseClicked.connect(self.on_plot_click_plot_1)
        self.plot_2.scene().sigMouseClicked.connect(self.on_plot_click_plot_2)
        self.compute_frame_button.clicked.connect(self.compute_frame)
        self.sum_z.clicked.connect(self.trace_sum_z)
        self.frame_button.clicked.connect(self.toggle_frame)
        self.roi_plot_1.sigRegionChangeFinished.connect(self.on_roi1_changed)
        self.roi_plot_2.sigRegionChangeFinished.connect(self.on_roi2_changed)

    def load_file(self):
        """Charge un fichier EMD"""
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open .emd file", "", "Electron microscopy data files (*.emd)")

        if not filePath:
            return

        taille_signal = self.traitement_EDX.load(filePath)

        if taille_signal == 0:
            QtWidgets.QMessageBox.critical(self, "Erreur", "Impossible de charger le fichier")
            return

        self.start_indice = 1
        self.stop_indice = taille_signal - 1

        # Trouver l'image BF/HAADF
        self.BF_img = self.traitement_EDX.find_text_in_info()
        if self.BF_img is None:
            QtWidgets.QMessageBox.warning(self, "Attention", "Aucune image BF/HAADF trouvée")
            return

        # Remplir le dropdown (exclure le premier et dernier signal généralement)
        self.dropdown.clear()
        self.dropdown.addItems([str(i) for i in range(1, taille_signal - 1)])

        # Afficher l'image principale
        self.display_image_plot1()

        # Afficher la première image secondaire
        if self.dropdown.count() > 0:
            self.change_indice_and_display()

        # Afficher la somme totale des spectres
        self.trace_sum_z()

    def display_image_plot1(self):
        """Affiche l'image dans plot_1"""
        if self.BF_img is None:
            return

        data = self.traitement_EDX.send_image_edx(indice=self.BF_img)
        if data is None:
            return

        data_transposed = np.transpose(data)
        axis_info = self.traitement_EDX.get_axis_info(indice=self.BF_img)
        self.axis_info_cache[self.BF_img] = axis_info

        # Supprimer l'ancienne image si elle existe
        if self.img_item_plot1 is not None:
            self.plot_1.removeItem(self.img_item_plot1)

        self.img_item_plot1 = pg.ImageItem(data_transposed)

        # Appliquer l'échelle SANS offset
        if axis_info and 'x' in axis_info and 'y' in axis_info:
            x_info = axis_info['x']
            y_info = axis_info['y']

            x_max = x_info['size'] * x_info['scale']
            y_max = y_info['size'] * y_info['scale']

            rect = QtCore.QRectF(0, 0, x_max, y_max)
            self.img_item_plot1.setRect(rect)

            self.plot_1.setLabel('bottom', f'X ({x_info["units"]})')
            self.plot_1.setLabel('left', f'Y ({y_info["units"]})')

            # Initialiser le ROI avec une taille raisonnable (10% de l'image)
            roi_width = x_max * 0.1
            roi_height = y_max * 0.1
            self.roi_plot_1.setPos([x_max * 0.4, y_max * 0.4])
            self.roi_plot_1.setSize([roi_width, roi_height])

        self.plot_1.addItem(self.img_item_plot1)

        # Afficher les infos
        text = self.traitement_EDX.info(indice=self.BF_img)
        if text:
            self.text_field_1.setText(text)

    def change_indice_and_display(self):
        """Change l'image affichée dans plot_2"""
        if self.dropdown.count() == 0:
            return

        try:
            indice = int(self.dropdown.currentText())
        except ValueError:
            return

        self.current_indice = indice

        data = self.traitement_EDX.send_image_edx(indice=indice)
        if data is None:
            return

        data_transposed = np.transpose(data)
        axis_info = self.traitement_EDX.get_axis_info(indice=indice)
        self.axis_info_cache[indice] = axis_info

        # Supprimer l'ancienne image
        if self.img_item_plot2 is not None:
            self.plot_2.removeItem(self.img_item_plot2)

        self.img_item_plot2 = pg.ImageItem(data_transposed)

        # Appliquer l'échelle SANS offset
        if axis_info and 'x' in axis_info and 'y' in axis_info:
            x_info = axis_info['x']
            y_info = axis_info['y']

            x_max = x_info['size'] * x_info['scale']
            y_max = y_info['size'] * y_info['scale']

            rect = QtCore.QRectF(0, 0, x_max, y_max)
            self.img_item_plot2.setRect(rect)

            self.plot_2.setLabel('bottom', f'X ({x_info["units"]})')
            self.plot_2.setLabel('left', f'Y ({y_info["units"]})')

            # Synchroniser le ROI
            pos = self.roi_plot_1.pos()
            size = self.roi_plot_1.size()
            self.roi_plot_2.setPos(pos, update=False)
            self.roi_plot_2.setSize(size, update=False)

        self.plot_2.addItem(self.img_item_plot2)

        # Afficher les infos
        text = self.traitement_EDX.info(indice=indice)
        if text:
            self.text_field_2.setText(text)

    def on_plot_click_plot_1(self, event):
        """Gère le clic sur plot_1"""
        if self.BF_img is None:
            return

        pos = event.pos()
        point = self.plot_1.plotItem.vb.mapSceneToView(pos)
        x_real = point.x()
        y_real = point.y()

        self.plot_spectrum_at_position(x_real, y_real, self.BF_img)

    def on_plot_click_plot_2(self, event):
        """Gère le clic sur plot_2"""
        if self.current_indice is None:
            return

        pos = event.pos()
        point = self.plot_2.plotItem.vb.mapSceneToView(pos)
        x_real = point.x()
        y_real = point.y()

        self.plot_spectrum_at_position(x_real, y_real, self.current_indice)

    def plot_spectrum_at_position(self, x_real, y_real, img_indice):
        """Affiche le spectre à une position donnée"""
        axis_info = self.axis_info_cache.get(img_indice)
        if not axis_info or 'x' not in axis_info or 'y' not in axis_info:
            return

        # Convertir en pixels SANS offset
        x_px = int(round(x_real / axis_info['x']['scale']))
        y_px = int(round(y_real / axis_info['y']['scale']))

        # Vérifier les limites
        if not (0 <= x_px < axis_info['x']['size'] and 0 <= y_px < axis_info['y']['size']):
            print(f"Click outside image bounds: pixel ({x_px}, {y_px}), real ({x_real:.2f}, {y_real:.2f})")
            return

        print(f"Clicked at pixel ({x_px}, {y_px}) = ({x_real:.2f}, {y_real:.2f}) {axis_info['x']['units']}")

        data = self.traitement_EDX.x_y_to_spectrum(x_px, y_px, self.stop_indice)
        if data is None:
            return

        self.plot_spectrum(data)

    def trace_sum_z(self):
        """Trace la somme de tous les spectres"""
        data = self.traitement_EDX.sum_z_values(self.stop_indice)
        if data is not None:
            self.plot_spectrum(data)

    def compute_frame(self):
        """Calcule et affiche le spectre de la région sélectionnée"""
        if self.BF_img is None:
            return

        position = self.roi_plot_1.pos()
        size = self.roi_plot_1.size()
        x1_real = position.x()
        y1_real = position.y()
        x2_real = x1_real + size.x()
        y2_real = y1_real + size.y()

        axis_info = self.axis_info_cache.get(self.BF_img)
        if not axis_info or 'x' not in axis_info or 'y' not in axis_info:
            return

        # Convertir en pixels SANS offset
        x1 = x1_real / axis_info['x']['scale']
        y1 = y1_real / axis_info['y']['scale']
        x2 = x2_real / axis_info['x']['scale']
        y2 = y2_real / axis_info['y']['scale']

        # S'assurer que min < max
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Arrondir et convertir en int
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        # Limiter aux dimensions de l'image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(axis_info['x']['size'], x2)
        y2 = min(axis_info['y']['size'], y2)

        # Vérifier qu'on a bien une région valide
        if x1 >= x2 or y1 >= y2:
            print("Erreur : région invalide")
            return

        print(f"Computing frame: x=[{x1}:{x2}], y=[{y1}:{y2}]")

        data = self.traitement_EDX.frame_z_spectrum(x1, y1, x2, y2, direction="xy", indice=self.stop_indice)
        if data is not None:
            self.plot_spectrum(data)

    def plot_spectrum(self, data):
        """Affiche un spectre avec l'échelle d'énergie"""
        energy_axis = self.traitement_EDX.get_axis_info(indice=0)

        self.plot_3.clear()

        if energy_axis and 'Energy' in energy_axis:
            e_info = energy_axis['Energy']
            x_energy = np.arange(len(data)) * e_info['scale'] + e_info['offset']
            self.plot_3.plot(x_energy, data)
            self.plot_3.setLabel('bottom', f'Energy ({e_info["units"]})')
            self.plot_3.setLabel('left', 'Intensity (counts)')
        else:
            self.plot_3.plot(data)
            self.plot_3.setLabel('bottom', 'Channel')
            self.plot_3.setLabel('left', 'Intensity (counts)')

    def toggle_frame(self):
        """Affiche/masque les ROIs"""
        if self.roi_plot_1.isVisible():
            self.roi_plot_1.hide()
            self.roi_plot_2.hide()
        else:
            self.roi_plot_1.show()
            self.roi_plot_2.show()

    def on_roi1_changed(self):
        """Synchronise ROI2 avec ROI1"""
        position = self.roi_plot_1.pos()
        size = self.roi_plot_1.size()
        self.roi_plot_2.setPos(position, update=False)
        self.roi_plot_2.setSize(size, update=False)

    def on_roi2_changed(self):
        """Synchronise ROI1 avec ROI2"""
        position = self.roi_plot_2.pos()
        size = self.roi_plot_2.size()
        self.roi_plot_1.setPos(position, update=False)
        self.roi_plot_1.setSize(size, update=False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
