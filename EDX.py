import time

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import xraydb


class Traitement_EDX:

    def __init__(self):
        self.signal = None
        self.information = None
        self.indice = 0

    def load(self, file: str):
        self.signal = hs.load(file)
        return len(self.signal)

    def info(self, indice : int = 0):
        if self.signal is None:
            print("No signal loaded.")
            return

        print("Signal information:")
        signal = self.signal[indice]

        # Print signal shape
        print(f"Shape: {signal.data.shape}")

        # Print data type
        print(f"Data type: {signal.data.dtype}")

        # If the signal has axes, print them
        if hasattr(signal, 'axes_manager'):
            print("Axes:")
            if signal.axes_manager:
                for axis in signal.axes_manager:
                    print(axis)
                    #print(f"{axis.name}: from {axis.low_value} to {axis.high_value} in {axis.size} steps, units: {axis.units}")

        # If the signal has metadata, print it
        if signal.metadata:
            print("Metadata:")
            print(signal.metadata)
            self.information = signal.metadata.as_dictionary()
            print(self.information)
            if 'Acquisition_instrument' in self.information:
                return self.information['General']['title'] + "\n" + str(self.information['Acquisition_instrument'])

        print("Nombre de signaux : " + str(len(self.signal)))

    def find_text_in_info(self, text_to_find: list = None):
        if text_to_find is None:
            text_to_find = ['BF', 'HAADF']

        good_i = None
        for i, signal in enumerate(self.signal):
            print(("Indice" + str(i) + "  " + str(type(signal))))

            signal = signal.metadata.as_dictionary()
            if 'General' in signal:
                if 'title' in signal['General']:
                    for j, text in enumerate(text_to_find):
                        if signal['General']['title'] == text:
                            if j == 0:
                                return i
                            if j == 1:
                                good_i = i
        if good_i is not None:
            return good_i
        return 2

    def on_click(self, event):
        if event.inaxes is not None:
            x, y = int(event.xdata), int(event.ydata)
            spectrum = self.signal[9].data[y, x] # supposant que data soit un tableau 2D
            print("Nb coup max : " + str(max(self.signal[9].data[y, x])))
            self.plot_spectrum(spectrum)

    def x_y_to_spectrum(self, x, y, indice: int = 9):
        return self.signal[indice].data[y, x]

    def plot_spectrum(self, spectrum):
        fig, ax = plt.subplots()

        ax.plot(np.arange(len(spectrum)), spectrum)

        ax.set(xlabel='Energy (keV)', ylabel='Intensity',
               title='X-ray spectrum')
        ax.grid()

        fig.canvas.draw()
        plt.show()

    def all_info_and_display(self):
        for i in range(0, len(self.signal)):
            self.info(i)
            if i >= 1:
                self.display(i)

    def send_image_edx(self, indice : int = 2):
        # Récupérer la donnée sous forme d'array numpy
        #print(self.signal[indice].data)
        return self.signal[indice].data

    def display(self, indice: int = 1):
        print(self.signal)
        self.indice = indice
        # Récupérer la donnée sous forme d'array numpy
        data = self.signal[indice].data

        # Utiliser matplotlib pour afficher la map
        fig, ax = plt.subplots()
        img = ax.imshow(data, cmap='gray')
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.colorbar(img, ax=ax)
        plt.show()

    def sum_z_values(self, indice: int = 9):
        if self.signal:
            longueur_des_x = len(self.signal[indice].data[0])
            longueur_des_y = len(self.signal[indice].data)
            longueur_des_z = len(self.signal[indice].data[0, 1])
            z = [0] * longueur_des_z
            for i in range(0, longueur_des_x):
                for j in range(0, longueur_des_y):
                    a = self.x_y_to_spectrum(i, j, indice=indice)
                    z += a

            return np.array(z)
        return None

    def frame_z_spectrum(self, x1, y1, x2, y2, direction: str = "y", indice: int = 9):
        print(indice)
        if self.signal:
            longueur_des_z = len(self.signal[indice].data[0, 1])

            img_z = []
            if direction == "y":
                for i in range(int(x1), int(x2)):
                    z = [0] * longueur_des_z
                    for j in range(int(y1), int(y2)):
                        a = self.x_y_to_spectrum(i, j, indice=indice)
                        z += a
                    img_z.append(z)
            elif direction == "x":
                for i in range(int(y1), int(y2)):
                    z = [0] * longueur_des_z
                    for j in range(int(x1), int(x2)):
                        a = self.x_y_to_spectrum(j, i, indice=indice)
                        z += a
                    img_z.append(z)
            elif direction == "xy":
                z = [0] * longueur_des_z
                for i in range(int(x1), int(x2)):
                    for j in range(int(y1), int(y2)):
                        a = self.x_y_to_spectrum(i, j, indice=indice)
                        z += a
                return np.array(z)
            return np.array(img_z)

        return None

    '''
    def convert_nm(self):
        # return nb nm / px
        return None

    def convert_eV(self):
        # return nb eV / px
        return None
    '''
    
    '''
    def display(self):
        # Récupérer la donnée sous forme d'array numpy
        data = self.signal[2].data

        # Utiliser matplotlib pour afficher la map
        plt.imshow(data, cmap='gray') # ici, je choisi la colormap 'gray', mais vous pouvez choisir celle qui vous convient le mieux.
        plt.colorbar()
        plt.show()
    '''


edx = Traitement_EDX()

edx.load("InAs-ZnTe-NWs-M3714 20250818 1116 SI 380 kx.emd")
edx.display(4)

'''
import matplotlib
matplotlib.rcParams["backend"] = "Agg"
import hyperspy.api as hs
'''



