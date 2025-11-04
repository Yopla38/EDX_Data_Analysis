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
        """Charge un fichier EMD et retourne le nombre de signaux"""
        try:
            self.signal = hs.load(file)
            return len(self.signal)
        except Exception as e:
            print(f"Erreur lors du chargement du fichier : {e}")
            return 0

    def info(self, indice: int = 0):
        """Affiche les informations d'un signal"""
        if self.signal is None or indice >= len(self.signal):
            print("No signal loaded or invalid index.")
            return None

        print("Signal information:")
        signal = self.signal[indice]

        # Print signal shape
        print(f"Shape: {signal.data.shape}")
        print(f"Data type: {signal.data.dtype}")

        # If the signal has axes, print them
        if hasattr(signal, 'axes_manager'):
            print("Axes: -----ICI-----")
            print(signal.axes_manager)

        # If the signal has metadata, print it
        if signal.metadata:
            print("Metadata:")
            print(signal.metadata)
            self.information = signal.metadata.as_dictionary()
            print(self.information)
            if 'Acquisition_instrument' in self.information:
                return self.information['General']['title'] + "\n" + str(self.information['Acquisition_instrument'])

        print("Nombre de signaux : " + str(len(self.signal)))
        return None

    def get_axis_info(self, indice: int = 0):
        """Récupère les informations des axes (offset, scale, units)"""
        if self.signal is None or indice >= len(self.signal):
            return None

        signal = self.signal[indice]
        axis_info = {}

        if hasattr(signal, 'axes_manager'):
            for axis in signal.axes_manager.signal_axes:
                axis_info[axis.name] = {
                    'offset': axis.offset,
                    'scale': axis.scale,
                    'units': axis.units,
                    'size': axis.size
                }
            for axis in signal.axes_manager.navigation_axes:
                axis_info[axis.name] = {
                    'offset': axis.offset,
                    'scale': axis.scale,
                    'units': axis.units,
                    'size': axis.size
                }

        return axis_info

    def find_text_in_info(self, text_to_find: list = None):
        """Trouve l'indice du signal correspondant aux textes recherchés"""
        if text_to_find is None:
            text_to_find = ['BF', 'HAADF']

        if self.signal is None:
            return None

        good_i = None
        for i, signal in enumerate(self.signal):
            print(f"Indice {i}: {type(signal)}")

            metadata = signal.metadata.as_dictionary()
            if 'General' in metadata:
                if 'title' in metadata['General']:
                    for j, text in enumerate(text_to_find):
                        if metadata['General']['title'] == text:
                            if j == 0:
                                return i
                            if j == 1:
                                good_i = i

        # Si aucun texte trouvé, retourne le premier signal spatial disponible
        if good_i is not None:
            return good_i

        # Cherche le premier signal 2D
        for i, signal in enumerate(self.signal):
            if len(signal.data.shape) == 2:
                print(f"Aucun texte trouvé, utilisation du signal 2D à l'indice {i}")
                return i

        print("Attention : aucun signal approprié trouvé")
        return 0

    def x_y_to_spectrum(self, x, y, indice: int = 9):
        """Retourne le spectre au point (x, y)"""
        if self.signal is None or indice >= len(self.signal):
            return None

        data = self.signal[indice].data

        # Vérifier les dimensions
        if len(data.shape) < 3:
            print(f"Erreur : le signal {indice} n'a pas de dimension spectrale")
            return None

        # Vérifier les limites
        if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
            return data[y, x]
        else:
            print(f"Coordonnées hors limites : ({x}, {y})")
            return None

    def send_image_edx(self, indice: int = 2):
        """Retourne la donnée sous forme d'array numpy"""
        if self.signal is None or indice >= len(self.signal):
            return None
        return self.signal[indice].data

    def sum_z_values(self, indice: int = 9):
        """Somme tous les spectres de la carte EDX"""
        if self.signal is None or indice >= len(self.signal):
            return None

        data = self.signal[indice].data

        if len(data.shape) == 3:
            # Somme sur les deux premières dimensions (spatial)
            z = np.sum(data, axis=(0, 1))
            return z
        else:
            print(f"Erreur : dimension incorrecte pour signal {indice}")
            return None

    def frame_z_spectrum(self, x1, y1, x2, y2, direction: str = "y", indice: int = 9):
        """
        Calcule les spectres dans une région rectangulaire
        direction="xy" : somme totale de la région
        direction="y" : profil selon y (somme selon x pour chaque y)
        direction="x" : profil selon x (somme selon y pour chaque x)
        """
        print(f"frame_z_spectrum: x=[{x1}:{x2}], y=[{y1}:{y2}], direction={direction}")

        if self.signal is None or indice >= len(self.signal):
            return None

        data = self.signal[indice].data

        if len(data.shape) != 3:
            print(f"Erreur : le signal {indice} doit être 3D (y, x, energy)")
            return None

        longueur_des_z = data.shape[2]

        # Conversion en int et vérification
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)

        # S'assurer que les limites sont dans l'ordre correct
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if direction == "xy":
            # Somme totale sur la région
            z = np.sum(data[y1:y2, x1:x2, :], axis=(0, 1))
            return z

        elif direction == "y":
            # Profil selon y (chaque ligne est la somme selon x)
            img_z = []
            for j in range(y1, y2):
                z = np.sum(data[j, x1:x2, :], axis=0)
                img_z.append(z)
            return np.array(img_z)

        elif direction == "x":
            # Profil selon x (chaque ligne est la somme selon y)
            img_z = []
            for i in range(x1, x2):
                z = np.sum(data[y1:y2, i, :], axis=0)
                img_z.append(z)
            return np.array(img_z)

        return None


# Test uniquement si exécuté directement
if __name__ == "__main__":
    edx = Traitement_EDX()
    nb_signals = edx.load("InAs-ZnTe-NWs-M3714 20250818 1116 SI 380 kx.emd")

    if nb_signals > 0:
        for i in range(min(5, nb_signals)):  # Affiche les 5 premiers signaux
            print(f"\n{'=' * 60}\nSignal {i}\n{'=' * 60}")
            edx.info(i)
