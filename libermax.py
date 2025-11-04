import os.path

import cv2
import h5py
import numpy as np
import libertem.api as lt
from hyperspy.io_plugins.digital_micrograph import DigitalMicrographReader
from libertem.analysis import PickFrameAnalysis
from scipy import ndimage
from scipy.ndimage import median_filter, fourier_shift
from skimage.io import imsave, imread_collection
from libertem.udf import UDF
from skimage.registration import phase_cross_correlation
from tqdm import tqdm  # Progress bar


class SumFramesUDF(UDF):
    def get_result_buffers(self):
        return {"sum_frames": self.buffer(kind="sig", dtype="float32")}

    def process_frame(self, frame):
        self.results.sum_frames[:] += frame
        # self.results.sum_frames[y1:y2, x1:x2] += frame[y1:y2, x1:x2] # pour les cadres

    def merge(self, dest, src):
        dest.sum_frames[:] += src.sum_frames

    def rotate_result(self, angle):
        # Effectuez la rotation après le calcul de la somme totale
        rotated_sum = ndimage.rotate(self.results.sum_frames[:], angle, reshape=False)
        return rotated_sum


class DriftCorrectionUDF(UDF):
    def get_result_buffers(self):
        return {
            "corrected_images": self.buffer(kind="nav", dtype="float32",
                                            extra_shape=(np.prod(self.meta.dataset_shape.sig),)),
            "shifts": self.buffer(kind="nav", dtype="float32", extra_shape=(2,)),
        }

    def process_frame(self, frame):
        shift, _, _ = phase_cross_correlation(self.params.base_frame, frame)
        corrected_frame = fourier_shift(np.fft.fftn(frame), shift)
        self.results.corrected_images[:] = corrected_frame.flatten()
        self.results.shifts[:] = shift

    def merge(self, dest, src):
        dest.corrected_images[:] = src.corrected_images
        dest.shifts[:] = src.shifts


class Legend:
    def __init__(self, mode: str = '8bit'):
        self.mode = mode
        self.position = (10, 40)
        self.background = (0, 0, 0)
        self.color = (255, 255, 255) if mode == '8bit' else (65535, 65535, 65535)
        self.font_scale = 1.0
        self.thickness = 2
        self.y_expand = 5

    def add_legend(self, image, text):
        lines = text.split('\n')
        y_offset = 0

        for line in lines:
            text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness)
            if self.background is not None:
                text_x, text_y = self.position[0], self.position[1] + y_offset
                bg_x1, bg_y1 = text_x, text_y - text_size[1]
                bg_x2, bg_y2 = text_x + text_size[0], text_y + self.y_expand
                cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), self.background, cv2.FILLED)

            cv2.putText(image, line, (self.position[0], self.position[1] + y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, self.color, self.thickness)
            y_offset += text_size[1]

        cv2.putText(image, text, self.position, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.color, self.thickness)


class ImageProcessor:

    def __init__(self):

        self.actual_frame = None
        self.gtg_path_filename = None
        self.metadata = None
        self.ctx = lt.Context.make_with(cpus=4)
        self.dataset = None
        self.images = []
        self.fps = None
        self.output_directory = "."
        self.position_legend = (10, 40)

    def load_data(self, gtg_path_file_name):
        self.gtg_path_filename = gtg_path_file_name
        print("Loading data...")
        self.dataset = self.ctx.load("K2IS", path=gtg_path_file_name, sync_offset=0)
        self.read_meta_info()
        print("Data loaded !")

    def define_output_directory(self, path: str = '.'):
        if os.path.exists(path):
            self.output_directory = path
        else:
            print("Path doesn't exist !")

    def extract_image(self, dest_path_folder=None, sum_frames: int = 40, img_format: str = 'tif', dtype: str = '8bit',
                      start_frame: int = 0, stop_frame: int = None, normalization_mode: str = 'percentile',
                      drift_correction: bool = False,
                      rotation=None, crop=None, mylegend=None):
        """
        This function extract, sum and  normalize images from bin K2-IS Gatan camera files
        :param dest_path_folder: folder of saving images
        :param sum_frames: number of frames summed
        :param img_format: can be 'tif'
        :param dtype: can be 8bit or 16bit
        :param start_frame: K2-IS make 400 frames by seconde. Default: 0
        :param stop_frame: 1 second is 400. Default: max frames
        :param normalization_mode: can be 'percentile', 'min_max', 'z_score'. Default: percentile
        :param rotation: Can rotate the frames during parallel processing (deg). Default: None
        :param crop: Crop an image with (x1, y1, x2, y2). Default: None
        :param legend: Add legend text. for example : ["get_xscale", "get_pixel_size"]
        """
        if stop_frame is None:
            stop_frame = self.dataset.shape.nav.size

        if dest_path_folder == None or not os.path.exists(dest_path_folder):
            dest_path_folder = self.output_directory

        # Add counter for progress bar
        progress_counter = tqdm(total=(stop_frame - start_frame) // sum_frames)

        # calculate the fps for computing future movie
        self.fps = 400 / sum_frames

        # Define the parallel function
        sum_udf = SumFramesUDF()

        # Add legend
        customlegend = None
        if mylegend:
            customlegend = Legend(dtype)

        # if drift
        if drift_correction:
            index = 0
            if crop:
                shape = (0, crop[3] - crop[1], crop[2] - crop[0])
                maxshape = (None, crop[3] - crop[1], crop[2] - crop[0])
                cadrey = crop[3] - crop[1]
                cadrex = crop[2] - crop[0]
            else:
                shape = (0, self.dataset.shape.sig[0], self.dataset.shape.sig[1])
                maxshape = (None, self.dataset.shape.sig[0], self.dataset.shape.sig[1])
                cadrey = self.dataset.shape.sig[0]
                cadrex = self.dataset.shape.sig[1]

            # Opening the HDF5 file (will stay open)
            f = h5py.File(os.path.join(self.output_directory, 'results.h5'), 'w')
            # Création d'un dataset vide dans le fichier HDF5
            dset = f.create_dataset('results', shape=shape, maxshape=maxshape, dtype=np.uint8)

        # Create the process
        for i in range(start_frame, stop_frame, sum_frames):

            roi = np.zeros(self.dataset.shape.nav, dtype=bool)
            roi[i:i + sum_frames] = True
            result = self.ctx.run_udf(dataset=self.dataset, udf=sum_udf, roi=roi)

            # Crop data
            if crop is not None:
                buffer = result['sum_frames'].data[crop[1]:crop[3], crop[0]:crop[2]]
            else:
                buffer = result['sum_frames'].data

            # Rotate data
            if rotation is not None:
                buffer = ndimage.rotate(buffer, rotation, reshape=False)

            # Normalize data
            result_norm = self.normalize_img(buffer, normalization_mode)

            if dtype == '8bit':
                img = (result_norm * 255).astype(np.uint8)
            elif dtype == '16bit':
                img = (result_norm * 65535).astype(np.uint16)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

            # Update progress bar
            progress_counter.update()

            # drift
            if drift_correction:
                # Redimensionnement du dataset pour accueillir le nouveau résultat
                dset.resize((index + 1, cadrey, cadrex))
                # Ajout du nouveau résultat au dataset
                dset[index] = img
                index += 1
            else:
                # path image
                filename = f"image_{i}_to_{i + sum_frames - 1}_{i // sum_frames}.{img_format}"
                path_filename = os.path.join(dest_path_folder, filename)
                self.images.append(path_filename)

                # Add legend
                if customlegend:
                    self.actual_frame = i + sum_frames  # used for tag time
                    customlegend.add_legend(img, self.concatenate_metadata(mylegend))

                # sauvegarder l'image
                '''
                cv2.imwrite('output.tif', img, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
                ```
                Les options pour `cv2.IMWRITE_TIFF_COMPRESSION` sont :
                - 0 : Aucune compression
                - 1 : Compression LZW
                - 2 : Compression RLE
                - 3 : Compression JPEG
                - 4 : Compression avec le format Adobe deflate.
                '''
                cv2.imwrite(path_filename, img, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
                # imsave(path_filename, img)

        if drift_correction:
            f.close()
            result_dataset = self.ctx.load("hdf5", path=os.path.join(self.output_directory, 'results.h5'),
                                           ds_path="/results")
            roi = np.zeros(result_dataset.shape.nav, dtype=bool)
            roi[(0,)] = True  # Ici on active seulement la première frame
            analysis = PickFrameAnalysis(dataset=result_dataset, parameters={"x": 0})
            first_frame_result = self.ctx.run(analysis)
            first_frame = first_frame_result.intensity.raw_data
            udf = DriftCorrectionUDF(base_frame=first_frame)
            drift_correction_result = self.ctx.run_udf(udf=udf, dataset=result_dataset)
            corrected_images = drift_correction_result["corrected_images"].data
            # Loop through each corrected image and save:
            for i in range(corrected_images.shape[0]):
                img = corrected_images[i]
                result_norm = self.normalize_img(img)
                if dtype == '8bit':
                    img = (result_norm * 255).astype(np.uint8)
                elif dtype == '16bit':
                    img = (result_norm * 65535).astype(np.uint16)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype}")
                # path image
                filename = f"image_{i * sum_frames}_to_{i * sum_frames + sum_frames - 1}_{i}.{img_format}"
                path_filename = os.path.join(dest_path_folder, filename)
                self.images.append(path_filename)

                # Add legend
                if customlegend:
                    self.actual_frame = i + 10  # used for tag time
                    customlegend.add_legend(img, self.concatenate_metadata(mylegend))

                # sauvegarder l'image
                '''
                cv2.imwrite('output.tif', img, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
                ```
                Les options pour `cv2.IMWRITE_TIFF_COMPRESSION` sont :
                - 0 : Aucune compression
                - 1 : Compression LZW
                - 2 : Compression RLE
                - 3 : Compression JPEG
                - 4 : Compression avec le format Adobe deflate.
                '''
                cv2.imwrite(path_filename, img, [int(cv2.IMWRITE_TIFF_COMPRESSION), 1])
                # imsave(path_filename, img)

        # Kill the progress bar
        progress_counter.close()

    def normalize_img(self, data, mode: str = 'percentile'):
        """

        :param data:
        :param mode: 'percentile', 'min_max', 'z_score'
        :return: Normalized data
        """
        # Normalisation percentile
        if mode == 'percentile':
            min_val = np.percentile(data, 1)
            max_val = np.percentile(data, 99)

        # Normalisation Min-max
        elif mode == 'min_max':
            min_val = np.min(data)
            max_val = np.max(data)

        # Normalisation Z-score
        elif mode == 'z_score':
            min_val = 0
            max_val = (data - np.mean(data)) / np.std(data)

        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")

        norm_data = (data - min_val) / (max_val - min_val)
        norm_data = np.clip(norm_data, 0, 1)

        return norm_data

    def alpha_trimmed_mean_filter(self, img, filter_size, alpha):

        padded_img = np.pad(img, [((filter_size[0] - 1) // 2,), ((filter_size[1] - 1) // 2,)], 'symmetric')
        filtered_img = np.zeros_like(img)
        row, col = np.shape(img)
        k = (filter_size[0] - 1) // 2
        l = (filter_size[1] - 1) // 2
        for i in range(k, row + k):
            for j in range(l, col + l):
                temp = padded_img[i - k:i + k + 1, j - l:j + l + 1]
                temp = np.sort(temp, axis=None)
                d_pq = np.mean(temp[alpha // 2: filter_size[0] * filter_size[1] - alpha // 2])
                filtered_img[i - k, j - l] = d_pq
        return filtered_img

    def remove_noise_salt_pepper(self, img, size: int = 3):
        """
        cette fonction supprime le bruit salt and pepper en utilisant un filtre median
        """
        return median_filter(img, size=size)

    def drift_correction(self, image_number):
        roi = self.dataset.get_roi("image")
        first_frame = self.dataset.data[roi.start[0], roi.start[1]]
        udf = DriftCorrectionUDF(base_frame=first_frame)
        results = self.ctx.run_udf(udf=udf, dataset=self.dataset)

    def compute_movies(self, dest_path_folder=None, video_format: str = 'AVI', codec_format: str = 'MJPG',
                       fps: str = 'auto'):
        """
        compute_movies make a movies from previously extracted 8 bits images.
        Must be called after extract_image with parameter 8bit
        :param dest_path_folder: The path folder for the movie file
        :param video_format: Can be AVI. Default: AVI
        :param codec_format: Can be MJPG. Default: MJPG
        :param fps: Define frame by second on movies. Default: auto (calculation with 400fps and previously extracted images)
        """

        if fps == 'auto':
            fps = self.fps
        else:
            fps = int(fps)

        imgs = imread_collection(self.images)

        if dest_path_folder == None or not os.path.exists(dest_path_folder):
            dest_path_folder = self.output_directory

        if video_format == 'AVI':
            fourcc = cv2.VideoWriter_fourcc(*codec_format)
            height, width = imgs[0].shape
            path_file = os.path.join(dest_path_folder, f"video.{video_format}")
            print(f'Write movie in {path_file}')
            video = cv2.VideoWriter(path_file, fourcc, fps, (width, height))

            # Add counter for progress bar
            progress_counter = tqdm(total=len(imgs))

            for img in imgs:
                video.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
                # Update progress bar
                progress_counter.update()

            # Kill the progress bar
            progress_counter.close()

            video.release()
            print("Movie wrote !")
        else:
            raise ValueError(f"Unsupported video format: {format}")

    def read_meta_info(self):
        self.metadata = metadata_K2IS(self.gtg_path_filename)

    def concatenate_metadata(self, attributs):
        result_string = ""
        for attribut in attributs:
            if attribut == 'get_actual_time':
                a = self.actual_frame * self.metadata.time_per_frame * 1E-9
                b = "{:.1f}".format(a)
                result_string += f"{attribut}: {b} s\n"
            elif hasattr(self.metadata, attribut):
                func = getattr(self.metadata, attribut)
                result, unity = func()
                result_string += f"{attribut}: {result} {unity}\n"
        return result_string


class metadata_K2IS:
    def __init__(self, file_path_gtg):
        # read the header
        with open(file_path_gtg, "rb") as f:
            reader = DigitalMicrographReader(f)
            reader.parse_file()
        self.reader_dict = reader.tags_dict
        self.device = self.reader_dict['Acquisition']['Device']
        self.time_per_frame = self.reader_dict['Acquisition']['Frame']['Sequence']['Exposure Time (ns)']
        # print(reader.tags_dict)

    def get_active_size(self):
        return self.device['Active Size (pixels)'], "pixels"

    def get_pixel_size(self):
        return self.device['CCD']['Pixel Size (um)'], "um"

    def get_active_sensor_region(self):
        return self.reader_dict['Acquisition']['Frame']['Active Sensor Region'], ""

    def get_binning(self):
        return self.reader_dict['Acquisition']['Frame']['Area']['Transform']['Transform List']['TagGroup0'][
            'Binning'], ""

    def get_xscale(self):
        return self.reader_dict['Acquisition']['Frame']['Calibration']['0']['Scale'], \
            self.reader_dict['Acquisition']['Frame']['Calibration']['0']['Unit']

    def get_yscale(self):
        return self.reader_dict['Acquisition']['Frame']['Calibration']['1']['Scale'], \
            self.reader_dict['Acquisition']['Frame']['Calibration']['1']['Unit']

    def get_temperature(self):
        return self.device['Temperature (C)'], "°C"

    def get_exposure_time(self):
        return self.device['Exposure Time (ns)'], "ns"

    def get_high_level_param(self, param):
        return self.reader_dict['Acquisition']['Parameters']['High Level'][param], ""

    def get_all_data(self):
        return self.reader_dict, ""


if __name__ == "__main__":
    # Create K2-IS processing object
    processor = ImageProcessor()

    # Define output directory
    processor.output_directory = "S:/532-PHELIQS/532.2-Nanofils2-6/Nanomax_2023/Capture22-40"

    # Load data
    processor.load_data(r"E:\capture22\capture22_.gtg")

    # Read some metadata
    value, unit = processor.metadata.get_pixel_size()
    print(f"Pixel Size: {value} {unit}")
    value, unit = processor.metadata.get_xscale()
    print(f"Scale: {value} {unit}")

    # Process images
    processor.extract_image(sum_frames=40, img_format='tif', dtype='8bit', normalization_mode='percentile',
                            mylegend=["get_xscale", "get_actual_time"], drift_correction=False)

    # Create movie from extracted 8bit images
    processor.compute_movies()
