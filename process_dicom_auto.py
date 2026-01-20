import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom

class ScrollablePlotter:
    # This class remains unchanged and handles the interactive plotting.
    def __init__(self, ct_volume, pet_volume, ct_uid, pet_uid, transform_matrix_str):
        self.ct_volume, self.pet_volume = ct_volume, pet_volume
        self.num_slices, self.slice_index = ct_volume.shape[0], ct_volume.shape[0] // 2
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='black')
        self.fig.canvas.manager.set_window_title('PET/CT Interactive Viewer')
        title_text = (f"Target (CT) UID: ...{ct_uid[-12:]}\n"
                      f"Source (PET) UID: ...{pet_uid[-12:]}\n"
                      f"{transform_matrix_str}")
        self.fig.suptitle(title_text, color='white', fontsize=12, y=0.96)
        self.im_ct = self.axes[0].imshow(self.ct_volume[self.slice_index], cmap='gray')
        self.im_pet = self.axes[1].imshow(self.pet_volume[self.slice_index], cmap='hot')
        self.im_overlay_ct = self.axes[2].imshow(self.ct_volume[self.slice_index], cmap='gray')
        self.im_overlay_pet = self.axes[2].imshow(self.pet_volume[self.slice_index], cmap='hot', alpha=0.6)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.update()

    def on_scroll(self, event):
        if event.button == 'up': self.slice_index = (self.slice_index + 1) % self.num_slices
        else: self.slice_index = (self.slice_index - 1) % self.num_slices
        self.update()

    def update(self):
        ct_slice, pet_slice = self.ct_volume[self.slice_index], self.pet_volume[self.slice_index]
        wc, ww = 40, 400; vmin_ct, vmax_ct = wc - ww / 2, wc + ww / 2
        ct_slice_windowed = np.clip(ct_slice, vmin_ct, vmax_ct)
        pet_data = pet_slice[pet_slice > 0]
        vmin_pet, vmax_pet = (0, 1)
        if pet_data.size > 0: vmin_pet, vmax_pet = np.percentile(pet_data, [95, 99.8])
        self.im_ct.set_data(ct_slice_windowed)
        self.im_pet.set_data(pet_slice); self.im_pet.set_clim(vmin_pet, vmax_pet)
        self.im_overlay_ct.set_data(ct_slice_windowed)
        self.im_overlay_pet.set_data(pet_slice); self.im_overlay_pet.set_clim(vmin_pet, vmax_pet)
        slice_title = f"Slice: {self.slice_index} / {self.num_slices - 1}"
        for i, title in enumerate(["CT Image", "PET Image", "PET/CT Overlay"]):
            self.axes[i].set_title(f"{title}\n{slice_title}", color='white')
            self.axes[i].axis('off'); self.axes[i].set_facecolor('black')
        self.fig.tight_layout(rect=[0, 0, 1, 0.9])
        self.fig.canvas.draw_idle()

def process_dicom_data(dicom_directory):
    print("--- Starting Final Automated PET/CT Processing ---")
    
    # Step 1: Catalog all series
    series_map = {}
    for root, _, files in os.walk(dicom_directory):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                header = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                if 'SeriesInstanceUID' in header:
                    series_map.setdefault(header.SeriesInstanceUID, {'header': header, 'modality': header.Modality.upper()})
            except pydicom.errors.InvalidDicomError: continue
    
    # Step 2: Sort into categories
    ct_series = {sid: info for sid, info in series_map.items() if info['modality'] == 'CT'}
    pet_series = {sid: info for sid, info in series_map.items() if info['modality'] in ['PT', 'PET']}
    reg_series = {sid: info for sid, info in series_map.items() if info['modality'] == 'REG'}

    if not ct_series or not pet_series: raise ValueError("Could not find both CT and PET series.")
    print(f"Found {len(reg_series)} REG objects to search.")

    # Step 3: Find the correct registration trio
    found_match = None
    for reg_sid, reg_info in reg_series.items():
        if found_match: break # Exit loop once a match is found
        reg_header = reg_info['header']
        try:
            target_for_uid = reg_header.FrameOfReferenceUID
            matching_ct_sid = next((sid for sid, info in ct_series.items() if info['header'].FrameOfReferenceUID == target_for_uid), None)
            if not matching_ct_sid: continue
            for item in reg_header.RegistrationSequence:
                matrix_seq = item.MatrixRegistrationSequence[0].MatrixSequence[0]
                matrix = np.array(matrix_seq.FrameOfReferenceTransformationMatrix, dtype=float)
                if np.allclose(matrix, np.identity(4).flatten()): continue
                source_for_uid = item.FrameOfReferenceUID
                matching_pet_sid = next((sid for sid, info in pet_series.items() if info['header'].FrameOfReferenceUID == source_for_uid), None)
                if matching_pet_sid:
                    found_match = {
                        "ct_sid": matching_ct_sid,
                        "pet_sid": matching_pet_sid,
                        "matrix": matrix
                    }
                    break # Exit inner loop
        except (AttributeError, IndexError): continue

    # Step 4: Process the images only if a match was found
    if found_match is None:
        raise ValueError("Could not find a valid REG object that links any of the found PET and CT scans.")

    ct_series_id = found_match["ct_sid"]
    pet_series_id = found_match["pet_sid"]
    matrix = found_match["matrix"]

    ct_description = ct_series[ct_series_id]['header'].get("SeriesDescription", "N/A")
    pet_description = pet_series[pet_series_id]['header'].get("SeriesDescription", "N/A")
    print("\n--- Found a Valid Registration Match ---")
    print(f"Target CT Description: {ct_description} (UID: ...{ct_series_id[-12:]})")
    print(f"Source PET Description: {pet_description} (UID: ...{pet_series_id[-12:]})")
    print("----------------------------------------\n")

    matrix_4x4 = np.reshape(matrix, (4, 4))
    inverse_matrix = np.linalg.inv(matrix_4x4)
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(inverse_matrix[0:3, 0:3].flatten())
    affine.SetTranslation(inverse_matrix[0:3, 3])
    transform_matrix_str = "Transformation: Automatically found in matching REG object"

    ct_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dicom_directory, ct_series_id))
    pet_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dicom_directory, pet_series_id))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(affine)
    resampler.SetDefaultPixelValue(0.0)
    resampled_pet_image = resampler.Execute(pet_image)

    ct_array = sitk.GetArrayFromImage(ct_image)
    resampled_pet_array = sitk.GetArrayFromImage(resampled_pet_image)
    
    print("--- Processing Complete ---")
    return ct_array, resampled_pet_array, ct_series_id, pet_series_id, transform_matrix_str

if __name__ == '__main__':
    dicom_folder = '../WholePelvis'
    try:
        ct_data, pet_data, ct_uid, pet_uid, transform_info = process_dicom_data(dicom_directory=dicom_folder)
        plotter = ScrollablePlotter(ct_volume=ct_data, pet_volume=pet_data, ct_uid=ct_uid, pet_uid=pet_uid, transform_matrix_str=transform_info)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")