import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom

class ScrollablePlotter:
    # This class remains unchanged.
    def __init__(self, target_ct_vol, source_ct_vol, target_uid, source_uid, transform_info):
        self.target_ct_vol, self.source_ct_vol = target_ct_vol, source_ct_vol
        self.num_slices, self.slice_index = target_ct_vol.shape[0], target_ct_vol.shape[0] // 2
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='black')
        self.fig.canvas.manager.set_window_title('CT-to-CT Registration Viewer')
        title_text = (f"Target CT (Plan Sim) UID: ...{target_uid[-12:]}\n"
                      f"Source CT (Initial) UID: ...{source_uid[-12:]}\n"
                      f"{transform_info}")
        self.fig.suptitle(title_text, color='white', fontsize=12, y=0.96)
        self.im_target_ct = self.axes[0].imshow(self.target_ct_vol[self.slice_index], cmap='gray')
        self.im_source_ct = self.axes[1].imshow(self.source_ct_vol[self.slice_index], cmap='gray')
        self.im_overlay_target = self.axes[2].imshow(self.target_ct_vol[self.slice_index], cmap='gray')
        self.im_overlay_source = self.axes[2].imshow(self.source_ct_vol[self.slice_index], cmap='viridis', alpha=0.5)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.update()

    def on_scroll(self, event):
        if event.button == 'up': self.slice_index = (self.slice_index + 1) % self.num_slices
        else: self.slice_index = (self.slice_index - 1) % self.num_slices
        self.update()

    def update(self):
        target_slice, source_slice = self.target_ct_vol[self.slice_index], self.source_ct_vol[self.slice_index]
        wc, ww = 40, 400; vmin_ct, vmax_ct = wc - ww / 2, wc + ww / 2
        target_windowed, source_windowed = np.clip(target_slice, vmin_ct, vmax_ct), np.clip(source_slice, vmin_ct, vmax_ct)
        self.im_target_ct.set_data(target_windowed)
        self.im_source_ct.set_data(source_windowed)
        self.im_overlay_target.set_data(target_windowed)
        self.im_overlay_source.set_data(source_windowed)
        slice_title = f"Slice: {self.slice_index} / {self.num_slices - 1}"
        self.axes[0].set_title(f"Target CT (Plan Sim)\n{slice_title}", color='white')
        self.axes[1].set_title(f"Source CT (Initial)\n{slice_title}", color='white')
        self.axes[2].set_title(f"CT/CT Overlay\n{slice_title}", color='white')
        for ax in self.axes: ax.axis('off'); ax.set_facecolor('black')
        self.fig.tight_layout(rect=[0, 0, 1, 0.9])
        self.fig.canvas.draw_idle()

def find_ct_to_ct_link(reg_series, ct_series, initial_ct_sid):
    """Searches REG objects to find a link from the initial CT to another CT."""
    for reg_sid, reg_info in reg_series.items():
        reg_header = reg_info['header']
        try:
            for item in reg_header.RegistrationSequence:
                matrix_seq = item.MatrixRegistrationSequence[0].MatrixSequence[0]
                matrix = np.array(matrix_seq.FrameOfReferenceTransformationMatrix, dtype=float)
                if np.allclose(matrix, np.identity(4).flatten()): continue

                source_for_uid = item.FrameOfReferenceUID
                target_for_uid = reg_header.FrameOfReferenceUID

                source_header = ct_series.get(initial_ct_sid, {}).get('header')
                if source_header and source_header.FrameOfReferenceUID == source_for_uid:
                    target_ct_sid = next((sid for sid, info in ct_series.items() 
                                          if sid != initial_ct_sid and info['header'].FrameOfReferenceUID == target_for_uid), None)
                    if target_ct_sid:
                        return initial_ct_sid, target_ct_sid, matrix
        except (AttributeError, IndexError): continue
    return None, None, None

def process_ct_to_ct_registration(dicom_directory):
    print("--- Starting Automated CT-to-CT Registration ---")
    
    # 1. Catalog all series
    series_map = {}
    for root, _, files in os.walk(dicom_directory):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                header = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                if 'SeriesInstanceUID' in header:
                    series_map.setdefault(header.SeriesInstanceUID, {'header': header, 'modality': header.Modality.upper()})
            except pydicom.errors.InvalidDicomError: continue
    
    # 2. Sort into categories
    ct_series = {sid: info for sid, info in series_map.items() if info['modality'] == 'CT'}
    pet_series = {sid: info for sid, info in series_map.items() if info['modality'] in ['PT', 'PET']}
    reg_series = {sid: info for sid, info in series_map.items() if info['modality'] == 'REG'}

    if len(ct_series) < 2 or not pet_series: raise ValueError("Required data not found: Need at least two CT series and one PET series.")
    print(f"Found {len(ct_series)} CT series and {len(reg_series)} REG objects.")

    # 3. Identify the "Initial CT" (taken with the PET)
    pet_study_uid = list(pet_series.values())[0]['header'].StudyInstanceUID
    initial_ct_sid = next((sid for sid, info in ct_series.items() if info['header'].StudyInstanceUID == pet_study_uid), None)
    if not initial_ct_sid: raise ValueError("Could not identify the initial CT scan associated with the PET.")
    
    # 4. Find the link between the initial CT and the plan CT
    source_uid, target_uid, matrix = find_ct_to_ct_link(reg_series, ct_series, initial_ct_sid)

    if source_uid is None:
        raise ValueError("Could not find a valid REG object that links the initial CT to another CT scan.")
    
    # 5. Only now that a match is confirmed, print success and process
    print(f"\nSUCCESS: Found REG object linking the two CT scans.")
    matrix_4x4 = np.reshape(matrix, (4, 4))
    inverse_matrix = np.linalg.inv(matrix_4x4)
    
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(inverse_matrix[0:3, 0:3].flatten())
    affine.SetTranslation(inverse_matrix[0:3, 3])
    transform_info = "Transformation: Automatically found in matching REG object"

    target_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dicom_directory, target_uid))
    source_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dicom_directory, source_uid))

    # Get the header for the source CT to find the correct default value for padding.
    # This is typically the value for air, which corresponds to the RescaleIntercept.
    source_header = ct_series[source_uid]['header']
    default_padding_value = float(source_header.get('RescaleIntercept', -1000))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(affine)
    resampler.SetDefaultPixelValue(default_padding_value)
    resampled_source_image = resampler.Execute(source_image)

    target_array = sitk.GetArrayFromImage(target_image)
    resampled_source_array = sitk.GetArrayFromImage(resampled_source_image)
    
    print("--- Processing Complete ---")
    return target_array, resampled_source_array, target_uid, source_uid, transform_info, matrix_4x4

if __name__ == '__main__':
    dicom_folder = '../WholePelvis'
    try:
        target_ct_data, source_ct_data, target_uid, source_uid, transform_info, matrix = process_ct_to_ct_registration(dicom_directory=dicom_folder)
        
        print("\n--- Found Valid CT-to-CT Registration ---")
        print(f"Target CT (Plan Sim) UID: {target_uid}")
        print(f"Source CT (Initial) UID: {source_uid}")
        print("Applied 4x4 Transformation Matrix:")
        with np.printoptions(precision=4, suppress=True):
            print(matrix)
        print("----------------------------------------\n")

        plotter = ScrollablePlotter(target_ct_vol=target_ct_data, source_ct_vol=source_ct_data, target_uid=target_uid, source_uid=source_uid, transform_info=transform_info)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

