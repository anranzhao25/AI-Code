import SimpleITK as sitk
import numpy as np
import os
import pydicom
import pydicom.uid
import shutil
import time

def write_dicom_series(image, reference_header, output_directory, series_description):

    os.makedirs(output_directory, exist_ok=True)
    
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    min_val, max_val = stats.GetMinimum(), stats.GetMaximum()
    int_type = np.int16
    int_min, int_max = np.iinfo(int_type).min, np.iinfo(int_type).max
    if max_val == min_val:
        rescale_slope = 1.0
        rescale_intercept = min_val
    else:
        rescale_slope = (max_val - min_val) / (int_max - int_min)
        rescale_intercept = min_val - rescale_slope * int_min
    rescaled_image = sitk.RescaleIntensity(image, outputMinimum=int_min, outputMaximum=int_max)
    image_to_write = sitk.Cast(rescaled_image, sitk.sitkInt16)
    
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    new_series_uid = pydicom.uid.generate_uid()
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    tags_to_copy = [
        "0010|0010", "0010|0020", "0010|0030", "0010|0040", "0020|000d",
        "0008|0020", "0008|0030", "0008|1030", "0008|0050", "0018|5100"
    ]

    for i in range(image_to_write.GetDepth()):
        image_slice = image_to_write[:, :, i]

        for tag_str in tags_to_copy:
            group, element = int(tag_str[:4], 16), int(tag_str[5:], 16)
            tag_key = pydicom.tag.Tag(group, element)
            if tag_key in reference_header:
                value = str(reference_header[tag_key].value)
                if value:
                    image_slice.SetMetaData(tag_str, value)
        
        image_slice.SetMetaData("0008|0012", modification_date)
        image_slice.SetMetaData("0008|0013", modification_time)
        image_slice.SetMetaData("0020|000e", new_series_uid)
        image_slice.SetMetaData("0008|0018", pydicom.uid.generate_uid())
        image_slice.SetMetaData("0020|0013", str(i + 1))
        image_slice.SetMetaData("0008|103e", series_description)
        image_slice.SetMetaData("0008|0060", "PT")
        image_slice.SetMetaData("0028|0103", "1")
        image_slice.SetMetaData("0028|1052", str(rescale_intercept))
        image_slice.SetMetaData("0028|1053", str(rescale_slope))
        
        filename = os.path.join(output_directory, f"slice_{i+1:04d}.dcm")
        writer.SetFileName(filename)
        writer.Execute(image_slice)

def process_and_save_patient_data(input_dir, output_dir):
    print(f"--- Starting Data Processing for Patient in: {os.path.basename(input_dir)} ---")
    
    series_map = {}
    for root, _, files in os.walk(input_dir):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                header = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                if 'SeriesInstanceUID' in header:
                    series_map.setdefault(header.SeriesInstanceUID, {'header': header, 'modality': header.Modality.upper()})
            except pydicom.errors.InvalidDicomError: continue
    
    ct_series = {sid: info for sid, info in series_map.items() if info['modality'] == 'CT'}
    pet_series = {sid: info for sid, info in series_map.items() if info['modality'] in ['PT', 'PET']}
    reg_series = {sid: info for sid, info in series_map.items() if info['modality'] == 'REG'}

    if not ct_series or not pet_series: raise ValueError("Could not find both CT and PET series.")
    print(f"Found {len(reg_series)} REG objects to search.")

    found_match = None
    for reg_sid, reg_info in reg_series.items():
        if found_match: break
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
                    found_match = {"ct_sid": matching_ct_sid, "pet_sid": matching_pet_sid, "matrix": matrix}
                    break
        except (AttributeError, IndexError): continue

    if found_match is None:
        raise ValueError("Could not find a valid REG object that links any of the found PET and CT scans.")

    ct_series_id = found_match["ct_sid"]
    pet_series_id = found_match["pet_sid"]
    matrix = found_match["matrix"]

    print(f"SUCCESS: Found REG linking CT (...{ct_series_id[-12:]}) and PET (...{pet_series_id[-12:]})")

    matrix_4x4 = np.reshape(matrix, (4, 4))
    
    inverse_matrix = np.linalg.inv(matrix_4x4)
    
    affine = sitk.AffineTransform(3)
    
    rotation = inverse_matrix[0:3, 0:3]
    affine.SetMatrix(rotation.flatten())

    translation = inverse_matrix[0:3, 3]
    flipped_translation = [-translation[0], -translation[1], translation[2]]
    affine.SetTranslation(flipped_translation)

    ct_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(input_dir, ct_series_id))
    pet_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(input_dir, pet_series_id))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(affine)
    resampler.SetDefaultPixelValue(0.0)
    resampled_pet_image = resampler.Execute(pet_image)
    
    resampled_pet_image.SetSpacing(ct_image.GetSpacing())
    resampled_pet_image.SetOrigin(ct_image.GetOrigin())
    resampled_pet_image.SetDirection(ct_image.GetDirection())

    print("--- Registration Complete. Now saving new DICOM files. ---")

    patient_id = ct_series[ct_series_id]['header'].get("PatientID", "UnknownPatient")
    patient_output_folder = os.path.join(output_dir, patient_id)
    
    ct_folder_name = f"CT_{ct_series_id[-12:]}"
    pet_folder_name = f"PET_resampled_to_CT_{ct_series_id[-12:]}"

    ct_output_path = os.path.join(patient_output_folder, ct_folder_name)
    pet_output_path = os.path.join(patient_output_folder, pet_folder_name)

    original_ct_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(input_dir, ct_series_id)
    if os.path.exists(ct_output_path): shutil.rmtree(ct_output_path)
    os.makedirs(ct_output_path, exist_ok=True)
    for f in original_ct_files:
        shutil.copy(f, ct_output_path)
    print(f"Original CT series saved to: {ct_output_path}")
    
    if os.path.exists(pet_output_path): shutil.rmtree(pet_output_path)
    reference_pet_header = pet_series[pet_series_id]['header']
    write_dicom_series(resampled_pet_image, reference_pet_header, pet_output_path, "PET RESAMPLED")
    print(f"New resampled PET series saved to: {pet_output_path}")

if __name__ == '__main__':
    input_folder = '../WholePelvis'
    output_folder = '../CorrectedImages'

    os.makedirs(output_folder, exist_ok=True)
    
    try:
        process_and_save_patient_data(input_dir=input_folder, output_dir=output_folder)
        print("\n--- Pipeline finished successfully! ---")
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")