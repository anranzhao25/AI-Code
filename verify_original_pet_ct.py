import SimpleITK as sitk
import numpy as np
import os
import pydicom
import pydicom.uid
import shutil
import time

def write_dicom_series(image, reference_header, output_directory, series_description):
    """
    Saves a SimpleITK image as a new DICOM series by rescaling float data to integers,
    and explicitly writing geometry information for each slice.
    """
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
                if value: image_slice.SetMetaData(tag_str, value)
        
        position = image_to_write.TransformIndexToPhysicalPoint((0, 0, i))
        position_str = "\\".join(map(str, position))
        image_slice.SetMetaData("0020|0032", position_str)
        
        direction = image_to_write.GetDirection()
        orientation_str = "\\".join(map(str, [direction[0], direction[3], direction[6], direction[1], direction[4], direction[7]]))
        image_slice.SetMetaData("0020|0037", orientation_str)
        
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

def process_original_pet_ct(input_dir, output_dir):
    """
    Finds the original PET and its corresponding same-day CT, calculates their
    inherent registration from their metadata, and saves them as a new co-registered pair.
    """
    print(f"--- Processing Original PET/CT Pair from: {os.path.basename(input_dir)} ---")
    
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

    if not ct_series or not pet_series: raise ValueError("Could not find both CT and PET series.")

    # Step 1: Reliably identify the "Original CT" by finding the one with the same StudyInstanceUID as the PET.
    # This ensures we are not grabbing a CT from a different day.
    pet_header = list(pet_series.values())[0]['header']
    pet_study_uid = pet_header.StudyInstanceUID
    
    initial_ct_sid = next((sid for sid, info in ct_series.items() if info['header'].StudyInstanceUID == pet_study_uid), None)
    if not initial_ct_sid: raise ValueError("Could not find the original CT scan associated with the PET's study.")
    
    pet_series_id = pet_header.SeriesInstanceUID
    ct_series_id = initial_ct_sid
    ct_header = ct_series[ct_series_id]['header']

    # Step 2: Verify they are already in the same coordinate system (Frame of Reference).
    if ct_header.FrameOfReferenceUID != pet_header.FrameOfReferenceUID:
        raise ValueError("Original PET and CT do not share a Frame of Reference. A REG object would be needed.")
    
    # Step 3: Calculate the inherent transformation, which is the minor translation between their origins.
    # This is the real transformation matrix based on the metadata.
    ct_origin = np.array(ct_header.ImagePositionPatient, dtype=float)
    pet_origin = np.array(pet_header.ImagePositionPatient, dtype=float)
    translation_vector = pet_origin - ct_origin
    
    # Create the 4x4 transformation matrix (this is for display purposes).
    matrix = np.identity(4)
    matrix[0:3, 3] = translation_vector

    print("\n--- Found Original PET/CT Pair ---")
    print(f"Original CT Series UID: {ct_series_id}")
    print(f"Original PET Series UID: {pet_series_id}")
    print("Inherent Transformation Matrix (Translation Only):")
    with np.printoptions(precision=4, suppress=True):
        print(matrix)
    print("------------------------------------\n")

    # Step 4: Load the images and create the transform object for resampling.
    ct_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(input_dir, ct_series_id))
    pet_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(input_dir, pet_series_id))

    # The transform is the calculated translation vector.
    affine_transform = sitk.TranslationTransform(3, translation_vector)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image) # Set the output grid to match the CT.
    resampler.SetInterpolator(sitk.sitkLinear)
    # CRITICAL: For resampling, we must use the INVERSE of the transform.
    resampler.SetTransform(affine_transform.GetInverse())
    resampler.SetDefaultPixelValue(0.0)
    resampled_pet_image = resampler.Execute(pet_image)
    
    # Step 5: Explicitly copy the CT's geometry to ensure the new DICOMs are perfect.
    resampled_pet_image.SetSpacing(ct_image.GetSpacing())
    resampled_pet_image.SetOrigin(ct_image.GetOrigin())
    resampled_pet_image.SetDirection(ct_image.GetDirection())
    
    print("--- Resampling Complete. Now saving new DICOM files. ---")

    patient_id = ct_header.get("PatientID", "UnknownPatient")
    patient_output_folder = os.path.join(output_dir, f"{patient_id}_Original")
    
    ct_folder_name = f"CT_original_{ct_series_id[-12:]}"
    pet_folder_name = f"PET_resampled_to_CT_{ct_series_id[-12:]}"

    ct_output_path = os.path.join(patient_output_folder, ct_folder_name)
    pet_output_path = os.path.join(patient_output_folder, pet_folder_name)

    original_ct_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(input_dir, ct_series_id)
    if os.path.exists(ct_output_path): shutil.rmtree(ct_output_path)
    os.makedirs(ct_output_path, exist_ok=True)
    for f in original_ct_files: shutil.copy(f, ct_output_path)
    print(f"Original CT series saved to: {ct_output_path}")
    
    if os.path.exists(pet_output_path): shutil.rmtree(pet_output_path)
    write_dicom_series(resampled_pet_image, pet_header, pet_output_path, "PET RESAMPLED TO ORIGINAL CT")
    print(f"New resampled PET series saved to: {pet_output_path}")

if __name__ == '__main__':
    input_folder = '../WholePelvis'
    output_folder = '../CorrectedImages'

    os.makedirs(output_folder, exist_ok=True)
    
    try:
        process_original_pet_ct(input_dir=input_folder, output_dir=output_folder)
        print("\n--- Pipeline finished successfully! ---")
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")