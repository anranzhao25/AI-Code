import SimpleITK as sitk
import os
import shutil
import pydicom
import numpy as np
import argparse
from collections import defaultdict

def bin_dicom_data(root_dir):
    print(f"Scanning directory: {root_dir}")
    binned_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if not f.lower().endswith('.dcm'):
                continue
                
            file_path = os.path.join(dirpath, f)
            try:
                dcm = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                modality = dcm.Modality
                study_date = dcm.StudyDate
                series_uid = dcm.SeriesInstanceUID
                
                if series_uid not in binned_data[modality][study_date]:
                    desc = dcm.get("SeriesDescription", "N/A")
                    binned_data[modality][study_date][series_uid] = {
                        'description': desc,
                        'files': []
                    }
                
                binned_data[modality][study_date][series_uid]['files'].append(file_path)

            except Exception as e:
                print(f"Warning: Could not read {file_path}. {e}")
                
    print("Scan complete. Found data:")
    for modality, dates in binned_data.items():
        print(f"  Modality: {modality}")
        for date, series in dates.items():
            print(f"    Date: {date} ({len(series)} series)")
            for uid, data in series.items():
                print(f"      - {uid} ({data['description']}): {len(data['files'])} files")
    
    return binned_data

def select_series(modality_data, date, description):
    series_on_date = modality_data.get(date)
    
    if not series_on_date:
        print(f"Error: No series found for {description} on date {date}")
        return None
        
    if len(series_on_date) == 1:
        uid = list(series_on_date.keys())[0]
        print(f"Automatically selected for {description}: {uid} ({series_on_date[uid]['description']})")
        return uid
        
    print(f"\nMultiple series found for {description} on {date}. Please choose one:")
    series_list = list(series_on_date.items())
    for i, (uid, data) in enumerate(series_list):
        print(f"  [{i+1}] {uid} ({data['description']}) - {len(data['files'])} files")
        
    while True:
        try:
            choice = input(f"Enter number (1-{len(series_list)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(series_list):
                return series_list[choice_idx][0]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def load_dicom_series(file_list):
    print(f"Loading {len(file_list)} files into 3D volume...")
    
    slices = []
    for f in file_list:
        try:
            dcm = pydicom.dcmread(f)
            slices.append((float(dcm.ImagePositionPatient[2]), f))
        except Exception as e:
            print(f"Warning: Could not read slice {f} for sorting. {e}")

    slices.sort(key=lambda x: x[0])
    sorted_files = [s[1] for s in slices]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    image_3d = reader.Execute()
    return image_3d

def copy_dicom_series(file_list, output_dir):
    print(f"Copying {len(file_list)} files to {output_dir}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for f in file_list:
        shutil.copy(f, os.path.join(output_dir, os.path.basename(f)))
    print("Copy complete.")

def save_as_dicom_series(image_3d, output_dir, reference_dicom_files):
    print(f"Saving resampled 3D image to {output_dir} as DICOM series...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    
    slices = []
    for f in reference_dicom_files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append((float(dcm.ImagePositionPatient[2]), f))
        except:
            continue
    slices.sort(key=lambda x: x[0])
    sorted_ref_files = [s[1] for s in slices]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_ref_files)
    reader.LoadPrivateTagsOn()
    reader.Execute()
    
    for i in range(image_3d.GetDepth()):
        image_slice = image_3d[:, :, i]
        
        tags_to_copy = reader.GetMetaDataKeys(i)
        for tag in tags_to_copy:
            image_slice.SetMetaData(tag, reader.GetMetaData(i, tag))
            
        image_slice.SetMetaData("0020|000e", sitk.CreateUniqueSeriesInstanceUID()) 
        image_slice.SetMetaData("0008|0060", "PT") 
        image_slice.SetMetaData("0008|103e", "PET Resampled to Post-Chemo CT")
        
        file_path = os.path.join(output_dir, f"slice_{i:04d}.dcm")
        writer.SetFileName(file_path)
        writer.Execute(image_slice)
        
    print(f"Successfully saved {image_3d.GetDepth()} resampled PET slices.")

def main(args):
    binned_data = bin_dicom_data(args.input_dir)
    
    ct_data = binned_data.get('CT', {})
    pet_data = binned_data.get('PT', {})
    
    if not ct_data or not pet_data:
        print("Error: Input directory must contain both CT and PT modalities.")
        return

    all_dates = sorted(list(ct_data.keys()) + list(pet_data.keys()))
    if len(all_dates) < 2:
        print("Error: At least two different study dates (pre/post) are required.")
        return
        
    oldest_date = all_dates[0]
    newest_date = all_dates[-1]
    
    print(f"\nOldest study date: {oldest_date} (assumed PRE-chemo)")
    print(f"Newest study date: {newest_date} (assumed POST-chemo)")
    
    post_chemo_ct_uid = select_series(ct_data, newest_date, "Post-Chemo CT")
    pre_chemo_ct_uid = select_series(ct_data, oldest_date, "Pre-Chemo CT")
    pre_chemo_pet_uid = select_series(pet_data, oldest_date, "Pre-Chemo PET")
    
    if not all([post_chemo_ct_uid, pre_chemo_ct_uid, pre_chemo_pet_uid]):
        print("\nError: Could not identify all required series. Exiting workflow.")
        return
        
    ct_post_files = ct_data[newest_date][post_chemo_ct_uid]['files']
    ct_pre_files = ct_data[oldest_date][pre_chemo_ct_uid]['files']
    pet_pre_files = pet_data[oldest_date][pre_chemo_pet_uid]['files']
    
    print(f"\nFound {len(ct_post_files)} files for Post-Chemo CT.")
    print(f"Found {len(ct_pre_files)} files for Pre-Chemo CT.")
    print(f"Found {len(pet_pre_files)} files for Pre-Chemo PET.")
    
    ct_output_dir = os.path.join(args.output_dir, 'CT_PostChemo_Untouched')
    copy_dicom_series(ct_post_files, ct_output_dir)

    print("\nLoading images into 3D volumes...")
    fixed_image = load_dicom_series(ct_post_files)
    moving_ct = load_dicom_series(ct_pre_files)
    moving_pet = load_dicom_series(pet_pre_files)
    
    print("\nStarting registration... (This may take a few minutes)")
    
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                          moving_ct, 
                                                          sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = R.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                sitk.Cast(moving_ct, sitk.sitkFloat32))
    
    print("Registration complete.")
    print(f"Final Transform: {final_transform}")

    print("\nResampling PET onto Post-Chemo CT grid...")
    
    resampled_pet = sitk.Resample(moving_pet,
                                  fixed_image,
                                  final_transform,
                                  sitk.sitkLinear,
                                  0.0,
                                  moving_pet.GetPixelID())
                                  
    print("Resampling complete.")

    pet_output_dir = os.path.join(args.output_dir, 'PET_Resampled_to_CT')
    save_as_dicom_series(resampled_pet, pet_output_dir, ct_post_files)

    print("\n--- Workflow Complete ---")
    print(f"Find your new files in: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Pre/Post Chemo PET/CT Registration.")
    parser.add_argument("-i", "--input_dir", 
                        required=True, 
                        help="Root directory containing all DICOM files for a single patient.")
    parser.add_argument("-o", "--output_dir", 
                        required=True, 
                        help="Base directory to save the processed output files.")
    
    args = parser.parse_args()
    
    main(args)