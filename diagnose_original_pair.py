import os
import pydicom
import numpy as np

def print_header_comparison(header1, header2, header3, title1, title2, title3):
    """Prints a side-by-side comparison of key geometric DICOM tags for three headers."""
    
    tags_to_compare = {
        "SeriesInstanceUID": "0020,000e",
        "StudyInstanceUID": "0020,000d",
        "FrameOfReferenceUID": "0020,0052",
        "ImagePositionPatient": "0020,0032",
        "ImageOrientationPatient": "0020,0037",
        "PixelSpacing": "0028,0030",
        "SliceThickness": "0018,0050",
    }
    
    print("-" * 120)
    print(f"{title1:<38} | {title2:<38} | {title3:<38}")
    print("-" * 120)
    
    for name, tag_str in tags_to_compare.items():
        # Convert string tag to a pydicom Tag object to prevent warnings
        group = int(tag_str.split(',')[0], 16)
        element = int(tag_str.split(',')[1], 16)
        tag = pydicom.tag.Tag(group, element)

        val1 = "Not Found"
        val2 = "Not Found"
        val3 = "Not Found"
        
        # Safely get the value if the tag exists
        if tag in header1: val1 = header1[tag].value
        if tag in header2: val2 = header2[tag].value
        if tag in header3: val3 = header3[tag].value

        # Format for better readability
        def format_val(v):
            if isinstance(v, (list, pydicom.multival.MultiValue)):
                return np.array2string(np.array(v, dtype=float), precision=4, suppress_small=True)
            return str(v)

        print(f"{name:<25}: {format_val(val1):<38} | {format_val(val2):<38} | {format_val(val3):<38}")
        
    print("-" * 120)
    
    # Also print the full registration sequence from the RTSTRUCT
    print("\n--- Full RegistrationSequence from RTSTRUCT File ---")
    # Use pydicom's keyword access for safety
    if 'RegistrationSequence' in header3:
        print(header3.RegistrationSequence)
    else:
        print("RegistrationSequence not found in RTSTRUCT header.")
    print("-" * 120)


def diagnose_original_pair(input_dir):
    """
    Finds the original PET, its corresponding same-day CT, and the RTSTRUCT file
    that links them, and prints a detailed comparison of their geometric metadata.
    """
    print(f"--- Running Diagnostic Scan on: {os.path.basename(input_dir)} ---")
    
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
    rtstruct_series = {sid: info for sid, info in series_map.items() if info['modality'] == 'RTSTRUCT'}

    if not ct_series or not pet_series: raise ValueError("Could not find both CT and PET series.")

    if not pet_series: raise ValueError("No PET series found in the directory.")
    pet_header = list(pet_series.values())[0]['header']
    pet_study_uid = pet_header.StudyInstanceUID
    
    original_ct_sid = next((sid for sid, info in ct_series.items() if info['header'].StudyInstanceUID == pet_study_uid), None)
    if not original_ct_sid: raise ValueError("Could not find the original CT scan associated with the PET's study.")
    
    registration_struct_sid = next((sid for sid, info in rtstruct_series.items() if info['header'].StudyInstanceUID == pet_study_uid), None)
    if not registration_struct_sid:
        raise ValueError("Could not find an RTSTRUCT file in the same study as the PET/CT pair.")
    
    ct_header = ct_series[original_ct_sid]['header']
    struct_header = rtstruct_series[registration_struct_sid]['header']

    print("\nFound a potential original PET/CT/RTSTRUCT trio. Displaying geometric information:")
    print_header_comparison(ct_header, pet_header, struct_header, "Original CT", "Original PET", "RTSTRUCT File")


if __name__ == '__main__':
    input_folder = '../WholePelvis'
    try:
        diagnose_original_pair(input_dir=input_folder)
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")