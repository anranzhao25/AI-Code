# %% [markdown]
# # DICOM Data Binning for Patient Analysis (V2)
#
# As an AI medical physics researcher, our first step is to organize the patient's DICOM data from the `WholePelvis` directory.
#
# **V2 Update:** We discovered that `SeriesDescription` is unreliable and often missing (causing many "unknown_series" entries). We are upgrading the binning function to use `SeriesInstanceUID` (0020, 000E), which is the 100% reliable, unique identifier for a scan series.
#
# This notebook will:
# 1.  Recursively scan the target directory for all files ending in `.dcm`.
# 2.  Read `Modality`, `AcquisitionDate`, `SeriesDescription`, and `SeriesInstanceUID`.
# 3.  "Bin" (group) the file paths by Modality -> Date -> SeriesInstanceUID.
# 4.  Print a clean, reliable summary of all data.

# %%
# Import necessary packages
import os
import pydicom
import sys
from collections import defaultdict

# %% [markdown]
# ## Helper Functions (from our notebooks)
#
# * `get_dicom_files`: Scans a directory tree.
# * `bin_by_series_uid`: Sorts a list of file paths by Modality, Date, and (most importantly) SeriesInstanceUID.

# %%
def get_dicom_files(dir_path):
    """
    Recursively walks through a directory and returns a list of all
    files ending with '.dcm'.

    Args:
        dir_path (str): The path to the top-level directory.

    Returns:
        list: A list of full file paths to all .dcm files.
    """
    dicom_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    return dicom_files


def bin_by_series_uid(dicom_files):
    """
    Bins a list of DICOM file paths based on their Modality, AcquisitionDate,
    and SeriesInstanceUID. This is more robust than SeriesDescription.

    Args:
        dicom_files (list): A list of paths to .dcm files.

    Returns:
        dict: A nested dictionary:
              {Modality: {Date: {SeriesUID: {'desc': str, 'files': [list_of_files]}}}}
    """
    # This will be our nested dictionary
    binned_data = {}

    print(f"Binning {len(dicom_files)} files by modality, date, and SeriesInstanceUID...")
    
    # We use enumerate to show progress, as this can be slow
    for i, file_path in enumerate(dicom_files):
        # Print progress update every 100 files
        if (i + 1) % 100 == 0:
            print(f"  ...processed {i + 1} / {len(dicom_files)} files")
            
        try:
            # Read only the header, not the full pixel data (faster)
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)

            # 1. Get Modality
            modality = getattr(dcm, 'Modality', 'unknown_modality')
            
            # 2. Get Date (format YYYYMMDD)
            if hasattr(dcm, 'AcquisitionDate') and dcm.AcquisitionDate:
                date = dcm.AcquisitionDate
            elif hasattr(dcm, 'StudyDate') and dcm.StudyDate:
                date = dcm.StudyDate
            else:
                date = 'unknown_date' # Group files with no date tag

            # 3. Get SeriesInstanceUID (the reliable 100% unique ID)
            if hasattr(dcm, 'SeriesInstanceUID'):
                series_uid = dcm.SeriesInstanceUID
            else:
                series_uid = 'unknown_uid' # This would be very unusual

            # 4. Get Series Description (the human-readable name, for reference)
            if hasattr(dcm, 'SeriesDescription') and dcm.SeriesDescription:
                series_desc = str(dcm.SeriesDescription).strip()
            else:
                series_desc = 'unknown_series' # We still note if it's missing

            # 5. Create nested dictionary keys if they don't exist
            
            # Level 1: Modality
            if modality not in binned_data:
                binned_data[modality] = {}

            # Level 2: Date
            if date not in binned_data[modality]:
                binned_data[modality][date] = {}

            # Level 3: SeriesInstanceUID
            if series_uid not in binned_data[modality][date]:
                # We store both the description and the list of files
                binned_data[modality][date][series_uid] = {
                    'desc': series_desc,
                    'files': []
                }

            # 6. Add the file path to the correct bin
            binned_data[modality][date][series_uid]['files'].append(file_path)

        except pydicom.errors.InvalidDicomError:
            print(f"Warning: Skipping non-DICOM or corrupt file: {file_path}")
        except Exception as e:
            # Catch other errors, e.g., missing tags
            print(f"Warning: Could not read {file_path}. Error: {e}")

    print("Binning complete.")
    return binned_data

# %% [markdown]
# ## Main Execution: Scan and Bin Data
#
# This cell runs the functions to scan your `Desktop/WholePelvis` folder (the data directory)
# as part of the `NodeContourAlgorithms` project.

# %%
# --- 1. Define Data Path ---
# This script is part of the 'NodeContourAlgorithms' project.
# The data is in a separate 'WholePelvis' folder, which is also on the Desktop.
#
# os.path.expanduser('~') gets the path to your home directory
# (e.g., /Users/your_user or C:\Users\your_user)
base_directory = os.path.join(os.path.expanduser('~'), 'Desktop', 'WholePelvis')

print(f"--- Starting DICOM Scan ---")
print(f"Target Directory: {base_directory}\n")

# --- 2. Check if Directory Exists ---
if not os.path.isdir(base_directory):
    print(f"Error: Directory not found at {base_directory}")
    print("Please make sure the 'WholePelvis' folder is on your Desktop.")
    # In a notebook, we'd stop here.
    # We'll use sys.exit() if running as a script, but in a notebook,
    # the 'else' block just won't run.
    all_dicom_files = []
else:
    # --- 3. Get All File Paths ---
    all_dicom_files = get_dicom_files(base_directory)
    print(f"Found {len(all_dicom_files)} total .dcm files.")

# --- 4. Bin Files by Modality, Date, and Series ---
if all_dicom_files:
    # Call the new, more robust binning function
    binned_data = bin_by_series_uid(all_dicom_files)

    print("\n--- Data Binning Summary (by SeriesInstanceUID) ---")
    # Loop through the 3-level nested dictionary
    for modality, date_bins in binned_data.items():
        print(f"\n  Modality: {modality}")
        for date, series_bins in date_bins.items():
            print(f"    Date: {date}")
            for series_uid, series_info in series_bins.items():
                series_desc = series_info['desc']
                files = series_info['files']
                # Print the series description and the count of files
                print(f"      - Series: {series_desc:<40} | Files: {len(files):<5} | UID: {series_uid}")

    # Store the binned data in a variable for use in other cells
    print("\nVariable 'binned_data' is ready for use.")
else:
    if os.path.isdir(base_directory):
        print("No .dcm files were found in the directory.")

# %% [markdown]
# ## Next Steps
#
# With this new `binned_data` structure, all "unknown_series" groups will now be correctly separated by their unique `SeriesInstanceUID`.
#
# Now you can run the *new* processing script (which I am providing in the next file). That script will use this `binned_data` variable to identify the correct file lists and perform the full registration and resampling workflow.

