# Author: Alex Seager
# Last Version: 6/17/25
#
# Description:  this file was used to scrape spectral data from the GNPS online library
# by loading GNPS metadata from local JSON file, download peaks, and store with compound names
# One of the libraries is stored as a CSV here so only use this if you want to download a different
# library

import json
import pandas as pd
import requests
import pickle

def get_entries_from_local_json(filename, max_to_fetch=None):
    print(f"📂 Loading metadata from {filename}...")
    with open(filename, "r") as f:
        data = json.load(f)
    entries = [entry for entry in data if "spectrum_id" in entry and "Compound_Name" in entry]
    if max_to_fetch is not None:
        entries = entries[:max_to_fetch]
    return entries

def fetch_peaks(accession):
    url = (
        f"https://metabolomics-usi.gnps2.org/json/"
        f"?usi1=mzspec:GNPS:GNPS-LIBRARY:accession:{accession}"
    )
    r = requests.get(url)
    if not r.ok:
        print(f"⚠️ Failed to fetch data for {accession}")
        return pd.DataFrame(columns=['mz', 'intensity'])

    entry = r.json()
    peak_data = entry.get('peaks') or entry.get('peaks_json')
    if peak_data is None:
        print(f"⚠️ No peak data found for {accession}")
        return pd.DataFrame(columns=['mz', 'intensity'])

    return pd.DataFrame(peak_data, columns=['mz', 'intensity'])

def main():
    local_json_file = "/Users/alexseager/Desktop/Summer Work 2025/Code/GNPS-SCIEX-LIBRARY.json"
    max_to_fetch = 300  # Adjust as needed

    entries = get_entries_from_local_json(local_json_file, max_to_fetch)
    print(f"✅ Found {len(entries)} usable entries")

    spectrum_matrix = {}
    name_lookup = {}

    for entry in entries:
        acc = entry["spectrum_id"]
        name = entry["Compound_Name"]
        df = fetch_peaks(acc)
        if not df.empty:
            print(f"  {acc} ({name}): {len(df)} peaks")
            spectrum_matrix[acc] = df
            name_lookup[acc] = name
        else:
            print(f"  {acc}: No peak data")

    print(f"\n✅ Finished. Collected {len(spectrum_matrix)} usable spectra.")

    # Save accession IDs
    with open("gnps_ids_used.txt", "w") as f:
        for acc in spectrum_matrix:
            f.write(acc + "\n")

    # Save spectrum data and compound names
    with open("spectrum_matrix.pkl", "wb") as f:
        pickle.dump(spectrum_matrix, f)
    with open("spectrum_names.pkl", "wb") as f:
        pickle.dump(name_lookup, f)
    print("✅ Saved spectrum_matrix.pkl and spectrum_names.pkl")

if __name__ == "__main__":
    main()
