#date: 2024-05-21T17:01:26Z
#url: https://api.github.com/gists/48a9a1bd6ad4c55a29d76aa79dd371a1
#owner: https://api.github.com/users/JeremyFyke

class filepaths():
 
    adj_data_dir=  '[LOCAL_CRIM_PATH]/qdm-adjusted-data/'
    climo_data_dir='[LOCAL_CRIM_PATH]/summary-stats/production_data/'
    mask_dir='[LOCAL_CRIM_PATH]/summary-stats/'
    maindir='[LOCAL_CRIM_PATH]/supporting-data/'
    preprocessed_quick_start_tab_dir='[LOCAL_CRIM_PATH]/summary-stats/production_data/Quick_Start_precalculated_data/output/'

    #Set append code appropriately for netCDF read.  If:
    # -reading locally from disk, leave blank.
    # -reading from over network (e.g. via URL from FTP) add append code.

    append_code=''
    #append_code='#mode=bytes'