#date: 2024-02-01T16:53:32Z
#url: https://api.github.com/gists/a62e4300475a9fbabd3c1fe1788fab7c
#owner: https://api.github.com/users/retsyn


def identify_sequences(directory_content: list) -> list:
    """Takes a list of files and returns "frame sequence names", in the form of name.###.ext.

    Args:
        directory_content (list): List of strings being files found in a folder.

    Returns:
        list: unique sequence names, with frames number as a wildcard.
    """    
    
    found_files = {}

    for frame_file in directory_content:

        file_string = frame_file.split('/')[-1]  # Get the local filename
        seq_name, frame_num, extension = file_string.split('.')

        # Based on the filename, store the frames in dict, sorting for repeats.
        if(seq_name in found_files):
            # Filename already found, just record the frame
            found_files[seq_name]['frames'].append(int(frame_num))        
        else:
            found_files[seq_name] = {}
            found_files[seq_name]['frames'] = [int(frame_num)]
            found_files[seq_name]['ext'] = extension

    sequences = []
    for sequence in found_files:
        sequence_str = f"{sequence}.####.{found_files[sequence]['ext']}"
        sequences += sequence_str

    return sequences