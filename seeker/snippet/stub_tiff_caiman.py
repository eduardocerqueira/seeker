#date: 2025-06-18T17:02:24Z
#url: https://api.github.com/gists/c1395a9296918a3a1c16bc01e985f0a9
#owner: https://api.github.com/users/h-mayorquin

"""
TIFF file stubber - Creates a stub of a TIFF file with only the first n pages.
Preserves all metadata, tags, and structure of the original pages.
"""

from pathlib import Path
import tifffile
import numpy as np


def stub_tiff_file(input_path, output_path, n_pages=10):
    """
    Create a stub TIFF file containing only the first n pages.
    
    Args:
        input_path: Path to the input TIFF file
        output_path: Path to the output stub TIFF file
        n_pages: Number of pages to include in the stub (default: 10)
    """
    print(f"Reading TIFF file: {input_path}")
    
    with tifffile.TiffFile(input_path) as tif:
        total_pages = len(tif.pages)
        pages_to_copy = min(n_pages, total_pages)
        
        print(f"Total pages in file: {total_pages}")
        print(f"Creating stub with first {pages_to_copy} pages")
        
        # Open output file for writing
        with tifffile.TiffWriter(output_path, bigtiff=tif.is_bigtiff) as writer:
            # Copy each page with all its metadata
            for i in range(pages_to_copy):
                page = tif.pages[i]
                
                # Read the image data
                image_data = page.asarray()
                
                # Get all tags from the page
                # We'll build a dictionary of tag arguments to pass to write()
                tag_dict = {}
                
                # Copy important tags if they exist
                if hasattr(page, 'photometric'):
                    tag_dict['photometric'] = page.photometric
                if hasattr(page, 'planarconfig'):
                    tag_dict['planarconfig'] = page.planarconfig
                if hasattr(page, 'compression'):
                    tag_dict['compression'] = page.compression
                if hasattr(page, 'predictor'):
                    tag_dict['predictor'] = page.predictor
                if hasattr(page, 'subsampling'):
                    tag_dict['subsampling'] = page.subsampling
                if hasattr(page, 'tile'):
                    tag_dict['tile'] = page.tile
                
                # Copy resolution tags
                if hasattr(page, 'resolution'):
                    tag_dict['resolution'] = page.resolution
                if hasattr(page, 'resolution_unit'):
                    tag_dict['resolution_unit'] = page.resolution_unit
                
                # Copy description and software tags
                if hasattr(page, 'description'):
                    tag_dict['description'] = page.description
                if hasattr(page, 'software'):
                    tag_dict['software'] = page.software
                if hasattr(page, 'datetime'):
                    tag_dict['datetime'] = page.datetime
                
                # Copy ImageJ metadata if present
                if hasattr(page, 'imagej_metadata'):
                    tag_dict['imagej_metadata'] = page.imagej_metadata
                
                # Handle extra tags that might not be standard
                if hasattr(page, 'tags'):
                    # Create extratags list for non-standard tags
                    extratags = []
                    
                    # Tags to skip (already handled above or will cause issues)
                    skip_tags = {
                        'ImageWidth', 'ImageLength', 'BitsPerSample',
                        'SamplesPerPixel', 'PhotometricInterpretation',
                        'PlanarConfiguration', 'Compression', 'Predictor',
                        'TileWidth', 'TileLength', 'RowsPerStrip',
                        'StripOffsets', 'StripByteCounts', 'TileOffsets',
                        'TileByteCounts', 'XResolution', 'YResolution',
                        'ResolutionUnit', 'Software', 'DateTime',
                        'ImageDescription', 'SampleFormat', 'YCbCrSubSampling'
                    }
                    
                    for tag in page.tags.values():
                        if tag.name not in skip_tags:
                            try:
                                # Create tuple of (tag_code, dtype, count, value, writeonce)
                                extratags.append((
                                    tag.code,
                                    tag.dtype.name,
                                    tag.count,
                                    tag.value,
                                    True
                                ))
                            except Exception as e:
                                print(f"  Warning: Could not copy tag {tag.name}: {e}")
                    
                    if extratags:
                        tag_dict['extratags'] = extratags
                
                # Write the page with all its metadata
                print(f"  Copying page {i + 1}/{pages_to_copy}")
                writer.write(image_data, **tag_dict)
        
        print(f"\nStub created successfully: {output_path}")
        print(f"Original file pages: {total_pages}")
        print(f"Stub file pages: {pages_to_copy}")
        
        # Verify the stub
        with tifffile.TiffFile(output_path) as stub:
            print(f"Verification - Stub contains {len(stub.pages)} pages")


if __name__ == "__main__":
    folder_path = Path("/home/heberto/data/segmentation_example/mini_sample")
    file_path = folder_path / "mini_1000.tiff"
    stub_path = file_path.with_name(file_path.stem + "_stub" + file_path.suffix)
    stub_tiff_file(file_path, stub_path, n_pages=5)