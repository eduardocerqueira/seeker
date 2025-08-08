#date: 2025-08-08T16:50:03Z
#url: https://api.github.com/gists/ab5059fab8c0a228821558d009e3c250
#owner: https://api.github.com/users/datavudeja

import os
import cv2
import shutil
import pandas as pd

from jinja2 import Environment, BaseLoader
from typing import List, TypedDict, Dict, Type, Tuple

from skimage import io
from pathlib import Path
from piffle.presentation import IIIFPresentation

SEGMONTO_VALUE_IDS = {
  'MainZone:commentary': 'BT01',
  'MainZone:primaryText': 'BT02',
  'MainZone:preface': 'BT03',
  'MainZone:translation': 'BT04',
  'MainZone:introduction': 'BT05',
  'NumberingZone:textNumber': 'BT06',
  'NumberingZone:pageNumber': 'BT07',
  'MainZone:appendix': 'BT08',
  'MarginTextZone:criticalApparatus': 'BT09',
  'MainZone:bibliography': 'BT10',
  'MarginTextZone:footnote': 'BT11',
  'MainZone:index': 'BT12',
  'RunningTitleZone': 'BT13',
  'MainZone:ToC': 'BT14',
  'TitlePageZone': 'BT15',
  'MarginTextZone:printedNote': 'BT16',
  'MarginTextZone:handwrittenNote': 'BT17',
  'CustomZone:other': 'BT18',
  'CustomZone:undefined': 'BT19',
  'CustomZone:line_region': 'BT20',
  'CustomZone:weird': 'BT21',
}

class PageDict(TypedDict):
    filename: str
    width: int
    height: int

def predictions_to_alto(
        page_dict: PageDict, 
        predictions: pd.DataFrame, 
        output_path: str, 
        segmonto_mappings: Dict[str, str] = SEGMONTO_VALUE_IDS
    ) -> None:
  """
  This function takes a list of YOLO predictions stored in a DataFrame (bounding boxes + region name)
  and serializes them according to the Alto/XML format. 
  """


  template_string = """<?xml version="1.0" encoding="UTF-8"?>
  <alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xmlns="http://www.loc.gov/standards/alto/ns-v4#"
        xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">
      <Description>
          <MeasurementUnit>pixel</MeasurementUnit>
          <sourceImageInformation>
              <fileName>{{page_dict['filename']}}</fileName>
          </sourceImageInformation>
      </Description>

      <Tags>
          {% for key in segmonto_value_ids.keys() %}
              <OtherTag ID="{{ segmonto_value_ids[key] }}" LABEL="{{ key }}" DESCRIPTION="block type {{ key }}"/>
          {% endfor %}
      </Tags>

      <Layout>
          <Page WIDTH="{{ page_dict['width'] }}"
                HEIGHT="{{ page_dict['height'] }}"
                ID="{{ page_dict['id'] }}"
                PHYSICAL_IMG_NR="">
              <PrintSpace HPOS="0" VPOS="0" WIDTH="{{ page_dict['width'] }}" HEIGHT="{{ page_dict['height'] }}">
                      {% for pred in predictions %}
                        {% set region_id = 'r_' ~ loop.index %}
                          <TextBlock ID="{{region_id}}"
                                    HPOS="{{ pred['hpos'] }}" VPOS="{{ pred['vpos'] }}"
                                    WIDTH="{{ pred['width'] }}" HEIGHT="{{ pred['height'] }}"
                                    TAGREFS="{{ segmonto_value_ids[pred['class']] }}">
                          </TextBlock>
                      {% endfor %}
              </PrintSpace>
          </Page>
      </Layout>
  </alto>
  """
  template = Environment(loader=BaseLoader).from_string(template_string)
  alto_xml_data = template.render(page_dict=page_dict, predictions=predictions, segmonto_value_ids=segmonto_mappings)
  output_path.write_text(alto_xml_data, encoding='utf-8')
  return

def pla_process_images(target_folder: str, yolo_model: object, save_predictions: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function runs a set of images stored in a folder through a YOLOv5 model. 
    It returns two DataFrames: one with information about the processed images and another one with the YOLO predictions.
    """

    # get the list of images to process
    input_files = list(Path(target_folder).glob('*.[jp][pn][g]'))
    print(len(input_files))

    image_dimensions = []
    for inpfile in input_files:
        img = cv2.imread(inpfile)
        img_height, img_width = img.shape[:2]
        image_dimensions.append({
            'id': inpfile, 
            'filename': os.path.basename(inpfile),
            'height': img_height, 
            'width': img_width
        })

    images_df = pd.DataFrame(image_dimensions).set_index('id')

    # run images through the model
    predictions = yolo_model(input_files)
    if save_predictions:
        predictions.save()

    temp = []
    for image, predictions_df in zip(input_files, predictions.pandas().xyxy):
        predictions_df['image_filename'] = os.path.basename(image)
        predictions_df['image_path'] = image
        temp.append(predictions_df)

    predictions_df = pd.concat(temp).reset_index()
    return images_df, predictions_df

def download_IA_book(book_id: str, target_folder: str, sample: int = None, start_at : int = None) -> None:

    book_target_folder = os.path.join(target_folder, book_id)
    Path(book_target_folder).mkdir(parents=True, exist_ok=True)
    
    iiif_manifest_link = f'https://iiif.archivelab.org/iiif/{book_id}/manifest.json'
    #print(iiif_manifest_link)

    manifest = IIIFPresentation.from_url(iiif_manifest_link)

    if sample and start_at:
        canvases = manifest.sequences[0].canvases[start_at:start_at + sample]
    elif sample and (start_at is None):
        canvases = manifest.sequences[0].canvases[:sample]
    else:
        canvases = manifest.sequences[0].canvases

    print(f"{len(canvases)} images will be downloaded...")
    
    for canvas in canvases:
        image_id = canvas.images[0].resource.id
        image_filename = f"{canvas.id.split('/')[-2]}.jpg"
        target_path = Path(book_target_folder) / image_filename
        img = io.imread(image_id)
        io.imsave(target_path, img)
        print(f"Image {image_id} was downloaded to {target_path}")
    print("Done.")

    shutil.make_archive(f'{book_target_folder}', 'zip', f'{book_target_folder}')
    print(f'A zip file containing {len(canvases)} image files was created at {book_target_folder}.zip')
    return

def alto_export(book_id: str, images_df: pd.DataFrame, predictions_df: pd.DataFrame) -> None:

    alto_basedir = Path(f'alto/{book_id}')
    alto_basedir.mkdir(parents=True, exist_ok=True)

    for idx, row in images_df.reset_index().iterrows():
        
        page_dict = row.to_dict()
        
        predictions = []
        for idx, row in predictions_df[predictions_df.image_filename == page_dict['filename']].iterrows():
            hpos = int(row['xmin'])
            vpos = int(row['ymin'])
            width = int(row['xmax']) - int(row['xmin'])
            height = int(row['ymax']) - int(row['ymin'])
            predictions.append(
                {
                    "class": row['name'],
                    "hpos": hpos,
                    "vpos": vpos,
                    "width": width,
                    "height": height
                }
            )
        predictions_to_alto(page_dict, predictions, alto_basedir / page_dict['filename'].replace('.jpg', '.xml'))

    print('The following region types were recognised:')
    print("\n".join(predictions_df.name.unique().tolist()))
    shutil.make_archive(f'alto/{book_id}', 'zip', f'alto/{book_id}/')
    print(f'A zip file containing {images_df.shape[0]} Alto/XML files was created at alto/{book_id}.zip')