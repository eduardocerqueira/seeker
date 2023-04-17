#date: 2023-04-17T17:09:53Z
#url: https://api.github.com/gists/b519fa731a9123dbca2406770796f050
#owner: https://api.github.com/users/maawoo

import pystac
from pathlib import Path
import argparse


def create_stac_hierarchy(root_dir, dry_run=False, verbose=False):
    """
    Create STAC hierarchy for already existing STAC Items.

    Parameters
    ----------
    root_dir: Path
        Path to the root directory of the STAC hierarchy.
    dry_run: bool
        If True, don't save the STAC hierarchy to disk.
    verbose: bool
        If True, print the number of STAC Items per Collection.

    Returns
    -------
    catalog: pystac.Catalog
        The STAC Catalog.
    """
    catalog = pystac.Catalog(id=f'{root_dir.stem}_catalog',
                             description=f'STAC Catalog for {root_dir.stem} products',
                             catalog_type=pystac.CatalogType.SELF_CONTAINED,
                             href=root_dir.joinpath('catalog.json'))
    sp_extent = pystac.SpatialExtent([None, None, None, None])
    tmp_extent = pystac.TemporalExtent([None, None])
    
    for sub in root_dir.iterdir():
        if sub.is_dir() and len(sub.stem) == 7:
            tile = sub.stem
            stac_item_paths = list(sub.glob("**/odc-metadata.stac-item.json"))
            
            collection = pystac.Collection(id=tile,
                                           description=f'STAC Collection for {root_dir.stem} products of tile {tile}.',
                                           extent=pystac.Extent(sp_extent, tmp_extent),
                                           href=sub.joinpath('collection.json'))
            catalog.add_child(collection)
            
            items = []
            for item_p in stac_item_paths:
                if tile in str(item_p.parent):
                    item = pystac.Item.from_file(href=str(item_p))
                    items.append(item)
                    collection.add_item(item=item)
                    
                    item.set_self_href(str(item_p))
                    for asset_key, asset in item.assets.items():
                        asset.href = Path(asset.href).stem
                else:
                    continue
            
            extent = collection.extent.from_items(items=items)
            collection.extent = extent
            
            if verbose:
                print(f"{tile} - {len(stac_item_paths)}")
    
    if not dry_run:
        catalog.save()
    
    return catalog


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Create STAC hierarchy for already existing STAC Items.')
    parser.add_argument('--root_dir', type=str, help='Path to the root directory of the STAC hierarchy.')
    parser.add_argument('--dry_run', action='store_true', help='If True, will not save the STAC hierarchy to disk.')
    parser.add_argument('--verbose', action='store_true', help='If True, print the number of STAC Items per Collection.')
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    dry_run = args.dry_run
    verbose = args.verbose
    
    create_stac_hierarchy(root_dir=root_dir, dry_run=dry_run, verbose=verbose)
