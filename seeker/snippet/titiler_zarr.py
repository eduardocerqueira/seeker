#date: 2022-11-28T17:00:27Z
#url: https://api.github.com/gists/7085801ecf403a45421802d8642aa0d3
#owner: https://api.github.com/users/vincentsarago

from urllib.parse import urlencode
from enum import Enum
from typing import List, Optional, Tuple

from fastapi import FastAPI, Query, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field, root_validator
from starlette.requests import Request
from starlette.responses import HTMLResponse

from titiler.core.models.mapbox import TileJSON
from titiler.core.factory import templates
from titiler.core.resources.enums import ImageType
from titiler.core.dependencies import ImageRenderingParams, ColorMapParams, RescalingParams

import xarray
from rio_tiler.io.xarray import XarrayReader


app = FastAPI()


@app.get("/tiles/{z}/{x}/{y}", response_class=Response)
def tile(
    z: int,
    x: int,
    y: int,
    url: str = Query(..., description="Zarr URL"),
    variable: str = Query(..., description="Zarr Variable"),
    rescale: Optional[List[Tuple[float, ...]]] = Depends(
        RescalingParams
    ),  # noqa
    color_formula: Optional[str] = Query(  # noqa
        None,
        title="Color Formula",
        description="rio-color formula (info: https://github.com/mapbox/rio-color)",
    ),
    colormap=Depends(ColorMapParams),  # noqa
    render_params=Depends(ImageRenderingParams),  # noqa
):
    with xarray.open_dataset(url, engine="zarr", decode_coords="all") as src:
        ds = src[variable][:1]

        # Make sure we are a CRS
        crs = ds.rio.crs or "epsg:4326"
        ds.rio.write_crs(crs, inplace=True)

        with XarrayReader(ds) as dst:
            image = dst.tile(x, y, z)


    format = ImageType.jpeg if image.mask.all() else ImageType.png

    if rescale:
        image.rescale(rescale)

    if color_formula:
        image.apply_color_formula(color_formula)

    content = image.render(
        img_format=format.driver,
        colormap=colormap,
        **format.profile,
        **render_params,
    )

    return Response(content, media_type=format.mediatype)

@app.get(
    "/tilejson.json",
    response_model=TileJSON,
    responses={200: {"description": "Return a tilejson"}},
    response_model_exclude_none=True,
)
def tilejson(
    request: Request,
    url: str = Query(description="Zarr URL"),
    variable: str = Query(description="Zarr Variable"),
    rescale: Optional[List[Tuple[float, ...]]] = Depends(
        RescalingParams
    ),  # noqa
    color_formula: Optional[str] = Query(  # noqa
        None,
        title="Color Formula",
        description="rio-color formula (info: https://github.com/mapbox/rio-color)",
    ),
    colormap=Depends(ColorMapParams),  # noqa
    render_params=Depends(ImageRenderingParams),  # noqa
):
    """Handle /tilejson.json requests."""
    kwargs: Dict[str, Any] = {"z": "{z}", "x": "{x}", "y": "{y}"}
    tile_url = request.url_for("tile", **kwargs)
    if request.query_params._list:
        tile_url += f"?{urlencode(request.query_params._list)}"

    with xarray.open_dataset(url, engine="zarr", decode_coords="all") as src:
        ds = src[variable][:1]
        crs = ds.rio.crs or "epsg:4326"
        ds.rio.write_crs(crs, inplace=True)
        with XarrayReader(ds) as dst:
            return dict(
                bounds=dst.geographic_bounds,
                minzoom=dst.minzoom,
                maxzoom=dst.maxzoom,
                name="xarray",
                tilejson="2.1.0",
                tiles=[tile_url],
            )


@app.get("/map", response_class=HTMLResponse)
def map_viewer(
    request: Request,
    url: str = Query(description="Zarr URL"),
    variable: str = Query(description="Zarr Variable"),
    rescale: Optional[List[Tuple[float, ...]]] = Depends(
        RescalingParams
    ),  # noqa
    color_formula: Optional[str] = Query(  # noqa
        None,
        title="Color Formula",
        description="rio-color formula (info: https://github.com/mapbox/rio-color)",
    ),
    colormap=Depends(ColorMapParams),  # noqa
    render_params=Depends(ImageRenderingParams),  # noqa
):
    """Return TileJSON document for a dataset."""
    tilejson_url = request.url_for("tilejson")
    if request.query_params._list:
        tilejson_url += f"?{urlencode(request.query_params._list)}"

    return templates.TemplateResponse(
        name="index.html",
        context={
            "request": request,
            "tilejson_endpoint": tilejson_url,
        },
        media_type="text/html",
    )
