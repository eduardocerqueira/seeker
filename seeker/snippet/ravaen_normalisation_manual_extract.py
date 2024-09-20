#date: 2024-09-20T17:11:24Z
#url: https://api.github.com/gists/c312c2e39bcceb88d41f88ae3dd2cb2a
#owner: https://api.github.com/users/previtus


class DataNormalizerLogManual():
    def __init__(self):
        self.setup()
    def setup(self):
        # These were edited to work with the 10 bands we had in Wildfires project (FireCLR)
        # only use 10m resolution bands (10): Blue (B2), Green (B3), Red (B4), VNIR (B5),
        # VNIR (B6), VNIR (B7), NIR (B8), VNIR (B8a), SWIR (B11), SWIR (B12) combining
        self.BANDS_S2_BRIEF = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

        self.RESCALE_PARAMS = {
            "B1": {"x0": 7.3,
                   "x1": 7.6,
                   "y0": -1,
                   "y1": 1,
                   },
            "B2": {"x0": 6.9,
                   "x1": 7.5,
                   "y0": -1,
                   "y1": 1,
                   },
            "B3": {"x0": 6.5,
                   "x1": 7.4,
                   "y0": -1,
                   "y1": 1,
                   },
            "B4": {"x0": 6.2,
                   "x1": 7.5,
                   "y0": -1,
                   "y1": 1,
                   },
            "B5": {"x0": 6.1,
                   "x1": 7.5,
                   "y0": -1,
                   "y1": 1,
                   },
            "B6": {"x0": 6.5,
                   "x1": 8,
                   "y0": -1,
                   "y1": 1,
                   },
            "B7": {"x0": 6.5,
                   "x1": 8,
                   "y0": -1,
                   "y1": 1,
                   },
            "B8": {"x0": 6.5,
                   "x1": 8,
                   "y0": -1,
                   "y1": 1,
                   },
            "B8A": {"x0": 6.5,
                    "x1": 8,
                    "y0": -1,
                    "y1": 1,
                    },
            "B9": {"x0": 6,
                   "x1": 7,
                   "y0": -1,
                   "y1": 1,
                   },
            "B10": {"x0": 2.5,
                    "x1": 4.5,
                    "y0": -1,
                    "y1": 1,
                    },
            "B11": {"x0": 6,
                    "x1": 8,
                    "y0": -1,
                    "y1": 1,
                    },
            "B12": {"x0": 6,
                    "x1": 8,
                    "y0": -1,
                    "y1": 1,
                    }
        }
        print("normalization params are manually found")

    def normalize_x(self, data):
        bands = data.shape[0]  # for example 15
        for band_i in range(bands):
            data_one_band = data[band_i, :, :]
            if band_i < len(self.BANDS_S2_BRIEF):
                # log
                data_one_band = np.log(data_one_band)
                data_one_band[np.isinf(data_one_band)] = np.nan

                # rescale
                r = self.RESCALE_PARAMS[self.BANDS_S2_BRIEF[band_i]]
                x0, x1, y0, y1 = r["x0"], r["x1"], r["y0"], r["y1"]
                data_one_band = ((data_one_band - x0) / (x1 - x0)) * (y1 - y0) + y0
            data[band_i, :, :] = data_one_band
        return data

    def denormalize_x(self, data):
        bands = data.shape[0]  # for example 15
        for band_i in range(bands):
            data_one_band = data[band_i, :, :]
            if band_i < len(self.BANDS_S2_BRIEF):
                # rescale
                r = self.RESCALE_PARAMS[self.BANDS_S2_BRIEF[band_i]]
                x0, x1, y0, y1 = r["x0"], r["x1"], r["y0"], r["y1"]
                data_one_band = (((data_one_band - y0) / (y1 - y0)) * (x1 - x0)) + x0

                # undo log
                data_one_band = np.exp(data_one_band)
                # data_one_band = np.log(data_one_band)
                # data_one_band[np.isinf(data_one_band)] = np.nan

            data[band_i, :, :] = data_one_band
        return data

      

normaliser = DataNormalizerLogManual()
# pseudocode of usage:
# imagin you have loaded data here
before = read_image(before_path, channels) # rasterio load for example
before_tiles = image2tiles(before) # tiling script
after = read_image(after_path, channels)
after_tiles = image2tiles(after)

for tile_i in range(len(before_tiles)):
    before_tiles[tile_i] = normaliser.normalize_x(before_tiles[tile_i])
for tile_i in range(len(after_tiles)):
    after_tiles[tile_i] = normaliser.normalize_x(after_tiles[tile_i])

# ... etc
# for example check the stats of your normalised data - is it between the expected -1 to +1 ?