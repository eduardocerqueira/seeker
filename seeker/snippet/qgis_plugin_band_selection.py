#date: 2025-08-14T16:47:21Z
#url: https://api.github.com/gists/7cf95892fa4cd8e4f57ba687b90aabdb
#owner: https://api.github.com/users/zacdezgeo

def update_band_dropdowns(self):
    band_selection_items = (
        get_available_bands(self.image_collection_id.text().strip(), silent=True)
        or []
    )
    for i in range(3):
        band_dropdown = self.findChild(QComboBox, f"viz_band_{i}")
        current = band_dropdown.currentText()
        band_dropdown.clear()
        band_dropdown.addItems(band_selection_items)
        band_dropdown.setCurrentText(current)