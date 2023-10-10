#date: 2023-10-10T16:59:32Z
#url: https://api.github.com/gists/aea1d7cb060ac57e806e25a19602f9c3
#owner: https://api.github.com/users/tcotten-scrypted

    def _load_image(self, image: Image, draw_method: callable) -> None:
        """Helper method to load an image and draw it using the specified draw method."""

        # Determine if the image has an alpha channel once outside the loops
        has_alpha = image.mode == "RGBA"

        # Iterate through each pixel of the image
        for y in range(image.height):
            for x in range(image.width):
                pixel = image.getpixel((x, y))

                if has_alpha:
                    r, g, b, a = pixel
                else:
                    r, g, b = pixel
                    a = 255  # Default to fully opaque

                color = ColorRGBA(r, g, b, a)

                # Use the specified draw method for each point
                draw_method(Point(x, y), color)

    def load_square_image(self, image: Image) -> None:
        # Here, you'd use your conversion functions to convert the image to a dimetric view
        # If no conversion, you'd use this image as the base for square drawing
        self._load_image(image, self.draw_on_square)

    def load_dimetric_image(self, image: Image) -> None:
        # Here, you'd use your conversion functions to convert the image to a square view
        # If no conversion, you'd use this image as the base for dimetric drawing
        self._load_image(image, self.draw_on_dimetric)