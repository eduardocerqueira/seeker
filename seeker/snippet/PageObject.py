#date: 2023-02-10T16:49:34Z
#url: https://api.github.com/gists/d9f89985358b78216fdabb1a4197e858
#owner: https://api.github.com/users/tzkmx

class PageObject:
    def __init__(self, driver):
        self.driver = driver

    def takeScreenshot(self):
        # Código para tomar una captura de pantalla
        pass

    def checkTitle(self, expectedTitle):
        self.takeScreenshot()
        # Verificación previa del título
        assert self.driver.title == expectedTitle, f"Expected title: {expectedTitle}. Actual title: {self.driver.title}"
        self.takeScreenshot()

    def checkHeader(self, expectedHeader):
        self.takeScreenshot()
        # Verificación previa del encabezado
        header = self.driver.find_element_by_id("header")
        assert header.text == expectedHeader, f"Expected header: {expectedHeader}. Actual header: {header.text}"
        self.takeScreenshot()

    def checkFooter(self, expectedFooter):
        self.takeScreenshot()
        # Verificación previa del pie de página
        footer = self.driver.find_element_by_id("footer")
        assert footer.text == expectedFooter, f"Expected footer: {expectedFooter}. Actual footer: {footer.text}"
        self.takeScreenshot()
