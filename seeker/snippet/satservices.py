#date: 2023-03-29T17:39:56Z
#url: https://api.github.com/gists/f1e3994e6cb946858af4ae2c9de1ecd6
#owner: https://api.github.com/users/jroqu3

import time
from zeep import xsd
from zeep import Client


class SATServices:
    def __init__(self, taxpayer_id, rtaxpayer_id, uuid, total):
        self.time_start = time.time()

        self.taxpayer_id = taxpayer_id
        self.rtaxpayer_id = rtaxpayer_id
        self.uuid = uuid.upper()
        self.total = total

        self.url = "https://consultaqr.facturaelectronica.sat.gob.mx/ConsultaCFDIService.svc?WSDL"

    def get_expresion_impresa(self):
        # expresion_impresa = "?re={taxpayer_id}&rr={rtaxpayer_id}&tt={total}&id={uuid}&fe="
        expresion_impresa = "<![CDATA[?re={taxpayer_id}&rr={rtaxpayer_id}&tt={total}&id={uuid}&fe=]]>"
        expresion_impresa = expresion_impresa.format(
            taxpayer_id=self.taxpayer_id,
            rtaxpayer_id=self.rtaxpayer_id,
            uuid=self.uuid, total=self.total
        )
        return expresion_impresa

    def get_response(self):
        client = Client(self.url)
        expresion_impresa_object = xsd.AnyObject(xsd.String(), self.get_expresion_impresa())
        response = client.service.Consulta(expresion_impresa_object)
        return response

    def get_response_result(self, verbose=False):
        response = self.get_response()
        if verbose:
            self.time_end = time.time()
            self.total_time = self.time_end - self.time_start
            print("[Time: {}] {} ({}) >> CodigoEstatus: {} | EsCancelable: {}".format(self.total_time, self.uuid, response.Estado, response.CodigoEstatus, response.EsCancelable))
        return response


obj = SATServices(
    taxpayer_id="CAU180123GEA",
    rtaxpayer_id="GACM650215GH9",
    uuid="699628be-2f06-4bc9-adc2-860b51eecbca",
    total="398.39",
)
response = obj.get_response_result(verbose=True)
