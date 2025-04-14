#date: 2025-04-14T16:37:43Z
#url: https://api.github.com/gists/84c7dac5315521b1e29e53706c9f74f6
#owner: https://api.github.com/users/mdiqbalahmad

from threading import Thread
import time

class TrafficMagnet(burp.IProxyListener):
    def __init__(self):
        callbacks.registerProxyListener(self)
        self._helpers = callbacks.getHelpers()
        self._callbacks = callbacks
        

    def listen(self):
        while True:
            time.sleep(1)
            if (not handler.running):
                callbacks.removeProxyListener(self)
                return

    def _issueAndShow(self, httpService, request):
        def AddToTable(req, rsp, lbl=""):
            x = burp.Request(req,[],False,lbl)
            x.response = rsp
            table.add(x)
        RequestResponse = self._callbacks.makeHttpRequest(httpService, request)
        response = RequestResponse.getResponse()
        AddToTable(self._helpers.bytesToString(request), self._helpers.bytesToString(response), lbl="Test Label")

    def issueRequest(self, httpService, headers, body):
        newRequest = self._helpers.buildHttpMessage(headers, self._helpers.stringToBytes(body))
        thread = Thread(target=self._issueAndShow, args=(httpService, newRequest,))
        thread.start()
   
    def processProxyMessage(self, messageIsRequest, message):
        if messageIsRequest:
            messageInfo = message.getMessageInfo()
            httpService = messageInfo.getHttpService()
            requestBytes = messageInfo.getRequest()
            requestInfo = self._helpers.analyzeRequest(requestBytes)
            headers = requestInfo.getHeaders()
            bodyBytes = requestBytes[requestInfo.getBodyOffset():]
            bodyStr = self._helpers.bytesToString(bodyBytes)

            ############
            #   Match  #
            ############

            # add your matching logic here to stay in-scope
            
            # host = str(httpService.getHost()).lower()
            # if "target.com" not in host:
            #     return

            ############
            #  Attack  #
            ############

            # Get the original method
            originalMethod = requestInfo.getMethod()

            # List of HTTP methods
            httpMethods = ["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "TRACE", "CONNECT"]

            for method in httpMethods:
                if method != originalMethod:
                    # Create a new header with the new method
                    newHeader = headers[0].replace(originalMethod, method)
                    # Replace the old header with the new one
                    newHeaders = [newHeader] + headers[1:]
                    # Issue the request through Burp (Not TI)
                    self.issueRequest(httpService, newHeaders, bodyStr)


def queueRequests(target, wordlists):
    # We don't use the TI engine, but we need to construct it for this to work
    engine = RequestEngine(endpoint=target.endpoint)
    # Keep this running until user cancels the attack
    TrafficMagnet().listen()

        
def handleResponse(req, interesting):
    table.add(req)