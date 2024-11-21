#date: 2024-11-21T17:13:24Z
#url: https://api.github.com/gists/eabd8e45845d024af397944fce9d1d67
#owner: https://api.github.com/users/sts-developer

import http.client
import json

conn = http.client.HTTPSConnection("testapi.taxbandits.com")
payload = json.dumps({
  "SubmissionManifest": {
    "SubmissionId": "b5bdb3ba-aed8-41e2-8f18-ed3c761d26bf",
    "TaxYear": "2024",
    "IRSFilingType": "IRIS",
    "IsFederalFiling": True,
    "IsPostal": True,
    "IsOnlineAccess": True,
    "IsScheduleFiling": False,
    "ScheduleFiling": None
  },
  "ReturnHeader": {
    "Business": {
      "BusinessId": None,
      "BusinessNm": "Snowdaze LLC",
      "FirstNm": None,
      "MiddleNm": None,
      "LastNm": None,
      "Suffix": None,
      "PayerRef": "Snow123",
      "TradeNm": "Iceberg Icecreams",
      "IsEIN": True,
      "EINorSSN": "71-3787159",
      "Email": "james@sample.com",
      "ContactNm": None,
      "Phone": "6634567890",
      "PhoneExtn": "12345",
      "Fax": "6634567890",
      "BusinessType": "ESTE",
      "SigningAuthority": None,
      "KindOfEmployer": "FederalGovt",
      "KindOfPayer": "REGULAR941",
      "IsBusinessTerminated": True,
      "IsForeign": False,
      "USAddress": {
        "Address1": "3576 AIRPORT WAY",
        "Address2": "UNIT 9",
        "City": "FAIRBANKS",
        "State": "AK",
        "ZipCd": "99709"
      },
      "ForeignAddress": None
    }
  },
  "ReturnData": [
    {
      "RecordId": "f22c477a-9680-456d-b7a5-a39342e4362a",
      "SequenceId": "1",
      "IsPostal": True,
      "IsOnlineAccess": True,
      "IsForced": True,
      "Recipient": {
        "RecipientId": None,
        "TINType": "EIN",
        "TIN": "36-3814577",
        "FirstPayeeNm": "Dairy Delights LLC",
        "SecondPayeeNm": "Coco Milk",
        "FirstNm": None,
        "MiddleNm": None,
        "LastNm": None,
        "Suffix": None,
        "IsForeign": True,
        "USAddress": None,
        "ForeignAddress": {
          "Address1": "120 Bremner Blvd",
          "Address2": "Suite 800",
          "City": "Toronto",
          "ProvinceOrStateNm": "Ontario",
          "Country": "CA",
          "PostalCd": "4168682600"
        },
        "Email": "shawn08@sample.com",
        "Fax": "6834567890",
        "Phone": "7634567890"
      },
      "BFormData": {
        "B1aDescrOfProp": "RFC",
        "B1bDateAcquired": "07/01/2022",
        "B1cDateSoldOrDisposed": "09/04/2021",
        "B1dProceeds": 40.55,
        "B1eCostOrOtherBasis": 30.89,
        "B1fAccruedMktDisc": 20.11,
        "B1gWashsaleLossDisallowed": 4.25,
        "B2TypeOfGainLoss": "ordinary short term",
        "B3IsProceedsFromCollectibles": True,
        "B3IsProceedsFromQOF": False,
        "B4FedTaxWH": 0,
        "B5IsNonCoveredSecurityNotReported": False,
        "B5IsNonCoveredSecurityReported": False,
        "B6IsGrossProceeds": True,
        "B6IsNetProceeds": False,
        "B7IsLossNotAllowedbasedOn1d": False,
        "B8PLRealizedOnClosedContract": 0,
        "B9PLUnrealizedOnOpenContractPrevTy": 0,
        "B10UnrealizedPLOnOpenContractCurTy": 0,
        "B11AggPLOnContract": 0,
        "B12IsBasisReportedToIRS": False,
        "B13Bartering": 43,
        "AccountNum": "789121",
        "CUSIPNum": "8988932143534",
        "IsFATCA": True,
        "Form8949Code": "X",
        "Is2ndTINnot": True,
        "States": [
          {
            "StateCd": "WV",
            "StateIdNum": "99999999",
            "StateWH": 257.94
          },
          {
            "StateCd": "ID",
            "StateIdNum": "999999999",
            "StateWH": 15
          }
        ]
      }
    }
  ]
})
headers = {
  'Authorization': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlNDdhN2I3MGMwNTY0NjI2OTU0M2RhNzQwZmNiNTZmNCIsImV4cCI6MTczMjAxNjY0OCwiaWF0IjoxNzMyMDEzMDQ4LCJpc3MiOiJodHRwczovL3Ricy1vYXV0aC5zdHNzcHJpbnQuY29tL3YyLyIsInN1YiI6ImJlZWQ0ZTAxYzM2NmQ4MjIiLCJ1c2VydW5pcXVlaWQiOiIifQ.S6hAjN56eo7fQuFL810aody1M-22xvhWj8ewVAHRxDU',
  'Content-Type': 'application/json'
}
conn.request("PUT", "/V1.7.3/form1099B/Update", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))