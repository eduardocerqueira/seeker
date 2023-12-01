#date: 2023-12-01T16:59:20Z
#url: https://api.github.com/gists/1abca0baf0e8bb1d25a2e650c0a539cd
#owner: https://api.github.com/users/pmasiphelps

import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
from presidio_analyzer import AnalyzerEngine

# Read recipe inputs
CALL_CENTER_TRANSCRIPTS = dataiku.Dataset("CALL_CENTER_TRANSCRIPTS")
CALL_CENTER_TRANSCRIPTS_df = CALL_CENTER_TRANSCRIPTS.get_dataframe()

analyzer = AnalyzerEngine()

def anonymize_text(input_text):
    # Anonymize the following entities detected in a text column
    analyzer_results = analyzer.analyze(text=input_text,
                               entities=["PERSON",
                                         "PHONE_NUMBER",
                                         "EMAIL_ADDRESS",
                                         "US_BANK_NUMBER",
                                         "US_SSN",
                                         "IBAN_CODE", 
                                         "CREDIT_CARD"],
                               language='en')
    engine = AnonymizerEngine()
    
    result = engine.anonymize(
        text=input_text,
        analyzer_results=analyzer_results,
        operators={"PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                   "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
                   "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_ADDRESS>"}),
                   "US_BANK_NUMBER": OperatorConfig("replace", {"new_value": "<US_BANK_NUMBER>"}),
                   "US_SSN": OperatorConfig("replace", {"new_value": "<US_SSN>"}),
                   "IBAN_CODE": OperatorConfig("replace", {"new_value": "<IBAN_CODE>"}), 
                   "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"})}
    )
    
    return result.text

CALL_CENTER_TRANSCRIPTS_df["call_transcript_anon"] = CALL_CENTER_TRANSCRIPTS_df["call_transcript"].apply(lambda x: anonymize_text(x))

# Drop the non-anonymized call transcript
CALL_CENTER_TRANSCRIPTS_df.drop(columns = ["call_transcript"], inplace = True)

# Write recipe outputs
CALL_CENTER_TRANSCRIPTS_ANON = dataiku.Dataset("CALL_CENTER_TRANSCRIPTS_ANON")
CALL_CENTER_TRANSCRIPTS_ANON.write_with_schema(CALL_CENTER_TRANSCRIPTS_df)