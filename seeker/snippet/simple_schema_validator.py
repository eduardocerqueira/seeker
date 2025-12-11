#date: 2025-12-11T17:16:56Z
#url: https://api.github.com/gists/1bef3c9d6fd4d6d27fd52a00d0452008
#owner: https://api.github.com/users/masterPiece93

"""
Simple Schema Validator
=======================

efficient for basic straightforward usecases

This script shows an exerpt from a part of
process , where a incoming pub/sub message
is validated against a predefined schema.

* Class:DictValidator - is developed as a reusable
  validator meta class for validating a python dict 
  against a schema .
  The schema can be defined declaratively in a sub class 
  that uses DictValidator as a metaclass .
  
* Class:IngestionMessage - is the class that defines
  the schema for a pub/sub message dict . It uses
  `class:DictValidator` as a metaclass for being able
  to define a schema .

"""
from typing import ClassVar, Optional, Callable
from abc import abstractmethod
import logging
log=logging.getLogger(__name__)

def msg(message: str, pubsub_message_id: str = None):
    """Message Formatter"""
    if pubsub_message_id:
        return f" [{pubsub_message_id[:6]}...{pubsub_message_id[-3:]}] : " + message
    return message

class ReadOnlyMeta:
    """
    Abstract Base for ReadOnly Classes

    Prohibits the modification of class
        varibales .
    
    Usage:
        class Xyz(metaclass=ReadOnlyMeta):
            ...
    """
    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise AttributeError(f"Cannot modify constant '{name}' on ReadOnly Class {cls.__qualname__}")
        super().__setattr__(cls, name, value)


class ValidtaorMeta(type):
    """
    Abstract Base for Validator Classes

    * class must have `validate` method
    * `validate` method must be an instance method
    * class variables are READ ONLY

    Usage:
        class Xyz(metaclass=ValidtaorMeta):
            ...
    """
    def __new__(cls, name, bases, dct):
        # Enforce that the 'validate' method is in the class dictionary
        method_name='validate'
        if 'validate' not in dct or isinstance(dct["validate"], classmethod) or isinstance(dct["validate"], staticmethod):
            raise TypeError(
                f"Can't instantiate abstract class {name}"
                f"without an implementation for abstract class method {method_name}"
            )
        
        # Call the superclass's (ABCMeta's) __new__ method to create the class
        return super().__new__(cls, name, bases, dct)
    
    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise AttributeError(f"Cannot modify constant '{name}' on ReadOnly Class {cls.__qualname__}")
        super().__setattr__(cls, name, value)


class DictValidator(metaclass=ValidtaorMeta):
    """
    Validates the provided dict paylod against
        the provided validation specification
    
    - applies any additional formatting if specified
    """
    class SchemaViolation(Exception):
        """Indicates the violation of validation specification"""

    SCHEMEA_VIOLATION_EXEPTION: ClassVar[Exception] = SchemaViolation
    ALLOWED_EXTRA_KEYS : ClassVar[bool] = False
    FORMATTERS: ClassVar[dict] = {}

    @abstractmethod
    def validate(self, json_payload: dict, logger: Optional[Callable] = None, message_wrapper: Optional[Callable] = None) -> Optional[Exception]:
        """
        Default validation logic
        """

        def do_logging(msg: str, level: str = 'info'):
            msg = message_wrapper(msg)
            if logger:
                logger(msg, level=level)
            else:
                print(f'{level.upper()} : ', msg)
        
        # schema validation and formatting
        for key, spec in self.VALIDATION_SPECIFICATION.items():
            is_required, expected_type, default_value = spec
            
            if key not in json_payload:
                if is_required:
                    log_msg = f'{self.SCHEMEA_VIOLATION_EXEPTION.__name__}:`{key}` Key Required'
                    do_logging(log_msg, 'error')
                    raise self.SCHEMEA_VIOLATION_EXEPTION(log_msg)
                else:
                    json_payload[key]=default_value

            value = json_payload[key]

            if not isinstance(value, expected_type):
                log_msg = f'`{key=}` is expected of type {expected_type}, got {type(value)}'
                do_logging(log_msg, 'error')
                raise self.SCHEMEA_VIOLATION_EXEPTION(log_msg)
            
            if not self.ALLOWED_EXTRA_KEYS:
                extra_keys = json_payload.keys() - self.VALIDATION_SPECIFICATION.keys()
                if extra_keys:
                    log_msg = f'Extra Key : {extra_keys} Not Allowed'
                    do_logging(log_msg, 'error')
                    raise self.SCHEMEA_VIOLATION_EXEPTION(log_msg)

            json_payload[key] = self.FORMATTERS.get(key, lambda v:v)(value)

# main entrypoint
if __name__ == '__main__':
    class IngestionMessage(DictValidator):
        """A Simple Dict Validator for 
        Ingestion Message Json Payload
        """
        VALIDATION_SPECIFICATION: ClassVar[dict] = {    # Required
            # KEY       ( Req, type, default )
            "eventId":  (True, str, None),
            "username": (True, str, None),
            "url":      (True, str, None),
            "orgId":    (True, str, None),
            "tenancy":  (True, str, None),
            "orgName":  (True, str, None),
            "channel":  (True, str, None),
            "extra":    (True, str, None),
        }
        ALLOWED_EXTRA_KEYS: ClassVar[bool] = False      # Optional
        FORMATTERS: ClassVar[dict] = {                  # Optional
            "url": lambda value: value.lstrip("/"),
        }

        def validate(self, json_payload: dict, message_id: str) -> None:
            """Validate Ingestion Message"""
            # logger
            def _logger(msg: str, level: str):
                match level:
                    case 'info':
                        log.info(msg)
                    case 'debug':
                        log.debug(msg)
                    case 'warning':
                        log.warning(msg)
                    case 'error':
                        log.error(msg)
                    case 'critical':
                        log.critical(msg)
            # message formulation
            def _message_wrapper(log_msg: str):
                return msg(log_msg, pubsub_message_id=message_id)
            # using default schema validation
            return super().validate(json_payload, logger=_logger, message_wrapper=_message_wrapper)

    try:
        IngestionMessage.ALLOWED_EXTRA_KEYS=True
    except AttributeError as e:
        assert str(e) == "Cannot modify constant 'ALLOWED_EXTRA_KEYS' on ReadOnly Class IngestionMessage"
    
    data: dict = {
            "eventId":  '(True, str, None)',
            "username": '(True, str, None)',
            "url":      '/(True, str, None)/',
            "orgId":    '(True, str, None)',
            "tenancy":  '(True, str, None)',
            "orgName":  '(True, str, None)',
            "channel":  '(True, str, None)',
            "extra":    '(True, str, None)',
        }
    IngestionMessage().validate(
        data, '187129034567124876'
    )
    print(data)