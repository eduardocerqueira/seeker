#date: 2024-08-21T17:02:35Z
#url: https://api.github.com/gists/5512cd7a080d2ce9df8496f9efc2cb88
#owner: https://api.github.com/users/rayansostenes

import datetime
import enum
from typing import Literal
from uuid import UUID


class UsStateEnum(enum.StrEnum): ...


class PolicyTypeEnum(enum.StrEnum):
    NEW = enum.auto()
    RENEWAL = enum.auto()
    REWRITE = enum.auto()
    ROLL_OVER = enum.auto()

    __display__ = {
        NEW: "New",
        RENEWAL: "Renewal",
        REWRITE: "Rewrite",
        ROLL_OVER: "Roll Over",
    }


class FirstNamedInsuredType(enum.StrEnum):
    INDIVIDUAL = enum.auto()
    PARTNERSHIP = enum.auto()
    CORPORATION = enum.auto()
    LLC = enum.auto()
    LLP = enum.auto()
    PC_PA = enum.auto()
    OTHER = enum.auto()

    __display__ = {
        INDIVIDUAL: "Individual",
        PARTNERSHIP: "Partnership",
        CORPORATION: "Corporation",
        LLC: "Limited Liability Company",
        LLP: "Limited Liability Partnership",
        PC_PA: "Professional Corporation/Professional Association",
        OTHER: "Other",
    }


class QuoteTypeEnum(enum.StrEnum):
    BINDABLE_QUOTE = enum.auto()
    INDICATION = enum.auto()

    __display__ = {
        BINDABLE_QUOTE: "Bindable Quote",
        INDICATION: "Preliminary Indication",
    }


class MedicalDegreeEnum(enum.StrEnum):
    DDS = enum.auto()
    DMD = enum.auto()
    MD = enum.auto()
    BDS = enum.auto()
    MS = enum.auto()

    __display__ = {
        DDS: "Doctor of Dental Surgery",
        DMD: "Doctor of Medicine in Dentistry",
        MD: "Doctor of Medicine",
        BDS: "Bachelor of Dental Surgery",
        MS: "Master of Science",
    }


class InsuranceLinesEnum(enum.StrEnum):
    PL = enum.auto()
    EPL = enum.auto()
    GL_ERISA_EBL = enum.auto()
    PROPERTY = enum.auto()

    __display__ = {
        PL: "Professional Liability",
        EPL: "Employment Practices Liability",
        GL_ERISA_EBL: "General Liability/ERISA Fiduciary & Employee Benefits Liability",
        PROPERTY: "Commercial Property",
    }


class NewDentistType(enum.StrEnum):
    NEW_GRAD = enum.auto()
    MILITARY_DENTIST = enum.auto()
    FOREIGN_GRAD = enum.auto()
    PUBLIC_SERVICE = enum.auto()

    __display__ = {
        NEW_GRAD: "New Grad",
        MILITARY_DENTIST: "Military Dentist",
        FOREIGN_GRAD: "Foreign Grad",
        PUBLIC_SERVICE: "Public Service",
    }


class WeeklyHoursEnum(enum.IntEnum):
    FULL_TIME = enum.auto()
    PART_TIME = enum.auto()
    MOONLIGHTING = enum.auto()

    __display__ = {
        FULL_TIME: "Full Time (More than 20 hours per week)",
        PART_TIME: "Part Time (20 hours or less per week)",
        MOONLIGHTING: "Moonlighting (10 Hours Or Less Per Week And For Second Job)",
    }


class Location:
    id: UUID
    address_1: str
    address_2: str | None = None
    city: str
    state: UsStateEnum
    zip_code: str
    county: str | None = None


class NamedInsured:
    type: FirstNamedInsuredType
    other_entity_type: str | None = None
    entity_name: str | None
    first_name: str | None
    middle_name: str | None
    last_name: str | None
    professional_designation: set[MedicalDegreeEnum] = set()


class Dentist:
    first_name: str
    middle_name: str | None = None
    last_name: str
    professional_designation: set[MedicalDegreeEnum] = set()
    primary_location: Location
    other_locations: list[Location] = []
    is_named_insured: bool = False
    years_in_practice: Literal[0, 1, 2, 3, 4]
    new_dentist_type: NewDentistType | None = None
    weekly_hours: WeeklyHoursEnum
    part_time_verification: bool = False
    """
    Agent certifies that insured has agreed to make available his her work schedule in the event we
    decide to verify part time status eligibility, or, Agent certifies that a part-time supplement
    was completed by insured and after your review, qualifies for part-time status
    """


class PolicyDetails:
    policy_type: PolicyTypeEnum

    roll_over_expiring_policy_number: str | None = None
    """
    The policy number of the expiring policy.
    Rules:
    - Required if the `policy_type` is `ROLL_OVER`.
    """

    prior_policy_number: str | None = None
    """
    The previous `Aspen` policy number.
    Rules:
    - Required if the `policy_type` is `RENEWAL` or `REWRITE`.
    """

    quote_type: QuoteTypeEnum

    policy_effective_date: datetime.date
    """
    Rules:
    - Policy Effective Date cannot be greater than 60 days in the future
    """

    desired_insurance_lines: set[InsuranceLinesEnum]
    """
    Rules:
    - `EPL` selectable only if `PL` is selected.
    - `GL_ERISA_EBL` selectable only if `PL` OR `PROPERTY` is selected.
    """
    named_insured: NamedInsured
    headquarters_state: UsStateEnum
    mailing_address: Location | None = None
    website: str | None = None
    office_phone_number: str | None = None
    cell_phone_number: str | None = None

    practice_locations: list[Location]
