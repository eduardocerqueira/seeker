//date: 2024-08-27T16:59:43Z
//url: https://api.github.com/gists/42d5d3e80b62c284432d3437c3dd2493
//owner: https://api.github.com/users/ziyuji-pillpack

package ncpdptypes

import (
	"time"
)

// BillReqPatient stores information about the patient that can be provided in a B1 (billing) request.
// See NCPDP D.0, p 67-68.
type BillReqPatient struct {
	ID               *PatientID            `qualifier:"331-CX X(2) optional" field:"332-CY X(20) optional"`
	DateOfBirth      time.Time             `field:"304-C4 9(8)"`
	Gender           GenderCode            `field:"305-C5 9(1)"`
	FirstName        *string               `field:"310-CA X(12) optional"`
	LastName         string                `field:"311-CB X(15)"`
	StreetAddress    *string               `field:"322-CM X(30) optional"`
	City             *string               `field:"323-CN X(20) optional"`
	State            *StateOrProvinceCode  `field:"324-CO X(2) optional"`
	Zip              *string               `field:"325-CP X(15) optional"`
	PlaceOfService   *PlaceOfServiceCode   `field:"307-C7 9(2) optional"`
	EmployerID       *string               `field:"333-CZ X(15) optional"`
	PregnancyStatus  *PregnancyStatusCode  `field:"335-2C X(1) optional"`
	PatientResidence *PatientResidenceCode `field:"384-4X 9(2) optional"`
	// 326-CQ, 350-HN: not supported because we don't support phone numbers or email addresses
	// Note: we don't attempt to validate whether or not EmployerID follows the IRS-specified format.
}

// PatientID stores an ID for a patient with an associated Kind
type PatientID struct {
	Kind PatientIDKind
	ID   string
}

// PatientIDKind stores the value of the "Patient ID Qualifier" field (331-CX).
type PatientIDKind eclBase

// PatientIDKinds stores all valid values of PatientIDKind
var PatientIDKinds = struct {
	SSN,
	AssignedByLTCFacility,
	DriversLicenseNumber,
	USMilitaryID,
	AssignedByPlan,
	AssignedByPlanSSNBased,
	MedicaidID,
	StateIssuedID,
	PassportID,
	MedicareBeneficiaryID,
	AssignedByEmployer,
	AssignedByPayer,
	AlienNumber,
	StudentVisaNumber,
	IndialTribalID,
	UPI,
	LexID,
	Other,
	MedicalRecordID PatientIDKind
}{
	SSN:                    PatientIDKind{"01", "Social Security Number"},
	AssignedByLTCFacility:  PatientIDKind{"1J", "Facility ID Number"},
	DriversLicenseNumber:   PatientIDKind{"02", "Driver's License Number"},
	USMilitaryID:           PatientIDKind{"03", "U.S. Military ID"},
	AssignedByPlan:         PatientIDKind{"04", "Non-SSN-based patient identifier assigned by health plan"},
	AssignedByPlanSSNBased: PatientIDKind{"05", "SSN-based patient identifier assigned by health plan"},
	MedicaidID:             PatientIDKind{"06", "Medicaid ID"},
	StateIssuedID:          PatientIDKind{"07", "State Issued ID"},
	PassportID:             PatientIDKind{"08", "Passport ID (or other ID assigned by a national government)"},
	MedicareBeneficiaryID:  PatientIDKind{"09", "Medicare Beneficiary ID"},
	AssignedByEmployer:     PatientIDKind{"10", "Employer Assigned ID"},
	AssignedByPayer:        PatientIDKind{"11", "Payer/PBM Assigned ID"},
	AlienNumber:            PatientIDKind{"12", "Alien Number (Government Permanent Residence Number)"},
	StudentVisaNumber:      PatientIDKind{"13", "Government Student VISA Number"},
	IndialTribalID:         PatientIDKind{"14", "Indian Tribal ID"},
	UPI:                    PatientIDKind{"15", "NCPDP Universal Patient Identifier (UPI)"},
	LexID:                  PatientIDKind{"16", "LexID Universal Patient Identifier (UPI)"},
	Other:                  PatientIDKind{"99", "Other"},
	MedicalRecordID:        PatientIDKind{"EA", "Medical Record Identification Number (EHR)"},
}

// Values returns a list of all valid values of this type
func (PatientIDKind) Values() values { return valuesFromStruct(PatientIDKinds) }

// Code returns the serialized value
func (c PatientIDKind) Code() string { return c.code }
