//date: 2024-08-16T16:52:44Z
//url: https://api.github.com/gists/2937ebacefc46368256725d190b1b4bc
//owner: https://api.github.com/users/Raja696969

public interface PatientRepository extends JpaRepository<Patient, Long> {}

public interface DoctorRepository extends JpaRepository<Doctor, Long> {}

public interface AppointmentRepository extends JpaRepository<Appointment, Long> {}

public interface MedicationRepository extends JpaRepository<Medication, Long> {}
