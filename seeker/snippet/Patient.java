//date: 2024-08-16T16:52:44Z
//url: https://api.github.com/gists/2937ebacefc46368256725d190b1b4bc
//owner: https://api.github.com/users/Raja696969

// Patient.java
@Entity
public class Patient {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private String contactDetails;
    private String medicalHistory;
    // Getters and Setters
}

// Doctor.java
@Entity
public class Doctor {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private String specialization;
    // Getters and Setters
}

// Appointment.java
@Entity
public class Appointment {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private Date appointmentDate;
    
    @ManyToOne
    private Patient patient;

    @ManyToOne
    private Doctor doctor;
    // Getters and Setters
}

// Medication.java
@Entity
public class Medication {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private String dosage;

    @ManyToOne
    private Patient patient;
    // Getters and Setters
}
