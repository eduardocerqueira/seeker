//date: 2024-08-16T16:52:44Z
//url: https://api.github.com/gists/2937ebacefc46368256725d190b1b4bc
//owner: https://api.github.com/users/Raja696969

@SpringBootTest
public class PatientControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    public void testRegisterPatient() throws Exception {
        mockMvc.perform(post("/api/patients")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"name\": \"John Doe\", \"contactDetails\": \"1234567890\", \"medicalHistory\": \"None\"}"))
                .andExpect(status().isOk());
    }
}
