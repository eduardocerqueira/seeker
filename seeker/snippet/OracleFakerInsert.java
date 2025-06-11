//date: 2025-06-11T17:09:06Z
//url: https://api.github.com/gists/a5f827fd36288c19ae6dfbfc00079c59
//owner: https://api.github.com/users/dkirrane

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Timestamp;
import java.time.LocalDateTime;

import com.github.javafaker.Faker;

public class OracleFakerInsert {
    public static void main(String[] args) throws Exception {
        Faker faker = new Faker();

        // JDBC connection string
        String url = "jdbc:oracle:thin:@//HOST:PORT/SERVICE";  // e.g. @//localhost:1521/XEPDB1
        String user = "your_username";
        String password = "**********"

        // Load Oracle driver (optional in newer JVMs)
        Class.forName("oracle.jdbc.driver.OracleDriver");

        try (Connection conn = "**********"
            System.out.println("Connected to Oracle DB.");

            // Insert fake users
            String sql = "INSERT INTO users (id, name, email, created_at) VALUES (?, ?, ?, ?)";
            PreparedStatement stmt = conn.prepareStatement(sql);

            for (int i = 1; i <= 10; i++) {
                stmt.setInt(1, i); // Assuming 'id' is not auto-generated
                stmt.setString(2, faker.name().fullName());
                stmt.setString(3, faker.internet().emailAddress());
                stmt.setTimestamp(4, Timestamp.valueOf(LocalDateTime.now().minusDays(faker.number().numberBetween(0, 365))));

                stmt.executeUpdate();
            }

            System.out.println("Inserted 10 fake records into Oracle table.");
        }
    }
}
ds into Oracle table.");
        }
    }
}
