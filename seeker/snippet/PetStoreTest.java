//date: 2024-12-26T16:37:47Z
//url: https://api.github.com/gists/1119f2581924df66df420631b3d85e84
//owner: https://api.github.com/users/borodicht

import io.restassured.http.ContentType;
import org.testng.annotations.Test;

import static io.restassured.RestAssured.given;
import static org.hamcrest.Matchers.equalTo;

public class PetStoreTest {

    //Изолированные
    //CRUD

    @Test
    public void checkCreatePetWithoutBody() {
        given()
                .log().all()
                .contentType(ContentType.JSON)
                .body("")
        .when()
                .post("https://petstore.swagger.io/v2/pet")
        .then()
                .log().all()
                .statusCode(405)
                .body("code", equalTo(405))
                .body("type", equalTo("unknown"))
                .body("message", equalTo("no data"));
    }
}
