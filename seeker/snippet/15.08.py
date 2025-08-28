#date: 2025-08-28T17:01:24Z
#url: https://api.github.com/gists/c711c05a07bf7f10e5ad80a47c40d3c9
#owner: https://api.github.com/users/Dmytro-Pin

<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Форма реєстрації</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <form action="index.html" method="get" class="registration-form">
            
            <div class="form-section">
                <h2>Registration form</h2>
                
                <div class="input-group">
                    <label class="input-label">
                        <img src="img/username.png" alt="User_icon" class="icon">
                        <input type="text" name="username" placeholder="Username" required>
                    </label>
                </div>

                <div class="input-group">
                    <label class="input-label">
                        <img src="img/email.png" alt="User_icon" class="icon">
                        <input type="email" name="email" placeholder="Email address" required>
                    </label>
                </div>

                <div class= "**********"
                    <label class="input-label">
                        <img src="img/pswd.png" alt="User_icon" class="icon">
                        <input type= "**********"="password" placeholder="Password" required>
                    </label>
                    <label class="input-label">
                        <span class="icon lock-icon"></span>
                        <input type= "**********"="confirm_password" placeholder="Confirm password" required>
                    </label>
                </div>
            </div>

            <div class="form-section personal-details">
                <h2>Personal details</h2>
                
                <div class="name-group">
                    <label>
                        <input type="text" name="first_name" placeholder="First name" required>
                    </label>
                    <label>
                        <input type="text" name="last_name" placeholder="Last name" required>
                    </label>
                </div>

                <div class="details-group">
                    <label>
                        <select name="gender" required>
                            <option value="">Empty</option>
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                            <option value="other">Other</option>
                        </select>
                    </label>
                    <label>
                        <input type="text" name="birth_date" placeholder="mm/dd/yyyy" pattern="\d{2}/\d{2}/\d{4}">
                    </label>
                </div>
            </div>

            <div class="checkbox-group">
                <label class="checkbox-label">
                    <input type="checkbox" name="newsletter">
                    I want to receive news and special offers
                </label>
                
                <label class="checkbox-label">
                    <input type="checkbox" name="terms" required>
                    I agree with the Terms and Conditions
                </label>
            </div>

            <button type="submit" class="submit-btn">SUBMIT</button>
        </form>
    </div>
</body>
</html>
</html>