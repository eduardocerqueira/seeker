#date: 2025-05-29T17:01:27Z
#url: https://api.github.com/gists/124dc13093c9f7e0c29daf565cc6019e
#owner: https://api.github.com/users/KASVIK26

#!/usr/bin/env python3
"""
Credit Card Payment Test
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
from datetime import datetime

class CorrectedCreditCardTester:
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        # Create screenshots directory
        self.screenshot_dir = f"screenshots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
    def setup_driver(self):
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.wait = WebDriverWait(self.driver, 20)
        self.driver.set_window_size(1920, 1080)
    
    def teardown_driver(self):
        if self.driver:
            self.driver.quit()
    
    def take_screenshot(self, name):
        """Take screenshot with timestamp"""
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"{self.screenshot_dir}/{timestamp}_{name}.png"
        self.driver.save_screenshot(filename)
        print(f"📸 Screenshot saved: {filename}")
        return filename
    
    def enhanced_input(self, element, value, field_name="field"):
        """Enhanced input method with multiple strategies"""
        try:
            # Clear field first
            element.clear()
            time.sleep(0.5)
            
            # Method 1: Direct send_keys
            element.send_keys(value)
            time.sleep(0.5)
            
            # Verify input
            current_value = element.get_attribute('value')
            if current_value == str(value):
                print(f"✓ {field_name} entered successfully: {value}")
                return True
            
            # Method 2: JavaScript input if direct failed
            print(f"Retrying {field_name} input with JavaScript...")
            element.clear()
            self.driver.execute_script(f"arguments[0].value = '{value}';", element)
            self.driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", element)
            time.sleep(0.5)
            
            # Verify again
            current_value = element.get_attribute('value')
            if current_value == str(value):
                print(f"✓ {field_name} entered with JS: {value}")
                return True
                
            # Method 3: Character by character
            print(f"Retrying {field_name} character by character...")
            element.clear()
            for char in str(value):
                element.send_keys(char)
                time.sleep(0.1)
            
            final_value = element.get_attribute('value')
            print(f"✓ {field_name} final value: {final_value}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to input {field_name}: {e}")
            return False
    
    def login(self):
        """Perform login"""
        try:
            print("🔐 Starting login process...")
            self.driver.get("https://preview--utility-bill-blunders-project.lovable.app/")
            self.take_screenshot("01_initial_page")
            
            # Wait for login form
            account_input = self.wait.until(EC.presence_of_element_located((By.ID, "account")))
            zip_input = self.driver.find_element(By.ID, "zipcode")
            
            # Enhanced input for login fields
            self.enhanced_input(account_input, "87654321", "Account Number")
            self.enhanced_input(zip_input, "90210", "ZIP Code")
            
            self.take_screenshot("02_login_filled")
            
            # Click sign in
            sign_in_button = self.driver.find_element(By.XPATH, "//button[text()='Sign In']")
            sign_in_button.click()
            
            # Wait for dashboard
            self.wait.until(EC.url_contains("/dashboard"))
            self.take_screenshot("03_dashboard_loaded")
            print("✓ Login successful")
            return True
            
        except Exception as e:
            print(f"✗ Login failed: {e}")
            self.take_screenshot("ERROR_login_failed")
            return False
    
    def navigate_to_payment(self):
        """Navigate to payment page"""
        try:
            print("🧭 Navigating to payment...")
            
            pay_selectors = [
                "//button[contains(text(), 'Pay Now')]",
                "//button[contains(text(), 'Pay Bill')]",
                "//a[contains(text(), 'Pay Now')]",
                "//button[contains(text(), 'Make Payment')]"
            ]
            
            for selector in pay_selectors:
                try:
                    pay_button = self.driver.find_element(By.XPATH, selector)
                    if pay_button.is_displayed():
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", pay_button)
                        time.sleep(1)
                        pay_button.click()
                        time.sleep(3)
                        self.take_screenshot("04_payment_page_loaded")
                        print("✓ Payment page loaded")
                        return True
                except NoSuchElementException:
                    continue
                    
            print("✗ Pay button not found")
            self.take_screenshot("ERROR_pay_button_not_found")
            return False
            
        except Exception as e:
            print(f"✗ Navigation failed: {e}")
            self.take_screenshot("ERROR_navigation_failed")
            return False
    
    def select_credit_card_and_set_amount(self):
        """Select credit card and set full amount (both on same page)"""
        try:
            print("💳 Selecting credit card and setting full amount...")
            
            # First select credit card radio button
            credit_card_selected = False
            credit_selectors = [
                "//input[@type='radio' and following-sibling::text()[contains(., 'Credit')]]",
                "//input[@type='radio'][1]",  # First radio button (likely credit card)
                "//label[contains(text(), 'Credit/Debit Card')]/../input[@type='radio']",
                "//input[@type='radio' and @value='credit_card']"
            ]
            
            for selector in credit_selectors:
                try:
                    radio_button = self.driver.find_element(By.XPATH, selector)
                    if radio_button.is_displayed():
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", radio_button)
                        time.sleep(1)
                        self.driver.execute_script("arguments[0].click();", radio_button)
                        credit_card_selected = True
                        print("✓ Credit card radio button selected")
                        break
                except NoSuchElementException:
                    continue
            
            if not credit_card_selected:
                print("⚠️ Credit card radio button not found, trying label click...")
                try:
                    credit_label = self.driver.find_element(By.XPATH, "//label[contains(text(), 'Credit/Debit Card')]")
                    credit_label.click()
                    credit_card_selected = True
                    print("✓ Credit card selected via label")
                except:
                    print("✗ Could not select credit card")
                    return False
            
            time.sleep(2)
            
            # Now click Full Amount button
            print("💰 Clicking Full Amount button...")
            full_amount_selectors = [
                "//button[contains(text(), 'Full Amount')]",
                "//button[text()='Full Amount']",
                "//input[@value='Full Amount']",
                "//span[text()='Full Amount']/.."
            ]
            
            full_amount_clicked = False
            for selector in full_amount_selectors:
                try:
                    button = self.driver.find_element(By.XPATH, selector)
                    if button.is_displayed() and button.is_enabled():
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                        time.sleep(1)
                        self.driver.execute_script("arguments[0].click();", button)
                        full_amount_clicked = True
                        print("✓ Full Amount button clicked")
                        break
                except NoSuchElementException:
                    continue
            
            if not full_amount_clicked:
                print("⚠️ Full Amount button not found, amount may be pre-selected")
            
            self.take_screenshot("05_credit_card_and_amount_selected")
            time.sleep(2)
            return True
            
        except Exception as e:
            print(f"✗ Credit card/amount selection failed: {e}")
            self.take_screenshot("ERROR_credit_selection")
            return False
    
    def click_continue_to_card_details(self):
        """Click Continue to go to card details page"""
        try:
            print("▶️ Clicking Continue to go to card details...")
            
            continue_selectors = [
                "//button[contains(text(), 'Continue')]",
                "//button[text()='Continue']",
                "//button[contains(text(), 'Next')]",
                "//button[contains(text(), 'Proceed')]"
            ]
            
            for selector in continue_selectors:
                try:
                    button = self.driver.find_element(By.XPATH, selector)
                    if button.is_displayed() and button.is_enabled():
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                        time.sleep(1)
                        button.click()
                        time.sleep(3)
                        self.take_screenshot("06_card_details_page_loaded")
                        print("✓ Card details page loaded")
                        return True
                except NoSuchElementException:
                    continue
            
            print("✗ Continue button not found")
            self.take_screenshot("ERROR_continue_not_found")
            return False
            
        except Exception as e:
            print(f"✗ Continue click failed: {e}")
            self.take_screenshot("ERROR_continue_failed")
            return False
    
    def fill_credit_card_details(self):
        """Fill credit card details on the card information page"""
        try:
            print("📝 Filling credit card details...")
            
            # Wait for card form to be present
            time.sleep(2)
            
            # Card Number - try different approaches
            card_number_filled = False
            card_number_selectors = [
                "//input[contains(@placeholder, 'Card Number')]",
                "//input[contains(@placeholder, 'card number')]",
                "//label[text()='Card Number:']/following-sibling::input",
                "//label[contains(text(), 'Card Number')]/following-sibling::*//input",
                "//input[@type='text'][1]"  # First text input (likely card number)
            ]
            
            for selector in card_number_selectors:
                try:
                    card_field = self.driver.find_element(By.XPATH, selector)
                    if card_field.is_displayed():
                        self.enhanced_input(card_field, "4111111111111111", "Card Number")
                        card_number_filled = True
                        break
                except NoSuchElementException:
                    continue
            
            if not card_number_filled:
                print("✗ Card number field not found")
                return False
            
            # Expiry Date - try different formats
            expiry_selectors = [
                "//input[contains(@placeholder, 'MM/YY')]",
                "//input[contains(@placeholder, 'Expiry')]",
                "//label[text()='Expiry:']/following-sibling::input",
                "//label[contains(text(), 'Expiry')]/following-sibling::*//input"
            ]
            
            expiry_filled = False
            for selector in expiry_selectors:
                try:
                    expiry_field = self.driver.find_element(By.XPATH, selector)
                    if expiry_field.is_displayed():
                        self.enhanced_input(expiry_field, "12/25", "Expiry Date")
                        expiry_filled = True
                        break
                except NoSuchElementException:
                    continue
            
            # CVV
            cvv_selectors = [
                "//input[contains(@placeholder, 'CVV')]",
                "//input[contains(@placeholder, 'CVC')]",
                "//label[text()='CVV:']/following-sibling::input",
                "//label[contains(text(), 'CVV')]/following-sibling::*//input"
            ]
            
            for selector in cvv_selectors:
                try:
                    cvv_field = self.driver.find_element(By.XPATH, selector)
                    if cvv_field.is_displayed():
                        self.enhanced_input(cvv_field, "123", "CVV")
                        break
                except NoSuchElementException:
                    continue
            
            # Name on Card
            name_selectors = [
                "//input[contains(@placeholder, 'Name')]",
                "//label[text()='Name on Card:']/following-sibling::input",
                "//label[contains(text(), 'Name on Card')]/following-sibling::*//input"
            ]
            
            for selector in name_selectors:
                try:
                    name_field = self.driver.find_element(By.XPATH, selector)
                    if name_field.is_displayed():
                        self.enhanced_input(name_field, "John Doe", "Name on Card")
                        break
                except NoSuchElementException:
                    continue
            
            # Billing Address - Address field
            address_selectors = [
                "//input[contains(@placeholder, 'Address')]",
                "//label[text()='Address:']/following-sibling::input",
                "//label[contains(text(), 'Address')]/following-sibling::*//input",
                "//h3[text()='Billing Address']/following-sibling::*//input[1]"
            ]
            
            for selector in address_selectors:
                try:
                    address_field = self.driver.find_element(By.XPATH, selector)
                    if address_field.is_displayed():
                        self.enhanced_input(address_field, "123 Test Street", "Address")
                        break
                except NoSuchElementException:
                    continue
            
            # City
            city_selectors = [
                "//input[contains(@placeholder, 'City')]",
                "//label[text()='City:']/following-sibling::input",
                "//label[contains(text(), 'City')]/following-sibling::*//input"
            ]
            
            for selector in city_selectors:
                try:
                    city_field = self.driver.find_element(By.XPATH, selector)
                    if city_field.is_displayed():
                        self.enhanced_input(city_field, "Test City", "City")
                        break
                except NoSuchElementException:
                    continue
            
            # State
            state_selectors = [
                "//input[contains(@placeholder, 'State')]",
                "//label[text()='State:']/following-sibling::input",
                "//label[contains(text(), 'State')]/following-sibling::*//input"
            ]
            
            for selector in state_selectors:
                try:
                    state_field = self.driver.find_element(By.XPATH, selector)
                    if state_field.is_displayed():
                        self.enhanced_input(state_field, "CA", "State")
                        break
                except NoSuchElementException:
                    continue
            
            # ZIP
            zip_selectors = [
                "//input[contains(@placeholder, 'ZIP')]",
                "//label[text()='ZIP:']/following-sibling::input",
                "//label[contains(text(), 'ZIP')]/following-sibling::*//input"
            ]
            
            for selector in zip_selectors:
                try:
                    zip_field = self.driver.find_element(By.XPATH, selector)
                    if zip_field.is_displayed():
                        self.enhanced_input(zip_field, "90210", "ZIP Code")
                        break
                except NoSuchElementException:
                    continue
            
            # Check "Save this card" checkbox if present
            try:
                save_checkbox = self.driver.find_element(By.XPATH, "//input[@type='checkbox']")
                if save_checkbox.is_displayed() and not save_checkbox.is_selected():
                    self.driver.execute_script("arguments[0].click();", save_checkbox)
                    print("✓ Save card checkbox selected")
            except:
                pass
            
            self.take_screenshot("07_card_details_filled")
            print("✓ Credit card details filled")
            return True
            
        except Exception as e:
            print(f"✗ Credit card filling failed: {e}")
            self.take_screenshot("ERROR_card_filling")
            return False
    
    def process_payment_and_wait_for_result(self):
        """Click Process Payment and wait for success"""
        try:
            print("🎯 Processing payment...")
            
            # Click Process Payment button
            process_selectors = [
                "//button[contains(text(), 'Process Payment')]",
                "//button[text()='Process Payment']",
                "//button[contains(text(), 'Pay Now')]",
                "//button[contains(text(), 'Submit Payment')]"
            ]
            
            process_clicked = False
            for selector in process_selectors:
                try:
                    button = self.driver.find_element(By.XPATH, selector)
                    if button.is_displayed() and button.is_enabled():
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", button)
                        time.sleep(1)
                        button.click()
                        process_clicked = True
                        print("✓ Process Payment button clicked")
                        break
                except NoSuchElementException:
                    continue
            
            if not process_clicked:
                print("✗ Process Payment button not found")
                self.take_screenshot("ERROR_process_button_not_found")
                return False, "Process button not found"
            
            self.take_screenshot("08_process_payment_clicked")
            
            # Wait for processing page
            print("⏳ Waiting for processing...")
            time.sleep(3)
            
            # Look for processing indicators
            processing_found = False
            processing_selectors = [
                "//div[contains(text(), 'Processing')]",
                "//span[contains(text(), 'Processing')]",
                "//div[contains(text(), 'Processing Payment')]",
                "//h1[contains(text(), 'UTILIPAY')]",
                "//div[contains(@class, 'processing')]"
            ]
            
            for selector in processing_selectors:
                try:
                    element = self.driver.find_element(By.XPATH, selector)
                    if element.is_displayed():
                        processing_found = True
                        self.take_screenshot("09_processing_detected")
                        print("✓ Processing page detected")
                        break
                except NoSuchElementException:
                    continue
            
            # Wait for success page
            print("⏳ Waiting for payment result...")
            time.sleep(10)  # Give time for processing
            
            # Check for success
            success_selectors = [
                "//div[contains(text(), 'Payment Successful')]",
                "//h1[contains(text(), 'Payment Successful')]",
                "//div[contains(text(), 'Payment Successful!')]",
                "//span[contains(text(), 'Payment Successful')]"
            ]
            
            success_found = False
            success_message = ""
            
            for selector in success_selectors:
                try:
                    element = self.driver.find_element(By.XPATH, selector)
                    if element.is_displayed():
                        success_found = True
                        success_message = element.text
                        break
                except NoSuchElementException:
                    continue
            
            # Also check page content
            page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            
            if not success_found:
                success_keywords = ["payment successful", "successful", "confirmation number", "amount paid"]
                for keyword in success_keywords:
                    if keyword in page_text:
                        success_found = True
                        success_message = f"Success detected via keyword: {keyword}"
                        break
            
            if success_found:
                self.take_screenshot("10_PAYMENT_SUCCESS")
                print(f"🎉 PAYMENT SUCCESSFUL: {success_message}")
                
                # Try to capture confirmation details
                try:
                    confirmation_elements = self.driver.find_elements(By.XPATH, "//div[contains(text(), 'Confirmation Number')]|//div[contains(text(), 'Amount Paid')]")
                    for element in confirmation_elements:
                        print(f"📋 {element.text}")
                except:
                    pass
                
                return True, success_message
            else:
                self.take_screenshot("10_PAYMENT_RESULT_UNCLEAR")
                print("❓ Payment result unclear")
                return False, "Payment result unclear"
                
        except Exception as e:
            print(f"✗ Payment processing failed: {e}")
            self.take_screenshot("ERROR_payment_processing")
            return False, str(e)
    
    def run_corrected_payment_test(self):
        """Run the corrected payment test following actual UI flow"""
        print("="*60)
        print("🚀 CORRECTED CREDIT CARD PAYMENT TEST")
        print("="*60)
        
        start_time = time.time()
        
        try:
            self.setup_driver()
            
            # Step 1: Login
            if not self.login():
                return False, "Login failed"
            
            # Step 2: Navigate to payment
            if not self.navigate_to_payment():
                return False, "Navigation failed"
            
            # Step 3: Select credit card and set full amount (same page)
            if not self.select_credit_card_and_set_amount():
                return False, "Credit card/amount selection failed"
            
            # Step 4: Click Continue to go to card details
            if not self.click_continue_to_card_details():
                return False, "Continue to card details failed"
            
            # Step 5: Fill credit card details
            if not self.fill_credit_card_details():
                return False, "Credit card details filling failed"
            
            # Step 6: Process payment and check result
            success, message = self.process_payment_and_wait_for_result()
            
            duration = round(time.time() - start_time, 2)
            
            if success:
                print(f"🎉 TEST COMPLETED SUCCESSFULLY in {duration}s")
                print(f"📸 Screenshots saved in: {self.screenshot_dir}")
                return True, f"Success: {message}"
            else:
                print(f"❌ TEST FAILED in {duration}s: {message}")
                print(f"📸 Screenshots saved in: {self.screenshot_dir}")
                return False, message
                
        except Exception as e:
            duration = round(time.time() - start_time, 2)
            print(f"💥 CRITICAL ERROR in {duration}s: {e}")
            self.take_screenshot("CRITICAL_ERROR")
            return False, f"Critical error: {str(e)}"
            
        finally:
            self.teardown_driver()

def main():
    tester = CorrectedCreditCardTester()
    
    print("💳 Corrected Credit Card Payment Tester")
    print("Based on actual UI flow analysis")
    print("="*60)
    
    try:
        success, message = tester.run_corrected_payment_test()
        
        print("\n" + "="*60)
        if success:
            print("🎊 FINAL RESULT: PAYMENT TEST SUCCESSFUL")
        else:
            print("💔 FINAL RESULT: PAYMENT TEST FAILED")
        print(f"📋 Details: {message}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

if __name__ == "__main__":
    main()