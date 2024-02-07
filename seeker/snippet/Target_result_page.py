#date: 2024-02-07T16:59:42Z
#url: https://api.github.com/gists/52882ed6b93ffd951bb110299b2ae1e9
#owner: https://api.github.com/users/Farzmsh

@then('verify there are 5 benefit boxes in Target circle')
def verify_search_results_correct(context):
    Benefit_box = context.driver.find_elements(By.XPATH, "//li[contains(@class,'styles__BenefitCard-sc-9mx6dj-2')]")
    assert len(Benefit_box) == 5, f"there are {len(Benefit_box)} benefit boxes in Target circle "
    print("Test case passed")