#date: 2022-02-11T17:05:57Z
#url: https://api.github.com/gists/f4aa9a5b989b23734ca086724766006e
#owner: https://api.github.com/users/GrigoriiKushnir

@app.task(bind=True)
def generate_walmart_cookies(self):
    time_start = datetime.utcnow()
    logger.info('Start getting _px3 cookies')

    display_width, display_height = get_random_screen_resolution()
    profile_name = uuid.uuid4().hex
    chrome_options = get_chrome_options(
        profile=profile_name,
        display_height=display_height,
        display_width=display_width,
    )

    with Display(backend='xvfb', size=(display_width, display_height)):
        driver = create_chrome_driver(chrome_options)
        try:
            parse_walmart_items(driver)
        finally:
            driver.quit()
            shutil.rmtree(f'/var/tmp/{profile_name}')
            time_spent = (datetime.utcnow() - time_start).total_seconds()
            logger.info(f'Time spent: {time_spent}')

def parse_walmart_items(driver):
    redis_client = redis.Redis.from_url(REDIS_URL)
    walmart_item_ids = get_random_walmart_item_ids(CHROME_PROFILE_PAGES_LIMIT)

    for walmart_item_id in walmart_item_ids:
        cookie_wait = WebDriverWait(driver, CHROME_PAGE_COOKIE_WAIT_TIMEOUT)
        try:
            driver.get(f'https://www.walmart.com/ip/{walmart_item_id}')
            _px3_cookie = cookie_wait.until(lambda driver_: driver_.get_cookie("_px3"))
        except TimeoutException:
            logger.exception(f'Timeout on product {walmart_item_id}')
            continue

        logger.info(f'_px3: "{dig(_px3_cookie, "value")}"')

        if _px3_cookie is not None:
            redis_client.zadd(
                REDIS_ACTIVE_WALMART_PX3_COOKIES_SET,
                {
                    dig(_px3_cookie, "value"): int(datetime.utcnow().timestamp())
                }
            )

        driver.delete_all_cookies()
        driver.proxy = ProxyHelper.get_proxies()