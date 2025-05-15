#date: 2025-05-15T17:06:53Z
#url: https://api.github.com/gists/3223b465c4117d83c3fab761a33f4e36
#owner: https://api.github.com/users/joeblackwaslike

# ────────────── Scraper Logic ────────────── #
async def scrape_case_broward(case_number: str) -> CaseResult:
    # Check for cancellation at startup
    check_cancellation(case_number)
    check_cancellation_file(case_number)

    start_time = time.time()
    logger.info(f"🔍 [BROWARD] Starting scrape for case: {case_number}")

    # Initialize progress tracking
    progress = {
        "case_info_complete": False,
        "dispositions_complete": False,
        "events_complete": False,
        "total_files": 0,
        "downloaded_files": 0,
        "metadata_count": 0,
        "start_time": start_time,
    }

    base_dir = get_download_path(case_number)
    events_dir = base_dir / "events"
    dispo_dir = base_dir / "dispositions"
    ensure_directory(events_dir)
    ensure_directory(dispo_dir)

    downloaded_files: list[str] = []
    skipped_rows: list[int] = []
    metadata: list[dict[str, str]] = []

    logger.info(f"🏗️ [BROWARD] Set up directories for case {case_number}")

    # Set up a task to periodically check for cancellation
    async def periodic_cancellation_check():
        while True:
            # Check every 2 seconds
            await asyncio.sleep(2)
            if check_cancellation(case_number) or check_cancellation_file(case_number):
                return

    # Create the cancellation check task
    cancellation_task = asyncio.create_task(periodic_cancellation_check())

    # Track browser launch attempts
    browser_launch_attempts = 0
    max_browser_launch_attempts = 3

    try:
        async with async_playwright() as p:
            # Keep trying to launch browser with different proxy ports until successful or max attempts reached
            while browser_launch_attempts < max_browser_launch_attempts:
                # Check cancellation again before browser launch (expensive operation)
                check_cancellation(case_number)
                check_cancellation_file(case_number)

                browser_launch_attempts += 1
                logger.info(
                    f"🌐 [BROWARD] Launching browser for case {case_number} (attempt {browser_launch_attempts}/{max_browser_launch_attempts})"
                )

                try:
                    browser, context, page = await launch_browser(p, county="broward")

                    # Check if proxy is being used before testing connection
                    proxy_enabled = is_proxy_enabled("broward")

                    # Try to navigate to a test page to verify proxy connection
                    try:
                        time.sleep(1)
                        if proxy_enabled:
                            logger.info("🧪 [BROWARD] Testing proxy connection...")
                            await page.goto("https://httpbin.org/ip", timeout=10000)

                            # More robust way to get the IP - handle both JSON and text formats
                            try:
                                # Try to parse as JSON first
                                ip_json = await page.evaluate(
                                    "() => { try { return JSON.parse(document.body.textContent); } catch(e) { return null; } }"
                                )
                                if ip_json and "origin" in ip_json:
                                    logger.info(
                                        f"✅ [BROWARD] Proxy connection successful: IP={ip_json['origin']}"
                                    )
                                else:
                                    # Fallback to just getting the text content
                                    content = await page.evaluate(
                                        "() => document.body.textContent.trim()"
                                    )
                                    logger.info(
                                        f"✅ [BROWARD] Proxy connection successful: Response={content}"
                                    )
                            except Exception as parse_err:
                                logger.warning(
                                    f"⚠️ [BROWARD] Could not parse IP response: {parse_err}, but connection succeeded"
                                )
                                # Continue anyway as we successfully connected
                        else:
                            logger.info("🔌 [BROWARD] Proxy disabled, skipping connection test")
                        break  # Connection successful or proxy disabled, exit the retry loop
                    except Exception as e:
                        if proxy_enabled:
                            logger.warning(
                                f"⚠️ [BROWARD] Proxy connection test failed on attempt {browser_launch_attempts}: {e}"
                            )
                            await context.close()
                            await browser.close()
                            if browser_launch_attempts >= max_browser_launch_attempts:
                                raise Exception(
                                    f"Failed to establish proxy connection after {max_browser_launch_attempts} attempts"
                                )
                            logger.info(
                                "⏳ [BROWARD] Retrying with different proxy port in 2 seconds..."
                            )
                            await asyncio.sleep(2)
                            continue
                        # If proxy is disabled but we still got an error, log it but continue
                        logger.warning(
                            f"⚠️ [BROWARD] Connection test failed even with proxy disabled: {e}"
                        )
                        break  # Continue with execution despite error

                except Exception as e:
                    logger.warning(
                        f"⚠️ [BROWARD] Browser launch failed on attempt {browser_launch_attempts}: {e}"
                    )
                    if browser_launch_attempts >= max_browser_launch_attempts:
                        raise Exception(
                            f"Failed to launch browser after {max_browser_launch_attempts} attempts: {e}"
                        )
                    logger.info("⏳ [BROWARD] Retrying with different proxy port in 2 seconds...")
                    await asyncio.sleep(2)

            try:
                # Case Info
                try:
                    # Check cancellation before each major operation
                    check_cancellation(case_number)
                    check_cancellation_file(case_number)

                    logger.info(f"📋 [BROWARD] Fetching case info for {case_number}")
                    case_info_results = await scrape_case_info(case_number, page)
                    metadata += case_info_results
                    progress["case_info_complete"] = True
                    progress["metadata_count"] += len(case_info_results)

                    logger.info(
                        f"✅ [BROWARD] Case info scraped for {case_number} - Found {len(case_info_results)} entries"
                    )
                except Exception as e:
                    logger.error(f"❌ [BROWARD] Failed to scrape case info for {case_number}: {e}")
                    raise Exception(f"ScrapeCaseInfoError: Failed for {case_number}") from e

                # Check cancellation between major steps
                check_cancellation(case_number)
                check_cancellation_file(case_number)
                await reload_case_detail_view(page, case_number)

                # Dispositions (retry up to 3 times)
                logger.info(f"📄 [BROWARD] Starting dispositions scrape for {case_number}")
                for attempt in range(1, 4):
                    # Check cancellation before each attempt
                    check_cancellation(case_number)
                    check_cancellation_file(case_number)

                    try:
                        dispo_results = await scrape_dispositions(
                            case_number, page, dispo_dir, downloaded_files
                        )
                        metadata += dispo_results
                        progress["dispositions_complete"] = True
                        progress["metadata_count"] += len(dispo_results)
                        progress["total_files"] += len(dispo_results)
                        progress["downloaded_files"] += len(dispo_results)

                        logger.info(
                            f"✅ [BROWARD] Dispositions scraped for {case_number} - Found {len(dispo_results)} documents on attempt {attempt}"
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            f"⚠️ [BROWARD] Attempt {attempt} to scrape dispositions for {case_number} failed: {e}"
                        )
                        if attempt == 3:
                            logger.error(
                                f"❌ [BROWARD] Dispositions scraping for {case_number} ultimately failed after 3 attempts"
                            )
                            raise Exception(
                                f"ScrapeDispositionsError: Failed for {case_number}"
                            ) from e

                # Check cancellation between major steps
                check_cancellation(case_number)
                check_cancellation_file(case_number)
                await reload_case_detail_view(page, case_number)

                # Events
                try:
                    # Check cancellation before each major operation
                    check_cancellation(case_number)
                    check_cancellation_file(case_number)

                    logger.info(f"📅 [BROWARD] Starting events scrape for {case_number}")
                    events_results = await scrape_events(
                        case_number, page, events_dir, downloaded_files, skipped_rows
                    )
                    metadata += events_results
                    progress["events_complete"] = True
                    progress["metadata_count"] += len(events_results)
                    progress["total_files"] += len(events_results)
                    progress["downloaded_files"] += len(events_results)

                    if skipped_rows:
                        logger.warning(
                            f"⚠️ [BROWARD] Skipped {len(skipped_rows)} rows during events scrape for {case_number}"
                        )

                    logger.info(
                        f"✅ [BROWARD] Events scraped for {case_number} - Found {len(events_results)} documents"
                    )
                except Exception as e:
                    logger.error(f"❌ [BROWARD] Failed to scrape events for {case_number}: {e}")
                    raise Exception(f"ScrapeEventsError: Failed for {case_number}") from e

            finally:
                await context.close()
                await browser.close()
                logger.info(f"🧹 [BROWARD] Browser closed for case {case_number}")

    except asyncio.CancelledError:
        # Handle cancellation gracefully
        logger.warning(f"🛑 [BROWARD] Scraper task was cancelled for case {case_number}")
        return {
            "caseNumber": case_number,
            "status": "cancelled",
            "downloadedFiles": downloaded_files,
            "metadata": metadata,
            "statistics": {
                "status": "cancelled",
                "total_metadata_entries": len(metadata),
                "downloaded_files": len(downloaded_files),
                "skipped_rows": len(skipped_rows) if "skipped_rows" in locals() else 0,
            },
            "progress": progress,
        }
    finally:
        # Always clean up the cancellation task
        if "cancellation_task" in locals() and not cancellation_task.done():
            cancellation_task.cancel()
            try:
                await cancellation_task
            except asyncio.CancelledError:
                pass

    # Calculate final stats
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    progress["end_time"] = end_time
    progress["duration"] = duration

    # Final cancellation check before returning
    check_cancellation(case_number)
    check_cancellation_file(case_number)

    # Save metadata to json file
    if metadata:
        logger.info(f"💾 [BROWARD] Saving metadata to JSON for case {case_number}")
        save_metadata_json(case_number, base_dir, metadata)

    # Log completion statistics
    logger.info(f"🎯 [BROWARD] Scrape completed for case {case_number}")
    logger.info(f"⏱️ [BROWARD] Duration: {duration} seconds")
    logger.info(f"📊 [BROWARD] Statistics for {case_number}:")
    logger.info(f"   - Metadata entries: {len(metadata)}")
    logger.info(f"   - Downloaded files: {len(downloaded_files)}")
    if skipped_rows:
        logger.info(f"   - Skipped rows: {len(skipped_rows)}")

    # Categorize metadata by type
    metadata_by_type = {}
    for entry in metadata:
        entry_type = entry.get("type", "unknown")
        if entry_type not in metadata_by_type:
            metadata_by_type[entry_type] = 0
        metadata_by_type[entry_type] += 1

    # Log metadata breakdown
    for entry_type, count in metadata_by_type.items():
        logger.info(f"   - {entry_type}: {count} entries")

    # Return enhanced result with progress and stats
    return {
        "caseNumber": case_number,
        "downloadedFiles": downloaded_files,
        "metadata": metadata,
        "statistics": {
            "duration": duration,
            "total_metadata_entries": len(metadata),
            "downloaded_files": len(downloaded_files),
            "skipped_rows": len(skipped_rows),
            "metadata_by_type": metadata_by_type,
        },
        "progress": progress,
    }