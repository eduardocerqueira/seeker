//date: 2024-11-12T16:49:19Z
//url: https://api.github.com/gists/90142d3c9959b0342c8d798974ac4c78
//owner: https://api.github.com/users/trikitrok

package com.argentrose;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class ArgentRoseStoreTest {

	public static final int MIN_QUALITY = 0;
	public static final int MAX_QUALITY = 50;
	public static final int SELLIN_LAST_DAY = 0;
	public static final int EXPIRED = -1;
	private ArgentRoseStore store;

	@BeforeEach
	void setUp() {
		store = new ArgentRoseStore();
	}

	@Test
	public void regular_product_decrease_quality_by_two() {
		final Product regular = createRegular(1, 10);
		store.add(regular);

		store.update();

        	assertEquals(createRegular(SELLIN_LAST_DAY, 8), regular);
	}

	@Test
	public void expired_regular_product_decrease_quality_twice_as_fast() {
		final Product regular = createRegular(SELLIN_LAST_DAY, 10);
		store.add(regular);

		store.update();

		assertEquals(createRegular(EXPIRED, 6), regular);
	}

	@Test
	public void lanzarote_wine_increase_quality_by_one() {
		final Product lanzaroteWine = createLanzaroteWine(1, 10);
		store.add(lanzaroteWine);

		store.update();

		assertEquals(createLanzaroteWine(SELLIN_LAST_DAY, 12), lanzaroteWine);
	}

	@Test
	public void expired_lanzarote_wine_increase_quality_twice_as_fast() {
		final Product lanzaroteWine = createLanzaroteWine(SELLIN_LAST_DAY, 10);
		store.add(lanzaroteWine);

		store.update();

		assertEquals(createLanzaroteWine(EXPIRED, 14), lanzaroteWine);
	}

	@Test
	public void theatre_passes_increase_quality_by_one_when_sellIn_is_far_away() {
		final Product theatrePasses = createTheatrePasses(6, 12);
		store.add(theatrePasses);

		store.update();

		assertEquals(createTheatrePasses(5, 13), theatrePasses);
	}

	@Test
	public void theatre_passes_increase_quality_by_three_when_sellIn_is_near() {
		final Product theatrePasses = createTheatrePasses(3, 12);
		store.add(theatrePasses);

		store.update();

		assertEquals(createTheatrePasses(2, 15), theatrePasses);
	}

	@Test
	public void expired_theatre_passes_drop_quality_to_zero() {
		final Product theatrePasses = createTheatrePasses(SELLIN_LAST_DAY, 5);
		store.add(theatrePasses);

		store.update();

		assertEquals(createTheatrePasses(EXPIRED, MIN_QUALITY), theatrePasses);
	}

	@Test
	public void regular_product_quality_is_not_below_the_min_quality() {
		final Product regular = createRegular(2, 1);
		store.add(regular);

		store.update();

		assertEquals(createRegular(1, MIN_QUALITY), regular);
	}

	@Test
	public void expired_regular_product_quality_is_not_below_the_min_quality() {
		final Product regular = createRegular(SELLIN_LAST_DAY, 3);
		store.add(regular);

		store.update();

		assertEquals(createRegular(EXPIRED, MIN_QUALITY), regular);
	}

	@Test
	public void lanzarote_wine_quality_never_increases_over_the_maximum_quality() {
		final Product lanzaroteWine = createLanzaroteWine(3, MAX_QUALITY);
		store.add(lanzaroteWine);

		store.update();

		assertEquals(createLanzaroteWine(2, MAX_QUALITY), lanzaroteWine);
	}

	@Test
	public void expired_lanzarote_wine_quality_never_increases_over_the_maximum_quality() {
		final Product lanzaroteWine = createLanzaroteWine(SELLIN_LAST_DAY, MAX_QUALITY);
		store.add(lanzaroteWine);

		store.update();

		assertEquals(createLanzaroteWine(EXPIRED, MAX_QUALITY), lanzaroteWine);
	}

	@Test
	public void theatre_pass_quality_when_sellIn_is_far_away_never_increases_over_the_maximum_quality() {
		final Product theatrePasses = createTheatrePasses(2, MAX_QUALITY);
		store.add(theatrePasses);

		store.update();

		assertEquals(createTheatrePasses(1, MAX_QUALITY), theatrePasses);
	}

	@Test
	public void theatre_pass_quality_when_sellIn_is_near_never_increases_over_the_maximum_quality() {
		final Product theatrePasses = createTheatrePasses(2, 48);
		store.add(theatrePasses);

		store.update();

		assertEquals(createTheatrePasses(1, MAX_QUALITY), theatrePasses);
	}

	@Test
	public void update_multiple_products() {
		final Product regular1 = createRegular(10, 4);
		final Product regular2 = createRegular(3, 5);
		store.add(regular1, regular2);

		store.update();

		assertEquals(createRegular(9, 2), regular1);
		assertEquals(createRegular(2, 3), regular2);
	}

	private static Product createRegular(int sellIn, int quality) {
		return new Product("Regular", sellIn, quality);
	}

	private static Product createLanzaroteWine(int sellIn, int quality) {
		return new Product("Lanzarote Wine", sellIn, quality);
	}

	private static Product createTheatrePasses(int sellIn, int quality) {
		return new Product("Theatre Passes", sellIn, quality);
	}
}