//date: 2022-05-13T17:15:11Z
//url: https://api.github.com/gists/8a34876c69009d9caf9329c7ddc03cc9
//owner: https://api.github.com/users/gastonfournier

import javax.money.MonetaryAmount;
import javax.money.format.AmountFormatQueryBuilder;
import javax.money.format.MonetaryAmountFormat;
import javax.money.format.MonetaryFormats;

import org.javamoney.moneta.Money;
import org.javamoney.moneta.format.CurrencyStyle;
import org.junit.jupiter.api.Test;

import java.util.Locale;

public class MonetaEURFormatTest {

    private static final Locale EUR_PREFERRED_LOCALE = Locale.forLanguageTag("es-ES");

    static String format(MonetaryAmount money, Locale locale) {
        MonetaryAmountFormat formatter = MonetaryFormats.getAmountFormat(
            AmountFormatQueryBuilder.of(locale)
                .set(CurrencyStyle.SYMBOL)
                .setLocale(EUR_PREFERRED_LOCALE)
                .build()
        );
        return formatter.format(money);
    }

    @Test
    void testFormat() {
        MonetaryAmount monetaryAmount = Money.of(1234, "EUR");
        String esFormat = format(monetaryAmount, Locale.forLanguageTag("es-ES"));
        String usFormat = format(monetaryAmount, Locale.forLanguageTag("en-US"));
        System.out.println(esFormat); // outputs 1.234,00 €
        System.out.println(usFormat); // outputs 1.234,00 €
    }
}
