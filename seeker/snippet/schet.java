//date: 2024-02-09T16:43:51Z
//url: https://api.github.com/gists/c983f70ebff86087a514d98b3f3a5104
//owner: https://api.github.com/users/aretinsky

public class Schet {
    public LinkedList<Double> history;

    private final SomeCurrencyExchangeService service;
  
    public Schet(LinkedList<Double> initial) {
        this.history = initial;
        this.service = new SomeCurrencyExchangeService();
    }

    public void addMoney(Double money) {
        history.add(money);
     }

    public boolean equals(Object o) {
        if (this == o) {
           return true;
        }
        Schet other = (Schet) o;
        return other.history.equals(history);
    }

    Double transferToTheCurrency(String currency) {
        AtomicReference<Double> money = new AtomicReference<>(0.0);
        try {
            history.forEach(it -> {
                Double converted = service.getValueForCurrency(CurrencyEnum.valueOf(currency), it);
                money.accumulateAndGet(converted, Double::sum);
            });

            return money.get();
        } catch (Throwable t) {
            return 0.0;
        }
    }
}