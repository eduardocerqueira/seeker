//date: 2022-05-20T17:03:20Z
//url: https://api.github.com/gists/2069a916964614d399ddeff4b529cfd9
//owner: https://api.github.com/users/GrandJah

import static java.util.Objects.isNull;

import java.util.*;

/*
 * В одной далекой стране правил крайне сумасбродный король, который больше всего на свете любил власть.
 * Ему подчинялось множество людей, но вот незадача, у его подчиненных тоже были свои слуги.
 * Король обезумел от мысли, что какой-нибудь дворянин или даже зажиточный холоп может иметь больше слуг, чем он сам.
 * И приказал всем людям на бумаге через запятую написать свое имя и имена своих прямых подчиненных.
 *
 * По результатам опроса король получил огромный список из имен (see "pollResults")
 *
 * У короля разболелась голова. Что с этими данными делать, король не знал и делегировал задачу невезучему слуге.

 * Помогите слуге правильно составить иерархию и подготовить  отчет для короля следующим образом:
 *
 * король
       ...
 *     дворянин Кузькин
           жена Кузькина
 *         управляющий Семен Семеныч
               доярка Нюра
 *             крестьянин Федя
 *         ...
 *     секретарь короля
 *         зажиточный холоп
 *         ...
 *     ...
 *
 * Помните:
 *  1. Те, у кого нет подчиненных, просто написали свое имя.
 *  2. Те, кого никто не указал как слугу, подчиняются напрямую королю (ну, пускай бедный король так думает).
 *  3. Итоговый список должен быть отсортирован в алфавитном порядке на каждом уровне иерархии.
 *
 * Ответ присылайте ссылкой на опубликованный приватный Gist.
 * */

public class LazyKing {
    private static final List<String> pollResults = List.of("служанка Аня",
        "управляющий Семен Семеныч: крестьянин Федя, доярка Нюра",
        "дворянин Кузькин: управляющий Семен Семеныч, жена Кузькина, экономка Лидия Федоровна",
        "экономка Лидия Федоровна: дворник Гена, служанка Аня", "доярка Нюра",
        "кот Василий: человеческая особь Катя", "дворник Гена: посыльный Тошка", "киллер Гена",
        "зажиточный холоп: крестьянка Таня", "секретарь короля: зажиточный холоп, шпион Т",
        "шпион Т: кучер Д", "посыльный Тошка: кот Василий", "аристократ Клаус",
        "просветленный Антон");

    public static void main(String... args) {
        UnluckyVassal unluckyVassal = new UnluckyVassal();
        unluckyVassal.printReportForKing(pollResults);
    }
}

class UnluckyVassal {

    public void printReportForKing(List<String> pollResults) {
        System.out.println(new Person("король", pollResults).getReport());
    }
}

class Person {
    private final String name;

    private final List<Person> vassals = new ArrayList<>();

    public Person(String name) {
        this.name = name;
    }

    public Person(String PersonName, List<String> vassalsStrings) {
        PersonSet vassalSet = new PersonSet();
        vassalSet.addAll(vassalsStrings);
        name = PersonName;
        vassals.addAll(vassalSet.getPersons());
    }

    public String getName() {
        return name;
    }

    public List<Person> getVassals() {
        return vassals;
    }

    public String getReport() {
        return getFormatString(0);
    }

    private String getFormatString(int tabs) {
        char[] cs = new char[tabs * 4];
        Arrays.fill(cs, ' ');
        StringBuffer sb = new StringBuffer();
        sb.append(cs).append(name).append(System.lineSeparator());
        if (!vassals.isEmpty()) {
            vassals.stream()
                .sorted(Comparator.comparing(Person::getName))
                .forEach(vassals -> sb.append(vassals.getFormatString(tabs + 1)));
        }
        return sb.toString();
    }

    private static class PersonSet {
        private final Map<String, Person> persons = new HashMap<>();

        Collection<Person> getPersons() {
            return persons.values();
        }

        void addAll(List<String> reports) {
            reports.forEach(report -> {
                String[] splitReport = report.split(":");
                Person vassal = getPerson(splitReport[0].trim());
                if (splitReport.length > 1) {
                    Arrays.stream(splitReport[1].trim().split(","))
                        .forEach(
                            vassalName -> vassal.getVassals().add(getPerson(vassalName.trim())));
                }
            });
            clearForeignVassal();
        }

        private void clearForeignVassal() {
            persons.values()
                .stream()
                .flatMap(person -> person.getVassals().stream())
                .map(Person::getName)
                .toList()
                .forEach(persons::remove);
        }

        private Person getPerson(String name) {
            Person person = persons.get(name);
            if (isNull(person)) {
                person = new Person(name);
                persons.put(name, person);
            }
            return person;
        }
    }
}
