//date: 2025-02-13T17:00:42Z
//url: https://api.github.com/gists/529962bb9910a4ed8b3a5060c1107c05
//owner: https://api.github.com/users/rrogovski

// Crivo de Eratóstenes
// https://pt.khanacademy.org/computing/computer-science/cryptography/comp-number-theory/v/sieve-of-eratosthenes-prime-adventure-part-4
// https://pt.khanacademy.org/computing/computer-science/cryptography/comp-number-theory/v/trial-division-primality-test-using-a-sieve-prime-adventure-part-5
// Execução de teste dos primos até 10000000
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PrimosParalelo {

    public static boolean isPrimo(int numero) {
        if (numero <= 1) return false;
        if (numero == 2) return true;
        if (numero % 2 == 0) return false;

        double raiz = Math.sqrt(numero);

        for (int i = 3; i <= raiz; i += 2) {
            if (numero % i == 0) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int start = 1;
        int end;
        List<Integer> primos;

        System.out.println("Olá 🖖");
        System.out.println(String.format("Verificar os números primos no intervalo de (%d) até (número informado):", start));
        System.out.println("🔴 Dependendo do intervalo informado, a aplicação pode gerar sobre carga de processamento e eventualmente travar!");

        while (true) {
            System.out.print("✏️ ");
            if (!sc.hasNextInt()) {
                System.out.println("🚨 Valor inválido. Tente novamente!");
                sc.nextLine();
            } else {
                end = sc.nextInt();

                if (end < 0) {
                    System.out.println("🚨 Valor inválido. Tente novamente!");
                } else {
                    break;
                }
            }
        }

        Instant startTime = Instant.now();

        primos = IntStream.rangeClosed(start, end)
                          .parallel()
                          .filter(PrimosParalelo::isPrimo)
                          .boxed()
                          .collect(Collectors.toList());

        String primosString = primos.toString();

        System.out.println(
            String.format(
                "Primos de 1 até %d => %s",
                end,
                primosString.substring(1, primosString.length() - 1)
            )
        );

        System.out.println(String.format("Total de primos => %d", primos.size()));

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        System.out.println(
            String.format(
                "--- Tempo de execução | início: %s | fim: %s | total: %f segundos ---",
                formatter.format(LocalDateTime.ofInstant(startTime, ZoneId.systemDefault())),
                formatter.format(LocalDateTime.now()),
                (double) (Instant.now().toEpochMilli() - startTime.toEpochMilli()) / 1000
            )
        );
    }
}