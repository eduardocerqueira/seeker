//date: 2022-08-25T15:31:59Z
//url: https://api.github.com/gists/12cfca651fe42e80c1c829cb32dabdcc
//owner: https://api.github.com/users/thiagonuneszup

   public double exponenciar(int base, int expoente) {
        if (expoente == 0) {
            return 1;
        }
        if (expoente == -1) {
            return 1.0 / base;
        }

        if (expoente == 1) {
            return (double) base;
        }

        if (base == 1) {
            return 1;
        }

        if (base == -1) {
            return expoente % 2 == 0 ? 1 : -1;
        }

        return exponenciar((double) base, (double) expoente, 1);
    }


    double exponenciar(double base, double expoente, double resultado) {

        if (expoente < 0) {
            double baseInvertida = 1 / base;
            double expoenteComSinalInvertido = expoente * -1;
            return exponenciar(baseInvertida, expoenteComSinalInvertido - 1, resultado * baseInvertida * baseInvertida);
        }

        if (base < 0 && expoente % 2 == 0 && resultado == 1) {
            double baseComSinalInvertido = base * -1;
            return exponenciar(baseComSinalInvertido, expoente - 1, resultado * baseComSinalInvertido * baseComSinalInvertido);
        }

        if (resultado == 1) {
            return exponenciar(base, expoente - 1, resultado * base * base);
        }

        if (expoente == 1) {
            return resultado;
        }

        return exponenciar(base, expoente - 1, resultado * base);
    }