//date: 2021-09-24T17:05:03Z
//url: https://api.github.com/gists/a787730811a539e605652a213f1a8c01
//owner: https://api.github.com/users/PhaseRush

import java.util.concurrent.ThreadLocalRandom;

class Scratch {
    static final int HARD_PITY = 90;
    static final int SOFT_PITY = 75;
    static final double PRE_SOFT_PROB = 0.006;
    static final double POST_SOFT_PROB = 0.324;

    static final int NUM_TRIALS = 1_000_000;

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        long regularTotal = 0;
        long eventTotal = 0;
        long promoTotal = 0;

        int currRegular, currEvent, currPromo;
        for (var i = 0; i < NUM_TRIALS; i++) {
            currRegular = regularWish();
            regularTotal += currRegular;

            currEvent = nonPromoEventWish();
            eventTotal += currEvent;

            currPromo = promoEventWish();
            promoTotal += currPromo;
        }

        System.out.printf("%-15s%10.4f\n", "Trials", (float)NUM_TRIALS);
        System.out.printf("%-15s%10.4f\n", "Avg. Regular", (float) regularTotal / NUM_TRIALS);
        System.out.printf("%-15s%10.4f\n", "Avg. Event", (float) eventTotal / NUM_TRIALS);
        System.out.printf("%-15s%10.4f\n", "Avg. Promo", (float) promoTotal / NUM_TRIALS);
        System.out.printf("%-15s%10.4f\n", "Time (ms)",  (float)(System.currentTimeMillis() - start));
    }

    static boolean flipCoin() {
        return ThreadLocalRandom.current().nextBoolean();
    }

    static boolean selected(double prob) {
        return ThreadLocalRandom.current().nextFloat() < prob &&
                ThreadLocalRandom.current().nextInt(5) == 1; // correct 5 star (1 out of 5)
    }

    static int regularWish() {
        int wishes = 0;
        int pity = 0;
        while (true) {
            wishes++;
            if (pity == HARD_PITY) { // hard pity, 50/50
                pity = 0;
                if (selected(0.5)) return wishes;
            } else if (pity > SOFT_PITY) {
                pity++;
                if (ThreadLocalRandom.current().nextFloat() < POST_SOFT_PROB) {
                    pity = 0;
                    if (selected(0.5)) return wishes;
                }
            } else {
                pity++;
                if (ThreadLocalRandom.current().nextFloat() < PRE_SOFT_PROB) {
                    pity = 0;
                    if (selected(0.5)) return wishes;
                }
            }
        }
    }

    static int nonPromoEventWish() {
        int wishes = 0;
        int pity = 0;
        boolean guarantee = false;
        while (true) {
            wishes++;
            if (pity == HARD_PITY) {
                pity = 0;
                if (!guarantee) {
                    if (flipCoin()) {
                        if (ThreadLocalRandom.current().nextInt(5) == 1) return wishes;
                        guarantee = true;
                    }
                } else {
                    guarantee = false; // sad
                }
            } else if (pity > SOFT_PITY) {
                pity++;
                if (ThreadLocalRandom.current().nextFloat() < POST_SOFT_PROB) {
                    pity = 0;
                    if (!guarantee) {
                        if (flipCoin()) {
                            if (ThreadLocalRandom.current().nextInt(5) == 1) return wishes;
                            guarantee = true;
                        }
                    } else {
                        guarantee = false;
                    }
                }
            } else {
                pity++;
                if (ThreadLocalRandom.current().nextFloat() < PRE_SOFT_PROB) {
                    pity = 0;
                    if (!guarantee) {
                        if (flipCoin()) {
                            if (ThreadLocalRandom.current().nextInt(5) == 1) return wishes;
                            guarantee = true;
                        }
                    } else {
                        guarantee = false;
                    }
                }
            }
        }
    }

    static int promoEventWish() {
        int wishes = 0;
        int pity = 0;
        boolean guarantee = false;
        while (true) {
            wishes++;
            if (pity == HARD_PITY) {
                pity = 0;
                if (guarantee) return wishes;
                else if (flipCoin()) guarantee = true;
                else return wishes;
            } else if (pity > SOFT_PITY) {
                pity++;
                if (ThreadLocalRandom.current().nextFloat() < POST_SOFT_PROB) {
                    pity = 0;
                    if (guarantee) return wishes;
                    else if (flipCoin()) guarantee = true;
                    else return wishes;
                }
            } else {
                pity++;
                if (ThreadLocalRandom.current().nextFloat() < PRE_SOFT_PROB) {
                    pity = 0;
                    if (guarantee) return wishes;
                    else if (flipCoin()) guarantee = true;
                    else return wishes;
                }
            }
        }
    }
}