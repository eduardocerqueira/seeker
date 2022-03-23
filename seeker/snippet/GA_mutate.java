//date: 2022-03-23T16:53:48Z
//url: https://api.github.com/gists/f0b1de9864ff060a012e4b02a5856329
//owner: https://api.github.com/users/mananai

private List<Genome> mutate(List<Genome> population2) {
  Random random = new Random();
  for ( int i = 0 ; i < population2.size() ; i++) {
    if (random.nextDouble() <= this.mutationRate ) {
      population2.set(i, population2.get(i).mutate());
    }
  }
  return population2;		
}