//date: 2025-10-06T16:44:48Z
//url: https://api.github.com/gists/0edc8dba47fc985d25baef55550b184b
//owner: https://api.github.com/users/rgbk21

public void setTimeout(Duration duration) {
  // Check for negative Durations at API boundaries.
  this.duration = Durations.checkNotNegative(duration);
  // If you want to avoid Duration.ZERO as well.
  this.duration = Duration.checkPositive(duration);
}