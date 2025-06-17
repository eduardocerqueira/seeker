//date: 2025-06-17T17:04:28Z
//url: https://api.github.com/gists/eaee40972919ff27fd885b27a3af654c
//owner: https://api.github.com/users/blzjns

public boolean evenlySpaced(int a, int b, int c) {
  int sm = Math.min(Math.min(a, b), c);
  int lg = Math.max(Math.max(a, b), c);
  //int md = (a+b+c) - (Math.min(Math.min(a,b),c) + Math.max(Math.max(a,b),c));
  int md = (a+b+c)-(sm+lg);
  
  
  return md-sm == lg-md;
}
