#date: 2023-11-13T16:51:01Z
#url: https://api.github.com/gists/538087c53ae83a1ee3dd1726e3714ba1
#owner: https://api.github.com/users/SP5LMA

:>pdffiles.csv; for f in *pdf; do 
  p=$(pdfinfo "$f"|grep Pages |cut -f 12- -d ' ');
  fs=$(pdfinfo "$f"|grep 'File size'|cut -f 9 -d ' ');
  echo -e "$p\t$fs\t$f" >>pdffiles.csv;
done;