//date: 2023-05-16T17:04:15Z
//url: https://api.github.com/gists/ec3edb576e382b0f847fe07736145734
//owner: https://api.github.com/users/btinet

    
     public void addKennzeichen(String zeichen) {

        /* Version mit for each-Schleife
        *
        *  int pos = 0;
        *  for(String kfz : this.kennzeichen) {
        *      if(kfz == null) {
        *          this.kennzeichen[pos] = zeichen;
        *          break;
        *      }
        *      pos++;
        *  }
        */

         int pos;
         for(pos = 0; pos < this.kennzeichen.length; pos++) {
             if(this.kennzeichen[pos] == null) {
                 this.kennzeichen[pos] = zeichen;
                 break;
             }
         }
        
        if(this.kennzeichen.length == pos) {
            int i;
            for (i = 0; i < 4; i++) {
                this.kennzeichen[i] = this.kennzeichen[i+1];           
            }
            this.kennzeichen[i] = zeichen;
        }
    }
