#date: 2024-05-14T16:54:50Z
#url: https://api.github.com/gists/45df675227edb7f9055359c738537a7c
#owner: https://api.github.com/users/garethperky

function generateValidChiNumbers(): string[] {
  const chiNumbers: string[] = [];

  for (let j = 1; j <= 28; j++) { // Limiting to 28 to avoid invalid dates
    for (let l = 0; l < 10; l++) { // Generate 10 different serial numbers
      const dobDay = j.toString().padStart(2, '0');
      const dobMonth = '01'; // January
      const dobYear = '00'; // Year 2000
      const serial = l.toString().padStart(3, '0');

      const baseNumber = dobDay + dobMonth + dobYear + serial;
      let total = 0;
      for (let m = 0; m < baseNumber.length; m++) {
        total += parseInt(baseNumber[baseNumber.length - 1 - m]) * (m + 2);
      }
      const quotient = Math.floor(total / 11);
      let calculatedCheckDigit = 11 * (quotient + 1) - total;
      calculatedCheckDigit = calculatedCheckDigit === 11 ? 0 : calculatedCheckDigit;

      if (calculatedCheckDigit !== 10) {
        chiNumbers.push(baseNumber + calculatedCheckDigit);
      }

      if (chiNumbers.length === 10) {
        return chiNumbers;
      }
    }
  }

  return chiNumbers;
}

console.log(generateValidChiNumbers());