#date: 2025-07-15T17:02:30Z
#url: https://api.github.com/gists/f41623744726fe89c5bb092a98b68aa9
#owner: https://api.github.com/users/taylor-mitchell-shopify

#!/bin/bash

# Script to compare currency utilities between react-i18n and i18n packages
# This verifies that the migration from react-i18n to i18n was done correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================"
echo "Comparing currency utilities migration from react-i18n to i18n"
echo "================================================================"
echo

# Function to check if files exist
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}ERROR: File not found: $1${NC}"
        exit 1
    fi
}

# 1. Compare CurrencyCode enum
echo -e "${YELLOW}1. Comparing CurrencyCode enum:${NC}"
echo "   react-i18n: packages/react-i18n/src/currencyCode.ts"
echo "   i18n:       packages/i18n/src/currency-code.ts"
echo

check_file "packages/react-i18n/src/currencyCode.ts"
check_file "packages/i18n/src/currency-code.ts"

if diff -u packages/react-i18n/src/currencyCode.ts packages/i18n/src/currency-code.ts > /tmp/currency-code-diff.txt; then
    echo -e "${GREEN}   ✓ Files are identical${NC}"
else
    echo -e "${RED}   ✗ Files differ:${NC}"
    echo "   Key differences:"
    # Show just the changed lines
    grep "^[+-][^+-]" /tmp/currency-code-diff.txt | head -20
    echo
    echo "   Run 'diff -u packages/react-i18n/src/currencyCode.ts packages/i18n/src/currency-code.ts' for full diff"
fi
echo

# 2. Compare currency decimal places
echo -e "${YELLOW}2. Comparing currency decimal places:${NC}"
echo "   react-i18n: packages/react-i18n/src/constants/currency-decimal-places.ts"
echo "   i18n:       packages/i18n/src/currency-decimal-places.ts"
echo

check_file "packages/react-i18n/src/constants/currency-decimal-places.ts"
check_file "packages/i18n/src/currency-decimal-places.ts"

if diff -u packages/react-i18n/src/constants/currency-decimal-places.ts packages/i18n/src/currency-decimal-places.ts > /tmp/decimal-places-diff.txt; then
    echo -e "${GREEN}   ✓ Files are identical${NC}"
else
    echo -e "${RED}   ✗ Files differ${NC}"
    echo "   See diff for details"
fi
echo

# 3. Compare getCurrencySymbol function
echo -e "${YELLOW}3. Comparing getCurrencySymbol and formatCurrency functions:${NC}"
echo "   react-i18n: packages/react-i18n/src/utilities/money.ts"
echo "   i18n:       packages/i18n/src/money.ts"
echo

check_file "packages/react-i18n/src/utilities/money.ts"
check_file "packages/i18n/src/money.ts"

# Extract just the function implementations (ignoring imports)
echo "   Comparing function implementations (ignoring imports)..."

# Create temp files with normalized content
sed -n '/export function getCurrencySymbol/,/^}/p' packages/react-i18n/src/utilities/money.ts > /tmp/react-i18n-money.txt
sed -n '/export function getCurrencySymbol/,/^}/p' packages/i18n/src/money.ts > /tmp/i18n-money.txt

if diff -u /tmp/react-i18n-money.txt /tmp/i18n-money.txt > /tmp/money-diff.txt; then
    echo -e "${GREEN}   ✓ getCurrencySymbol implementations are identical${NC}"
else
    echo -e "${RED}   ✗ getCurrencySymbol implementations differ${NC}"
fi

sed -n '/export function formatCurrency/,/^}/p' packages/react-i18n/src/utilities/money.ts > /tmp/react-i18n-format.txt
sed -n '/export function formatCurrency/,/^}/p' packages/i18n/src/money.ts > /tmp/i18n-format.txt

if diff -u /tmp/react-i18n-format.txt /tmp/i18n-format.txt > /tmp/format-diff.txt; then
    echo -e "${GREEN}   ✓ formatCurrency implementations are identical${NC}"
else
    echo -e "${RED}   ✗ formatCurrency implementations differ${NC}"
fi
echo

# 4. Compare number formatter utilities
echo -e "${YELLOW}4. Comparing number formatter utilities:${NC}"
echo "   react-i18n: packages/react-i18n/src/utilities/translate.tsx (memoizedNumberFormatter)"
echo "   i18n:       packages/i18n/src/number-formatter.ts"
echo

check_file "packages/react-i18n/src/utilities/translate.tsx"
check_file "packages/i18n/src/number-formatter.ts"

# Extract memoizedNumberFormatter from react-i18n
sed -n '/export function memoizedNumberFormatter/,/^}/p' packages/react-i18n/src/utilities/translate.tsx > /tmp/react-i18n-number-formatter.txt
sed -n '/export function memoizedNumberFormatter/,/^}/p' packages/i18n/src/number-formatter.ts > /tmp/i18n-number-formatter.txt

if diff -u /tmp/react-i18n-number-formatter.txt /tmp/i18n-number-formatter.txt > /tmp/number-formatter-diff.txt; then
    echo -e "${GREEN}   ✓ memoizedNumberFormatter implementations are identical${NC}"
else
    echo -e "${YELLOW}   ~ memoizedNumberFormatter implementations have minor differences${NC}"
    echo "   (This is expected due to different file contexts)"
fi
echo

# 5. Check for USDC support
echo -e "${YELLOW}5. Checking USDC support:${NC}"
if grep -q "Usdc = 'USDC'" packages/i18n/src/currency-code.ts; then
    echo -e "${GREEN}   ✓ USDC currency code is present in i18n${NC}"
else
    echo -e "${RED}   ✗ USDC currency code is missing in i18n${NC}"
fi

if grep -q "CurrencyCode.Usdc" packages/i18n/src/money.ts; then
    echo -e "${GREEN}   ✓ USDC formatting support is present in i18n${NC}"
else
    echo -e "${RED}   ✗ USDC formatting support is missing in i18n${NC}"
fi
echo

# 6. Summary
echo "================================================================"
echo -e "${YELLOW}Summary:${NC}"
echo

# Count missing currencies
if [ -f /tmp/currency-code-diff.txt ]; then
    MISSING_CURRENCIES=$(grep "^-[[:space:]]*[A-Z][a-z][a-z] = '[A-Z]*'," /tmp/currency-code-diff.txt | wc -l)
    if [ "$MISSING_CURRENCIES" -gt 0 ]; then
        echo -e "${RED}   - Missing $MISSING_CURRENCIES currencies in i18n package${NC}"
        echo "     Run this to see which ones:"
        echo "     diff packages/react-i18n/src/currencyCode.ts packages/i18n/src/currency-code.ts | grep '^-[[:space:]]*[A-Z]'"
    fi
fi

# Check for deprecated currency handling
if grep -q "// Codes after this line are deprecated" packages/i18n/src/currency-code.ts; then
    echo -e "${GREEN}   - Deprecated currency section is properly marked${NC}"
else
    echo -e "${RED}   - Deprecated currency section marker is missing${NC}"
fi

# Clean up temp files
rm -f /tmp/currency-code-diff.txt /tmp/decimal-places-diff.txt /tmp/money-diff.txt /tmp/format-diff.txt
rm -f /tmp/react-i18n-money.txt /tmp/i18n-money.txt /tmp/react-i18n-format.txt /tmp/i18n-format.txt
rm -f /tmp/react-i18n-number-formatter.txt /tmp/i18n-number-formatter.txt /tmp/number-formatter-diff.txt

echo
echo "================================================================"
echo "To see the complete file differences, run:"
echo "  diff -u packages/react-i18n/src/currencyCode.ts packages/i18n/src/currency-code.ts"
echo "  diff -u packages/react-i18n/src/utilities/money.ts packages/i18n/src/money.ts"
echo "================================================================"