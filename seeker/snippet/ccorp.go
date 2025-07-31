//date: 2025-07-31T17:00:08Z
//url: https://api.github.com/gists/1a65a9e155b5e49ae75b879c8951bc76
//owner: https://api.github.com/users/szegedim

package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// This document is Licensed under Creative Commons CC0.
// To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights
// to this document to the public domain worldwide.
// This document is distributed without any warranty.
// You should have received a copy of the CC0 Public Domain Dedication along with this document.
// If not, see https://creativecommons.org/publicdomain/zero/1.0/legalcode.

// Experiment: This code is an experiment to create a financial management system to generate financial statements, returns, and other financial reports.
// Resources: 12 hours labor, LLM subscription

type Category string

const (
	FixedAssets                  Category = "fixed_assets"
	Cash                         Category = "cash"
	WorkingAssets                Category = "working_assets"
	Liabilities                  Category = "liabilities"
	CurrentLiabilities           Category = "current_liabilities"
	NonCurrentLiabilities        Category = "non_current_liabilities"
	AccountsReceivable           Category = "accounts_receivable"
	CapitalInvestments           Category = "capital_investments"
	OperatingExpenses            Category = "operating_expenses"
	NonCashOperatingExpenses     Category = "non_cash_operating_expenses" // New category
	ShareholdersEquity           Category = "shareholders_equity"
	Interest                     Category = "interest"
	Dividend                     Category = "dividend"
	EquityIssuance               Category = "equity_issuance"
	EmployeeReimbursableExpenses Category = "employee_reimbursable_expenses"
)

type Entry struct {
	ID          int
	Description string
	Category    Category
	Units       float64
	Value       float64
	Date        time.Time
	IsBeginning bool
	IsEnding    bool
	IsSale      bool
}

type Financials struct {
	Entries                          []Entry
	WorkingAssetStartingValue        float64
	WorkingAssetEndingValue          float64
	TotalWorkingAssetPurchases       float64
	TotalSales                       float64
	CostOfGoodsSold                  float64
	GrossIncome                      float64
	FixedAssetDepreciation           float64
	FixedAssetStartingValue          float64
	FixedAssetEndingValue            float64
	FixedAssetPurchases              float64
	EBITDA                           float64
	EBIT                             float64
	EBT                              float64
	DividendsPaid                    float64
	Taxes                            float64
	NetIncome                        float64
	CashFlowsFromOperations          float64
	ActualCashFlow                   float64
	NetIncreaseInCash                float64
	CashFlowDiscrepancy              float64
	CashFlowsFromInvestingActivities float64
	CashFlowsFromFinancingActivities float64
	BalanceSheetAssets               float64
	BalanceSheetLiabilities          float64
	BalanceSheetCapital              float64
	BalanceSheetTotal                float64
	StartingCashInventory            float64
	EndingCashInventory              float64
	OperatingExpenses                float64
	NonCashOperatingExpenses         float64 // New field
	Interest                         float64
	CurrentLiabilities               float64
	NonCurrentLiabilities            float64
	ShareholderEquityOne             float64
	ShareholderEquityTwo             float64
	IssuanceOfCommonStock            float64
	PreviousYearRetainedEarnings     float64
	RetainedEarnings                 float64
	EmployeeReimbursableExpenses     float64
}

func NewFinancials() *Financials {
	return &Financials{
		Entries: make([]Entry, 0, 10000),
	}
}

func (f *Financials) AddEntry(e Entry) {
	if len(f.Entries) < 10000 {
		f.Entries = append(f.Entries, e)
	}
}

func RoundToPrecision001(value float64) float64 {
	return math.Round(value*100) / 100
}

func (f *Financials) CalculateAggregates() {
	var totalUnitsPurchased float64
	var totalPurchaseCost float64
	var actualEndingCash float64

	for _, entry := range f.Entries {
		if entry.Category == Cash && entry.IsBeginning {
			f.StartingCashInventory = RoundToPrecision001(f.StartingCashInventory + entry.Value)
		}
		if entry.Category == Cash && entry.IsEnding {
			actualEndingCash = RoundToPrecision001(actualEndingCash + entry.Value)
		}
		if entry.Category == FixedAssets && entry.IsBeginning {
			f.FixedAssetStartingValue = RoundToPrecision001(f.FixedAssetStartingValue + entry.Value)
		}
		if entry.Category == WorkingAssets && entry.IsBeginning {
			f.WorkingAssetStartingValue = RoundToPrecision001(f.WorkingAssetStartingValue + entry.Value)
		}
	}

	for _, entry := range f.Entries {
		switch entry.Category {
		case WorkingAssets:
			if entry.IsBeginning {
			} else if entry.IsEnding {
				f.WorkingAssetEndingValue = RoundToPrecision001(f.WorkingAssetEndingValue + entry.Value)
			} else if entry.IsSale {
				f.TotalSales = RoundToPrecision001(f.TotalSales + entry.Value)
			} else {
				f.TotalWorkingAssetPurchases = RoundToPrecision001(f.TotalWorkingAssetPurchases + entry.Value)
				totalUnitsPurchased += entry.Units
				totalPurchaseCost = RoundToPrecision001(totalPurchaseCost + entry.Value)
			}
		case FixedAssets:
			if entry.IsBeginning {
			} else if entry.IsEnding {
				f.FixedAssetEndingValue = RoundToPrecision001(f.FixedAssetEndingValue + entry.Value)
			} else {
				f.FixedAssetPurchases = RoundToPrecision001(f.FixedAssetPurchases + entry.Value)
			}
		case OperatingExpenses:
			f.OperatingExpenses = RoundToPrecision001(f.OperatingExpenses + entry.Value)
		case NonCashOperatingExpenses:
			f.NonCashOperatingExpenses = RoundToPrecision001(f.NonCashOperatingExpenses + entry.Value)
		case Interest:
			f.Interest = RoundToPrecision001(f.Interest + entry.Value)
		case Dividend:
			f.DividendsPaid = RoundToPrecision001(f.DividendsPaid + entry.Value)
		case Liabilities:
			f.PreviousYearRetainedEarnings = RoundToPrecision001(f.PreviousYearRetainedEarnings + entry.Value)
		case CurrentLiabilities:
			f.CurrentLiabilities = RoundToPrecision001(f.CurrentLiabilities + entry.Value)
		case NonCurrentLiabilities:
			f.NonCurrentLiabilities = RoundToPrecision001(f.NonCurrentLiabilities + entry.Value)
		case CapitalInvestments:
			f.BalanceSheetCapital = RoundToPrecision001(f.BalanceSheetCapital + entry.Value)
		case AccountsReceivable:
			f.BalanceSheetAssets = RoundToPrecision001(f.BalanceSheetAssets + entry.Value)
		case ShareholdersEquity:
			if strings.Contains(strings.ToLower(entry.Description), "shareholder equity one") {
				f.ShareholderEquityOne = RoundToPrecision001(f.ShareholderEquityOne + entry.Value)
			} else if strings.Contains(strings.ToLower(entry.Description), "shareholder equity two") {
				f.ShareholderEquityTwo = RoundToPrecision001(f.ShareholderEquityTwo + entry.Value)
			}
		case EquityIssuance:
			f.IssuanceOfCommonStock = RoundToPrecision001(f.IssuanceOfCommonStock + entry.Value)
			f.CashFlowsFromFinancingActivities = RoundToPrecision001(f.CashFlowsFromFinancingActivities + entry.Value)
		case EmployeeReimbursableExpenses:
			f.EmployeeReimbursableExpenses = RoundToPrecision001(f.EmployeeReimbursableExpenses + entry.Value)
		}
	}

	f.CostOfGoodsSold = RoundToPrecision001(f.WorkingAssetStartingValue + f.TotalWorkingAssetPurchases - f.WorkingAssetEndingValue)
	f.GrossIncome = RoundToPrecision001(f.TotalSales - f.CostOfGoodsSold)
	f.EBITDA = f.GrossIncome
	if f.OperatingExpenses != 0 || f.NonCashOperatingExpenses != 0 {
		f.EBITDA = RoundToPrecision001(f.GrossIncome - f.OperatingExpenses - f.NonCashOperatingExpenses)
	}
	f.FixedAssetDepreciation = RoundToPrecision001(f.FixedAssetStartingValue + f.FixedAssetPurchases - f.FixedAssetEndingValue)
	f.EBIT = RoundToPrecision001(f.EBITDA - f.FixedAssetDepreciation)
	f.EBT = RoundToPrecision001(f.EBIT - f.Interest)
	f.DividendsPaid = RoundToPrecision001(f.DividendsPaid)
	if f.EBT > 0 {
		f.Taxes = RoundToPrecision001(f.EBT * 0.20)
	} else {
		f.Taxes = 0.0
	}
	f.NetIncome = RoundToPrecision001(f.EBT - f.Taxes)
	changeInInventory := RoundToPrecision001(f.WorkingAssetEndingValue - f.WorkingAssetStartingValue)
	// Added NonCashOperatingExpenses to be added back to CashFlowsFromOperations
	f.CashFlowsFromOperations = RoundToPrecision001(f.NetIncome + f.FixedAssetDepreciation + f.NonCashOperatingExpenses - changeInInventory + f.Interest)
	f.CashFlowsFromInvestingActivities = RoundToPrecision001(-(f.FixedAssetEndingValue - f.FixedAssetStartingValue + f.FixedAssetDepreciation))
	f.CashFlowsFromFinancingActivities = RoundToPrecision001(f.CashFlowsFromFinancingActivities - f.Interest - f.DividendsPaid)
	f.NetIncreaseInCash = RoundToPrecision001(f.CashFlowsFromOperations + f.CashFlowsFromInvestingActivities + f.CashFlowsFromFinancingActivities)

	f.EndingCashInventory = actualEndingCash
	f.ActualCashFlow = RoundToPrecision001(f.EndingCashInventory - f.StartingCashInventory)
	f.CashFlowDiscrepancy = RoundToPrecision001(f.NetIncreaseInCash - f.ActualCashFlow)

	cashDifference := RoundToPrecision001(actualEndingCash - f.EndingCashInventory)
	if cashDifference != 0 {
		newEntry := Entry{
			ID:          len(f.Entries) + 1,
			Description: fmt.Sprintf("Adjustment for cash difference: %.2f", cashDifference),
			Category:    CurrentLiabilities,
			Units:       1.0,
			Value:       cashDifference,
			Date:        time.Date(2025, time.December, 31, 0, 0, 0, 0, time.UTC),
			IsBeginning: false,
			IsEnding:    false,
		}
		f.AddEntry(newEntry)
		f.CurrentLiabilities = RoundToPrecision001(f.CurrentLiabilities + cashDifference)
	}

	f.BalanceSheetAssets = RoundToPrecision001(f.BalanceSheetAssets + f.WorkingAssetEndingValue + f.FixedAssetEndingValue + f.EndingCashInventory)
	f.RetainedEarnings = RoundToPrecision001(f.PreviousYearRetainedEarnings + f.NetIncome - f.DividendsPaid)
	f.BalanceSheetLiabilities = RoundToPrecision001(f.ShareholderEquityOne + f.ShareholderEquityTwo + f.IssuanceOfCommonStock + f.RetainedEarnings + f.CurrentLiabilities + f.NonCurrentLiabilities + f.EmployeeReimbursableExpenses)
	f.BalanceSheetTotal = RoundToPrecision001(f.BalanceSheetAssets - f.BalanceSheetLiabilities)
}

func (f *Financials) PrintAggregates() {
	fmt.Printf("Working Asset Inventory Starting Value: $%.2f\n", f.WorkingAssetStartingValue)
	fmt.Printf("Working Asset Inventory Ending Value: $%.2f\n", f.WorkingAssetEndingValue)
	fmt.Printf("Total Working Asset Purchases: $%.2f\n", f.TotalWorkingAssetPurchases)
	fmt.Printf("Total Sales: $%.2f\n", f.TotalSales)
	fmt.Printf("Cost of Goods Sold: $%.2f\n", f.CostOfGoodsSold)
	fmt.Printf("Gross Income: $%.2f\n", f.GrossIncome)
	fmt.Printf("Operating Expenses: $%.2f\n", f.OperatingExpenses)
	fmt.Printf("Non-Cash Operating Expenses: $%.2f\n", f.NonCashOperatingExpenses) // New field
	fmt.Printf("EBITDA: $%.2f\n", f.EBITDA)
	fmt.Printf("Fixed Asset Inventory Starting Value: $%.2f\n", f.FixedAssetStartingValue)
	fmt.Printf("Fixed Asset Inventory Ending Value: $%.2f\n", f.FixedAssetEndingValue)
	fmt.Printf("Fixed Asset Purchases: $%.2f\n", f.FixedAssetPurchases)
	fmt.Printf("Fixed Asset Depreciation: $%.2f\n", f.FixedAssetDepreciation)
	fmt.Printf("EBIT: $%.2f\n", f.EBIT)
	fmt.Printf("Interest: $%.2f\n", f.Interest)
	fmt.Printf("EBT: $%.2f\n", f.EBT)
	fmt.Printf("Dividends Paid: $%.2f\n", f.DividendsPaid)
	fmt.Printf("Taxes: $%.2f\n", f.Taxes)
	fmt.Printf("Net Income: $%.2f\n", f.NetIncome)
	fmt.Printf("Cash Flows from Operations: $%.2f\n", f.CashFlowsFromOperations)
	fmt.Printf("Cash Flows from Investing Activities: $%.2f\n", f.CashFlowsFromInvestingActivities)
	fmt.Printf("Cash Flows from Financing Activities: $%.2f\n", f.CashFlowsFromFinancingActivities)
	fmt.Printf("Net Increase in Cash: $%.2f\n", f.NetIncreaseInCash)
	fmt.Printf("Actual Cash Flow: $%.2f\n", f.ActualCashFlow)
	fmt.Printf("Cash Flow Discrepancy: $%.2f\n", f.CashFlowDiscrepancy)
	fmt.Printf("Total Starting Cash: $%.2f\n", f.StartingCashInventory)
	fmt.Printf("Total Ending Cash: $%.2f\n", f.EndingCashInventory)
	fmt.Printf("Ending Working Asset Inventory: $%.2f\n", f.WorkingAssetEndingValue)
	fmt.Printf("Ending Fixed Asset Inventory: $%.2f\n", f.FixedAssetEndingValue)
	fmt.Printf("Balance Sheet Assets: $%.2f\n", f.BalanceSheetAssets)
	fmt.Printf("Current Liabilities: $%.2f\n", f.CurrentLiabilities)
	fmt.Printf("Non-Current Liabilities: $%.2f\n", f.NonCurrentLiabilities)
	fmt.Printf("Employee Reimbursable Expenses: $%.2f\n", f.EmployeeReimbursableExpenses)
	fmt.Printf("Shareholder Equity One: $%.2f\n", f.ShareholderEquityOne)
	fmt.Printf("Shareholder Equity Two: $%.2f\n", f.ShareholderEquityTwo)
	fmt.Printf("Issuance of Common Stock: $%.2f\n", f.IssuanceOfCommonStock)
	fmt.Printf("Retained Earnings: $%.2f\n", f.RetainedEarnings)
	fmt.Printf("Balance Sheet Liabilities: $%.2f\n", f.BalanceSheetLiabilities)
	fmt.Printf("Asset Liability Difference: $%.2f\n", f.BalanceSheetTotal)
}

func ParseEntry(line string, id int) (Entry, error) {
	purchasePattern := regexp.MustCompile(`We purchased ([\d.]+) units of (.+?) for \$([\d.]+) on (\d{4}-\d{2}-\d{2}), a (.+?)\.`)
	boughtCashExpensePattern := regexp.MustCompile(`We bought ([\d.]+) units of (.+?) for \$([\d.]+) on (\d{4}-\d{2}-\d{2}), a (.+?)\.`)
	beginningPattern := regexp.MustCompile(`We have a beginning inventory of ([\d.]+) units of (.+?) for \$([\d,\.]+) on (\d{4}-\d{2}-\d{2}), a (.+?)\.`)
	endingPattern := regexp.MustCompile(`We have an ending inventory of ([\d.]+) units of (.+?) for \$([\d,\.]+) on (\d{4}-\d{2}-\d{2}), a (.+?)\.`)
	salePattern := regexp.MustCompile(`We sold ([\d.]+) units of (.+?) for \$([\d.]+) on (\d{4}-\d{2}-\d{2}), a (.+?)\.`)
	expensePattern := regexp.MustCompile(`We incurred ([\d.]+) units of (.+?) for \$(-?[\d,\.]+) on (\d{4}-\d{2}-\d{2}), an? (.+?)\.`)

	var e Entry
	e.ID = id

	switch {
	case purchasePattern.MatchString(line):
		matches := purchasePattern.FindStringSubmatch(line)
		if len(matches) != 6 {
			return e, fmt.Errorf("invalid purchase entry: %s", line)
		}
		e.Units, _ = strconv.ParseFloat(matches[1], 64)
		e.Description = fmt.Sprintf("Purchased %.2f units of %s", e.Units, matches[2])
		value, _ := strconv.ParseFloat(matches[3], 64)
		e.Value = RoundToPrecision001(value)
		e.Date, _ = time.Parse("2006-01-02", matches[4])
		switch matches[5] {
		case "fixed asset":
			e.Category = FixedAssets
		case "working asset":
			e.Category = WorkingAssets
		case "cash asset":
			e.Category = Cash
		default:
			return e, fmt.Errorf("unrecognized purchase category: %s", matches[5])
		}

	case boughtCashExpensePattern.MatchString(line): // NEW
		matches := boughtCashExpensePattern.FindStringSubmatch(line)
		if len(matches) != 6 {
			return e, fmt.Errorf("invalid bought cash expense entry: %s", line)
		}
		e.Units, _ = strconv.ParseFloat(matches[1], 64)
		e.Description = fmt.Sprintf("Bought %.2f units of %s", e.Units, matches[2])
		value, _ := strconv.ParseFloat(matches[3], 64)
		e.Value = RoundToPrecision001(value)
		e.Date, _ = time.Parse("2006-01-02", matches[4])
		switch matches[5] {
		case "fixed asset":
			e.Category = FixedAssets
		case "working asset":
			e.Category = WorkingAssets
		case "cash asset":
			e.Category = Cash
		case "operating expense":
			e.Category = OperatingExpenses
		case "cash expense":
			e.Category = OperatingExpenses // Accept "cash expense" as operating expense
		default:
			return e, fmt.Errorf("unrecognized bought cash expense category: %s", matches[5])
		}
	case beginningPattern.MatchString(line):
		matches := beginningPattern.FindStringSubmatch(line)
		if len(matches) != 6 {
			return e, fmt.Errorf("invalid beginning inventory entry: %s", line)
		}
		e.Units, _ = strconv.ParseFloat(matches[1], 64)
		e.Description = fmt.Sprintf("Starting inventory of %.2f units of %s", e.Units, matches[2])
		valueStr := strings.ReplaceAll(matches[3], ",", "")
		value, _ := strconv.ParseFloat(valueStr, 64)
		e.Value = RoundToPrecision001(value)
		e.Date, _ = time.Parse("2006-01-02", matches[4])
		switch matches[5] {
		case "fixed asset":
			e.Category = FixedAssets
		case "working asset":
			e.Category = WorkingAssets
		case "cash asset":
			e.Category = Cash
		default:
			return e, fmt.Errorf("unrecognized beginning category: %s", matches[5])
		}
		e.IsBeginning = true

	case endingPattern.MatchString(line):
		matches := endingPattern.FindStringSubmatch(line)
		if len(matches) != 6 {
			return e, fmt.Errorf("invalid ending inventory entry: %s", line)
		}
		e.Units, _ = strconv.ParseFloat(matches[1], 64)
		e.Description = fmt.Sprintf("Ending inventory of %.2f units of %s", e.Units, matches[2])
		valueStr := strings.ReplaceAll(matches[3], ",", "")
		value, _ := strconv.ParseFloat(valueStr, 64)
		e.Value = RoundToPrecision001(value)
		e.Date, _ = time.Parse("2006-01-02", matches[4])
		switch matches[5] {
		case "fixed asset":
			e.Category = FixedAssets
		case "working asset":
			e.Category = WorkingAssets
		case "cash asset":
			e.Category = Cash
		default:
			return e, fmt.Errorf("unrecognized ending category: %s", matches[5])
		}
		e.IsEnding = true

	case salePattern.MatchString(line):
		matches := salePattern.FindStringSubmatch(line)
		if len(matches) != 6 {
			return e, fmt.Errorf("invalid sale entry: %s", line)
		}
		e.Units, _ = strconv.ParseFloat(matches[1], 64)
		e.Description = fmt.Sprintf("Sold %.2f units of %s", e.Units, matches[2])
		value, _ := strconv.ParseFloat(matches[3], 64)
		e.Value = RoundToPrecision001(value)
		e.Date, _ = time.Parse("2006-01-02", matches[4])
		if matches[5] == "working asset" {
			e.Category = WorkingAssets
			e.IsSale = true
		} else {
			return e, fmt.Errorf("unrecognized sale category: %s", matches[5])
		}

	case expensePattern.MatchString(line):
		matches := expensePattern.FindStringSubmatch(line)
		if len(matches) != 6 {
			return e, fmt.Errorf("invalid incurred entry: %s", line)
		}
		e.Units, _ = strconv.ParseFloat(matches[1], 64)
		e.Description = fmt.Sprintf("Incurred %.2f units of %s", e.Units, matches[2])
		valueStr := strings.ReplaceAll(matches[3], ",", "")
		value, _ := strconv.ParseFloat(valueStr, 64)
		e.Value = RoundToPrecision001(value)
		e.Date, _ = time.Parse("2006-01-02", matches[4])
		switch matches[5] {
		case "operating expense":
			e.Category = OperatingExpenses
		case "non-cash operating expense": // New case
			e.Category = NonCashOperatingExpenses
		case "interest":
			e.Category = Interest
		case "dividend":
			e.Category = Dividend
		case "equity":
			if strings.Contains(strings.ToLower(matches[2]), "issuance") {
				e.Category = EquityIssuance
			} else if strings.Contains(strings.ToLower(matches[2]), "shareholder equity one") {
				e.Category = ShareholdersEquity
				e.Description = "Shareholder Equity One"
			} else if strings.Contains(strings.ToLower(matches[2]), "shareholder equity two") {
				e.Category = ShareholdersEquity
				e.Description = "Shareholder Equity Two"
			} else {
				e.Category = ShareholdersEquity
			}
		case "employee reimbursable expense":
			e.Category = EmployeeReimbursableExpenses
		case "liability":
			switch strings.ToLower(matches[2]) {
			case "current liability":
				e.Category = CurrentLiabilities
			case "non-current liability":
				e.Category = NonCurrentLiabilities
			case "shareholder's equity":
				e.Category = ShareholdersEquity
			case "retained earnings":
				e.Category = Liabilities
			case "employee reimbursable expenses":
				e.Category = EmployeeReimbursableExpenses
			default:
				e.Category = Liabilities
			}
		default:
			return e, fmt.Errorf("unrecognized incurred type: %s", matches[5])
		}

	default:
		return e, fmt.Errorf("unrecognized entry format: %s", line)
	}

	return e, nil
}

func LoadEntriesFromFile(filename string, f *Financials) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	id := 1
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		entry, err := ParseEntry(line, id)
		if err != nil {
			fmt.Printf("Error parsing line %d: %v\n", id, err)
			continue
		}
		f.AddEntry(entry)
		id++
	}

	return scanner.Err()
}

func main() {
	f := NewFinancials()

	err := LoadEntriesFromFile("entries.txt", f)
	if err != nil {
		fmt.Printf("Error loading entries: %v\n", err)
		return
	}

	f.CalculateAggregates()
	f.PrintAggregates()

	fmt.Println("\nEntries:")
	for _, entry := range f.Entries {
		fmt.Printf("ID: %d, Description: %s, Value: $%.2f, Date: %s, Category: %s, Beginning: %v, Ending: %v, Sale: %v\n",
			entry.ID, entry.Description, entry.Value, entry.Date.Format("2006-01-02"), entry.Category, entry.IsBeginning, entry.IsEnding, entry.IsSale)
	}
}
