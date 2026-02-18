#date: 2026-02-18T17:32:31Z
#url: https://api.github.com/gists/633e132c70dcd10db1fad86a6ee83f53
#owner: https://api.github.com/users/datavudeja

import pandas as pd
from typing import Dict, List

class DataQualityValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
    
    def check_nulls(self, columns: List[str], threshold: float = 0.05):
        """Check if null percentage exceeds threshold"""
        for col in columns:
            null_pct = self.df[col].isnull().sum() / len(self.df)
            if null_pct > threshold:
                self.issues.append(f"{col}: {null_pct:.2%} nulls (threshold: {threshold:.2%})")
    
    def check_duplicates(self, subset: List[str]):
        """Check for duplicate records"""
        dup_count = self.df.duplicated(subset=subset).sum()
        if dup_count > 0:
            self.issues.append(f"Found {dup_count} duplicate records")
    
    def check_range(self, column: str, min_val: float, max_val: float):
        """Check if values are within expected range"""
        out_of_range = ((self.df[column] < min_val) | (self.df[column] > max_val)).sum()
        if out_of_range > 0:
            self.issues.append(f"{column}: {out_of_range} values out of range [{min_val}, {max_val}]")
    
    def validate(self) -> Dict:
        """Run all validations and return report"""
        return {
            "passed": len(self.issues) == 0,
            "issues": self.issues,
            "row_count": len(self.df)
        }

# Usage
df = pd.read_csv("data.csv")
validator = DataQualityValidator(df)
validator.check_nulls(["customer_id", "order_date"])
validator.check_duplicates(["order_id"])
validator.check_range("amount", 0, 100000)
print(validator.validate())
