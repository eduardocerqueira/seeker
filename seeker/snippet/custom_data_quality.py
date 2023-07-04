#date: 2023-07-04T16:55:23Z
#url: https://api.github.com/gists/ea3528e0d8b08f7205b0fe441b4b9cfb
#owner: https://api.github.com/users/elijahbenizzy

from hamilton.function_modifiers import check_output_custom
from hamilton.data_quality.base import DataValidator, ValidationResult, DataValidationLevel
from hamilton.ad_hoc_utils import create_temporary_module
from hamilton import driver



class UniqueColumnsValidator(DataValidator):
    def __init__(self, importance: str):
        super(UniqueColumnsValidator, self).__init__(importance=importance)

    @classmethod
    def applies_to(cls, datatype: Type[Type]) -> bool:
        return issubclass(datatype, pd.DataFrame)

    def description(self) -> str:
        return "Columns must be unique"

    @classmethod
    def name(cls) -> str:
        return "unique_columns"

    def validate(self, dataset: pd.DataFrame) -> ValidationResult:
        passes = dataset.columns.is_unique
        return ValidationResult(
            passes=passes,
            message=f"Columns not unique: {dataset.columns}"
        )

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.equal_to == other.equal_to

@check_output_custom(UniqueColumnsValidator(DataValidationLevel.FAIL))
def df() -> pd.DataFrame:
    out = pd.DataFrame.from_records([{'a' : 1, 'b' : 2}])
    return out

@check_output_custom(UniqueColumnsValidator(DataValidationLevel.FAIL))
def df_non_unique() -> pd.DataFrame:
    out = pd.DataFrame.from_records([{'a' : 1, 'b' : 2}])
    out.columns = ['a', 'a']
    return out

if __name__ == '__main__':
    mod = create_temporary_module(df, df_non_unique)
    dr = driver.Driver({}, mod)
    dr.visualize_execution(['df', 'df_non_unique'], "./out", {})
    dr.execute(['df', 'df_non_unique'])