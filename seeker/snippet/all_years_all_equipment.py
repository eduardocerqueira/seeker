#date: 2024-06-03T17:04:22Z
#url: https://api.github.com/gists/ba2673780f6cef659167d10b73ee55c5
#owner: https://api.github.com/users/thejevans

# ------------------------------------------------------------------------------
# import packages
# ------------------------------------------------------------------------------

import pathlib

import polars as pl
from pypika import Query, Tables
import pyodbc

# ------------------------------------------------------------------------------
# define constants
# ------------------------------------------------------------------------------

OIP_SHARED_DRIVE = pathlib.Path(
    'X:/Shared drives/Office of Innovation Shared Drive/')

OUTPUT_DIR = pathlib.Path.home() / 'Desktop/all_years_all_equipment'

EQUIPMENT, FACILITY_REPORT, COMPANY_SUMMARY, WELLS, COMPANY_REPORT = Tables(
    'Equipment',
    'FacilityReport',
    'CompanySummary',
    'Wells',
    'CompanyReport',
)

# ------------------------------------------------------------------------------
# build queries
# ------------------------------------------------------------------------------

unique_years_query = str(
    Query()
        .from_(COMPANY_REPORT)
        .distinct()
        .select(COMPANY_REPORT.InventoryYear)
)

def unique_equipment_groups_query(year: int) -> str:
    return str(
        Query()
            .from_(EQUIPMENT)
            .left_join(FACILITY_REPORT)
            .on_field('FacilityReportID')
            .left_join(COMPANY_REPORT)
            .on_field('CompanyReportID')
            .distinct()
            .select(
                EQUIPMENT.EquipmentGroup,
                COMPANY_REPORT.InventoryYear,
            )
            .where(COMPANY_REPORT.InventoryYear == year)
    )

def equipment_query(equipment_group: str, year: int) -> str:
    return str(
        Query()
            .from_(EQUIPMENT)
            .left_join(FACILITY_REPORT)
            .on_field('FacilityReportID')
            .left_join(COMPANY_REPORT)
            .on_field('CompanyReportID')
            .select(
                EQUIPMENT.star,
                FACILITY_REPORT.FacilityLatitude,
                FACILITY_REPORT.FacilityLongitude,
                FACILITY_REPORT.COGCCLocationID,
                FACILITY_REPORT.AnnualFacilityProductionGas,
                FACILITY_REPORT.AnnualFacilityProductionHCLiquid,
                FACILITY_REPORT.AnnualFacilityProductionWater,
                COMPANY_REPORT.CompanyName,
                COMPANY_REPORT.CustomerNumber,
                COMPANY_REPORT.ReportType,
                COMPANY_REPORT.InventoryYear,
            )
            .where(EQUIPMENT.EquipmentGroup == equipment_group)
            .where(COMPANY_REPORT.InventoryYear == year)
    )

# ------------------------------------------------------------------------------
# execute queries on database and assign to polars dataframes
# ------------------------------------------------------------------------------

with pyodbc.connect(DSN='ONGAEIR') as cnxn:
    equipment_dfs: dict[int, dict[str, pl.DataFrame]] = {}
    years = pl.read_database(unique_years_query, cnxn)['InventoryYear'].to_list()

    for year in years:
        equipment_dfs[year] = {}
        equipment_groups = pl.read_database(
            unique_equipment_groups_query(year), cnxn)['EquipmentGroup'].to_list()

        for eq in equipment_groups:
            df = pl.read_database(
                equipment_query(eq, year),
                cnxn,
            )

            equipment_dfs[year][eq] = df[[
                s.name
                for s in df
                if not (s.null_count() == df.height)
            ]]

# ------------------------------------------------------------------------------
# save to csv
# ------------------------------------------------------------------------------

for year, year_df in equipment_dfs.items():
    for eq, df in year_df.items():
        save_dir = OUTPUT_DIR / str(year)
        save_dir.mkdir(parents=True, exist_ok=True)
        df.write_csv(save_dir / f'{eq}.csv')