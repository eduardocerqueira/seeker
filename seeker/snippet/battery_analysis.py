#date: 2025-10-07T17:01:16Z
#url: https://api.github.com/gists/d73aa2fe9867f25d35f0c63e85877f9b
#owner: https://api.github.com/users/larsenglund

#!/usr/bin/env python3
"""
Battery Storage Economic Analysis - 10kWh System, SEK Currency

This script analyzes the economic feasibility of installing a 10kWh battery storage system
costing 50,000 SEK for a residential house, with all costs converted to Swedish Kronor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BatteryAnalysisSEK:
    def __init__(self, battery_capacity_kwh=10, max_power_kw=10, efficiency=0.95, system_cost_sek=50000):
        """
        Initialize battery analysis parameters
        
        Args:
            battery_capacity_kwh: Battery storage capacity in kWh
            max_power_kw: Maximum charge/discharge power in kW
            efficiency: Round-trip efficiency (0-1)
            system_cost_sek: Total system cost in SEK
        """
        self.battery_capacity = battery_capacity_kwh
        self.max_power = max_power_kw
        self.efficiency = efficiency
        self.system_cost_sek = system_cost_sek
        self.eur_to_sek = 11.5  # Approximate exchange rate EUR to SEK
        
    def load_data(self):
        """Load and preprocess the consumption and price data"""
        print("Loading data files...")
        
        # Load consumption and production data
        consumption_df = pd.read_csv('hourly_production_and_consumption.csv', sep=';', decimal=',')
        consumption_df['Datum'] = pd.to_datetime(consumption_df['Datum'], format='%Y-%m-%d %H:%M')
        consumption_df.set_index('Datum', inplace=True)
        
        # Load price data  
        price_df = pd.read_csv('hourly_power_price.csv')
        
        # Parse the time range in price data to get start time
        price_df['start_time'] = pd.to_datetime(price_df['MTU (UTC)'].str.split(' - ').str[0], 
                                               format='%d/%m/%Y %H:%M:%S')
        
        # Convert to local time (assuming UTC+1 for Sweden)
        price_df['local_time'] = price_df['start_time'] + pd.Timedelta(hours=1)
        price_df.set_index('local_time', inplace=True)
        
        # Convert price from EUR/MWh to SEK/kWh
        price_df['price_sek_kwh'] = pd.to_numeric(price_df['Day-ahead Price (EUR/MWh)']) / 1000 * self.eur_to_sek
        
        # Convert consumption and production to float
        consumption_df['consumption_kwh'] = pd.to_numeric(consumption_df['El kWh'])
        consumption_df['production_kwh'] = pd.to_numeric(consumption_df['Produktion'])
        
        # Merge data on timestamp with inner join to get only matching times
        merged_df = consumption_df[['consumption_kwh', 'production_kwh']].join(
            price_df[['price_sek_kwh']], how='inner')
        
        # Clean data - remove any rows with missing values
        merged_df = merged_df.dropna()
        
        # Calculate net consumption (positive means buying from grid, negative means selling to grid)
        merged_df['net_consumption'] = merged_df['consumption_kwh'] - merged_df['production_kwh']
        
        print(f"Data loaded: {len(merged_df)} hourly records")
        print(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
        print(f"Average hourly consumption: {merged_df['consumption_kwh'].mean():.2f} kWh")
        print(f"Average hourly production: {merged_df['production_kwh'].mean():.2f} kWh")
        print(f"Average net consumption: {merged_df['net_consumption'].mean():.2f} kWh")
        print(f"Average price: {merged_df['price_sek_kwh'].mean():.3f} SEK/kWh")
        print(f"Price range: {merged_df['price_sek_kwh'].min():.3f} - {merged_df['price_sek_kwh'].max():.3f} SEK/kWh")
        
        return merged_df
    
    def baseline_scenario(self, df):
        """Calculate costs for baseline scenario (no battery)"""
        print("\nCalculating baseline scenario (no battery)...")
        
        # For baseline, we only buy electricity when net consumption is positive
        # Never sell excess production back to grid (no selling price data)
        # Add power transfer cost (grid fees) of 0.685 SEK/kWh for purchases
        power_transfer_cost = 0.685  # SEK/kWh
        
        df['grid_purchase_kwh'] = df['net_consumption'].clip(lower=0)  # Only positive values
        df['total_cost_per_kwh'] = df['price_sek_kwh'] + power_transfer_cost
        df['baseline_cost'] = df['grid_purchase_kwh'] * df['total_cost_per_kwh']
        
        total_cost = df['baseline_cost'].sum()
        total_purchased = df['grid_purchase_kwh'].sum()
        total_excess_wasted = df[df['net_consumption'] < 0]['net_consumption'].abs().sum()
        avg_total_cost_per_kwh = df['total_cost_per_kwh'].mean()
        
        print(f"Total annual cost (baseline): {total_cost:.0f} SEK")
        print(f"Total energy purchased from grid: {total_purchased:.0f} kWh")
        print(f"Total excess solar energy (unused): {total_excess_wasted:.0f} kWh")
        print(f"Average total cost (energy + transfer): {avg_total_cost_per_kwh:.3f} SEK/kWh")
        
        return df, total_cost
    
    def baseline_scenario_with_selling(self, df):
        """Calculate costs for baseline scenario with grid selling (no battery)"""
        print("\nCalculating baseline scenario with grid selling (no battery)...")
        
        # Can sell excess production at energy price (no transfer costs)
        # Must pay transfer costs when purchasing
        power_transfer_cost = 0.685  # SEK/kWh
        
        df['grid_purchase_kwh'] = df['net_consumption'].clip(lower=0)  # Only positive values
        df['grid_sale_kwh'] = df['net_consumption'].clip(upper=0).abs()  # Only negative values (made positive)
        
        df['purchase_cost'] = df['grid_purchase_kwh'] * (df['price_sek_kwh'] + power_transfer_cost)
        df['sale_revenue'] = df['grid_sale_kwh'] * df['price_sek_kwh']  # No transfer cost on sales
        df['baseline_selling_cost'] = df['purchase_cost'] - df['sale_revenue']
        
        total_cost = df['baseline_selling_cost'].sum()
        total_purchased = df['grid_purchase_kwh'].sum()
        total_sold = df['grid_sale_kwh'].sum()
        avg_purchase_cost = (df['price_sek_kwh'].mean() + power_transfer_cost)
        avg_sale_price = df['price_sek_kwh'].mean()
        
        print(f"Total annual cost (baseline with selling): {total_cost:.0f} SEK")
        print(f"Total energy purchased from grid: {total_purchased:.0f} kWh")
        print(f"Total energy sold to grid: {total_sold:.0f} kWh")
        print(f"Average purchase cost (energy + transfer): {avg_purchase_cost:.3f} SEK/kWh")
        print(f"Average sale price (energy only): {avg_sale_price:.3f} SEK/kWh")
        
        return df, total_cost
    
    def baseline_scenario_with_selling(self, df):
        """Calculate costs for baseline scenario with grid selling (no battery)"""
        print("\nCalculating baseline scenario with grid selling (no battery)...")
        
        # Can sell excess production at energy price (no transfer costs)
        # Must pay transfer costs when purchasing
        power_transfer_cost = 0.685  # SEK/kWh
        
        df['grid_purchase_kwh'] = df['net_consumption'].clip(lower=0)  # Only positive values
        df['grid_sale_kwh'] = df['net_consumption'].clip(upper=0).abs()  # Only negative values (made positive)
        
        df['purchase_cost'] = df['grid_purchase_kwh'] * (df['price_sek_kwh'] + power_transfer_cost)
        df['sale_revenue'] = df['grid_sale_kwh'] * df['price_sek_kwh']  # No transfer cost on sales
        df['baseline_selling_cost'] = df['purchase_cost'] - df['sale_revenue']
        
        total_cost = df['baseline_selling_cost'].sum()
        total_purchased = df['grid_purchase_kwh'].sum()
        total_sold = df['grid_sale_kwh'].sum()
        avg_purchase_cost = (df['price_sek_kwh'].mean() + power_transfer_cost)
        avg_sale_price = df['price_sek_kwh'].mean()
        
        print(f"Total annual cost (baseline with selling): {total_cost:.0f} SEK")
        print(f"Total energy purchased from grid: {total_purchased:.0f} kWh")
        print(f"Total energy sold to grid: {total_sold:.0f} kWh")
        print(f"Average purchase cost (energy + transfer): {avg_purchase_cost:.3f} SEK/kWh")
        print(f"Average sale price (energy only): {avg_sale_price:.3f} SEK/kWh")
        
        return df, total_cost
    
    def battery_scenario(self, df):
        """Calculate costs for battery scenario with AGGRESSIVE arbitrage strategy"""
        print(f"\nCalculating battery scenario with ENHANCED ARBITRAGE strategy...")
        print(f"Battery: {self.battery_capacity}kWh, {self.max_power}kW, Perfect 24h price foresight")
        
        # Initialize battery state
        df = df.copy()
        df['battery_soc'] = 0.0  # State of charge in kWh
        df['battery_charge'] = 0.0  # Power charged to battery (positive)
        df['battery_discharge'] = 0.0  # Power discharged from battery (positive)
        df['grid_purchase'] = 0.0  # Power bought from grid
        df['excess_solar_wasted'] = 0.0  # Excess solar not used (no selling)
        df['battery_cost'] = 0.0  # Cost for this hour
        
        # Power transfer cost
        power_transfer_cost = 0.685  # SEK/kWh
        
        # Advanced arbitrage strategy with perfect 24-hour foresight
        df_sorted = df.sort_index()
        current_soc = self.battery_capacity * 0.5  # Start with 50% charge
        
        for i, (timestamp, row) in enumerate(df_sorted.iterrows()):
            net_demand = row['net_consumption']
            price = row['price_sek_kwh']
            
            # Look ahead 24 hours for advanced price analysis
            end_lookahead = min(i + 24, len(df_sorted))
            lookahead_prices = df_sorted.iloc[i:end_lookahead]['price_sek_kwh']
            
            # Advanced price thresholds for aggressive arbitrage
            min_price_24h = lookahead_prices.min()
            max_price_24h = lookahead_prices.max()
            q25_price = lookahead_prices.quantile(0.25)  # Bottom 25% prices
            q75_price = lookahead_prices.quantile(0.75)  # Top 25% prices
            median_price = lookahead_prices.median()
            
            # Calculate price ranking (0-1, where 0 is cheapest, 1 is most expensive)
            price_range = max_price_24h - min_price_24h
            if price_range > 0:
                price_rank = (price - min_price_24h) / price_range
            else:
                price_rank = 0.5
            
            charge_power = 0.0
            discharge_power = 0.0
            grid_purchase = 0.0
            excess_solar_wasted = 0.0
            
            # NO-SELL ARBITRAGE STRATEGY: Only purchase from grid, use solar optimally
            
            if net_demand > 0:  # House needs power
                
                # VERY CHEAP prices (bottom 10%): Charge aggressively even beyond immediate need
                if price_rank < 0.1 and current_soc < self.battery_capacity * 0.95:
                    # Extremely cheap - charge battery to near full capacity
                    max_charge = min(self.max_power, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                # CHEAP prices (bottom 25%): Charge moderately while meeting demand
                elif price_rank < 0.25 and current_soc < self.battery_capacity * 0.85:
                    # Cheap - charge battery while meeting demand
                    max_charge = min(self.max_power * 0.8, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                # EXPENSIVE prices (top 25%): Use battery aggressively
                elif price_rank > 0.75 and current_soc > self.battery_capacity * 0.05:
                    # Expensive - discharge as much as possible
                    discharge_power = min(net_demand, self.max_power, current_soc * self.efficiency)
                    remaining_demand = net_demand - discharge_power
                    grid_purchase = max(0, remaining_demand)
                
                # VERY EXPENSIVE prices (top 10%): Use every bit of battery
                elif price_rank > 0.9 and current_soc > 0.01:
                    # Very expensive - discharge everything possible
                    discharge_power = min(net_demand, self.max_power, current_soc * self.efficiency)
                    remaining_demand = net_demand - discharge_power
                    grid_purchase = max(0, remaining_demand)
                
                # NEGATIVE prices: Charge to maximum regardless of demand
                elif price < 0 and current_soc < self.battery_capacity:
                    # Negative prices - charge everything possible!
                    max_charge = min(self.max_power, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                else:
                    # Normal/medium prices: just meet demand
                    grid_purchase = net_demand
            
            else:  # House has excess solar production
                excess = abs(net_demand)
                
                # ALWAYS use excess solar for battery charging first (free energy!)
                # Then waste any remaining excess (no selling to grid)
                
                if current_soc < self.battery_capacity:
                    # Charge battery with available excess solar
                    charge_power = min(excess, self.max_power, 
                                     (self.battery_capacity - current_soc) / self.efficiency)
                    remaining_excess = excess - charge_power
                    excess_solar_wasted = remaining_excess
                    
                    # For very cheap prices, consider buying additional power to charge more
                    if (price < 0 or price_rank < 0.05) and current_soc < self.battery_capacity:
                        additional_charge = min(self.max_power - charge_power, 
                                              (self.battery_capacity - current_soc) / self.efficiency - charge_power)
                        if additional_charge > 0:
                            grid_purchase = additional_charge
                            charge_power += additional_charge
                else:
                    # Battery full, waste all excess solar
                    excess_solar_wasted = excess
            
            # Update battery state of charge
            energy_in = charge_power * self.efficiency
            energy_out = discharge_power / self.efficiency
            current_soc += energy_in - energy_out
            current_soc = max(0, min(self.battery_capacity, current_soc))
            
            # Calculate cost for this hour (only purchases, no sales)
            total_cost_per_kwh = price + power_transfer_cost
            hour_cost = grid_purchase * total_cost_per_kwh
            
            # Store results
            df_sorted.loc[timestamp, 'battery_soc'] = current_soc
            df_sorted.loc[timestamp, 'battery_charge'] = charge_power
            df_sorted.loc[timestamp, 'battery_discharge'] = discharge_power
            df_sorted.loc[timestamp, 'grid_purchase'] = grid_purchase
            df_sorted.loc[timestamp, 'excess_solar_wasted'] = excess_solar_wasted
            df_sorted.loc[timestamp, 'battery_cost'] = hour_cost
        
        # Update the original dataframe
        df.update(df_sorted)
        
        total_cost = df['battery_cost'].sum()
        total_charged = df['battery_charge'].sum()
        total_discharged = df['battery_discharge'].sum()
        
        total_grid_purchases = df['grid_purchase'].sum()
        total_excess_wasted = df['excess_solar_wasted'].sum()
        
        print(f"Total annual cost (with battery): {total_cost:.0f} SEK")
        print(f"Total energy purchased from grid: {total_grid_purchases:.0f} kWh")
        print(f"Total energy charged to battery: {total_charged:.0f} kWh")
        print(f"Total energy discharged from battery: {total_discharged:.0f} kWh")
        print(f"Total excess solar wasted: {total_excess_wasted:.0f} kWh")
        if total_charged > 0:
            print(f"Battery utilization efficiency: {(total_discharged/total_charged)*100:.1f}%")
        
        return df, total_cost
    
    def battery_scenario_with_selling(self, df):
        """Calculate costs for battery scenario with grid selling capability"""
        print(f"\nCalculating battery scenario with GRID SELLING capability...")
        print(f"Battery: {self.battery_capacity}kWh, {self.max_power}kW, Can sell to grid at energy price")
        
        # Initialize battery state
        df = df.copy()
        df['battery_soc'] = 0.0  # State of charge in kWh
        df['battery_charge'] = 0.0  # Power charged to battery (positive)
        df['battery_discharge'] = 0.0  # Power discharged from battery (positive)
        df['grid_purchase'] = 0.0  # Power bought from grid
        df['grid_sale'] = 0.0  # Power sold to grid
        df['battery_selling_cost'] = 0.0  # Cost for this hour
        
        # Power transfer cost
        power_transfer_cost = 0.685  # SEK/kWh
        
        # Enhanced strategy with selling capability
        df_sorted = df.sort_index()
        current_soc = self.battery_capacity * 0.5  # Start with 50% charge
        
        for i, (timestamp, row) in enumerate(df_sorted.iterrows()):
            net_demand = row['net_consumption']
            price = row['price_sek_kwh']
            
            # Look ahead 24 hours for advanced price analysis
            end_lookahead = min(i + 24, len(df_sorted))
            lookahead_prices = df_sorted.iloc[i:end_lookahead]['price_sek_kwh']
            
            # Advanced price thresholds for aggressive arbitrage
            min_price_24h = lookahead_prices.min()
            max_price_24h = lookahead_prices.max()
            
            # Calculate price ranking (0-1, where 0 is cheapest, 1 is most expensive)
            price_range = max_price_24h - min_price_24h
            if price_range > 0:
                price_rank = (price - min_price_24h) / price_range
            else:
                price_rank = 0.5
            
            charge_power = 0.0
            discharge_power = 0.0
            grid_purchase = 0.0
            grid_sale = 0.0
            
            # SELLING-ENABLED ARBITRAGE STRATEGY
            
            if net_demand > 0:  # House needs power
                
                # VERY CHEAP prices (bottom 10%): Charge aggressively even beyond immediate need
                if price_rank < 0.1 and current_soc < self.battery_capacity * 0.95:
                    max_charge = min(self.max_power, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                # CHEAP prices (bottom 25%): Charge moderately while meeting demand
                elif price_rank < 0.25 and current_soc < self.battery_capacity * 0.85:
                    max_charge = min(self.max_power * 0.8, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                # EXPENSIVE prices (top 25%): Use battery aggressively
                elif price_rank > 0.75 and current_soc > self.battery_capacity * 0.05:
                    discharge_power = min(net_demand, self.max_power, current_soc * self.efficiency)
                    remaining_demand = net_demand - discharge_power
                    grid_purchase = max(0, remaining_demand)
                
                # VERY EXPENSIVE prices (top 10%): Use every bit of battery
                elif price_rank > 0.9 and current_soc > 0.01:
                    discharge_power = min(net_demand, self.max_power, current_soc * self.efficiency)
                    remaining_demand = net_demand - discharge_power
                    grid_purchase = max(0, remaining_demand)
                
                # NEGATIVE prices: Charge to maximum regardless of demand
                elif price < 0 and current_soc < self.battery_capacity:
                    max_charge = min(self.max_power, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                else:
                    # Normal/medium prices: just meet demand
                    grid_purchase = net_demand
            
            else:  # House has excess solar production
                excess = abs(net_demand)
                
                # VERY EXPENSIVE prices (top 10%): Sell all excess, even discharge battery to sell more
                if price_rank > 0.9 and current_soc > 0.01:
                    # Sell excess production
                    grid_sale = excess
                    # Also discharge battery to sell more at high prices
                    additional_discharge = min(self.max_power, current_soc * self.efficiency)
                    discharge_power = additional_discharge
                    grid_sale += additional_discharge
                
                # EXPENSIVE prices (top 25%): Sell all excess production
                elif price_rank > 0.75:
                    grid_sale = excess
                
                # CHEAP prices or battery not full: Use excess to charge battery first
                elif current_soc < self.battery_capacity:
                    charge_power = min(excess, self.max_power, 
                                     (self.battery_capacity - current_soc) / self.efficiency)
                    remaining_excess = excess - charge_power
                    grid_sale = remaining_excess
                
                # Battery full: Sell all excess
                else:
                    grid_sale = excess
            
            # Update battery state of charge
            energy_in = charge_power * self.efficiency
            energy_out = discharge_power / self.efficiency
            current_soc += energy_in - energy_out
            current_soc = max(0, min(self.battery_capacity, current_soc))
            
            # Calculate cost for this hour (purchases include transfer cost, sales don't)
            purchase_cost = grid_purchase * (price + power_transfer_cost)
            sale_revenue = grid_sale * price  # No transfer cost on sales
            hour_cost = purchase_cost - sale_revenue
            
            # Store results
            df_sorted.loc[timestamp, 'battery_soc'] = current_soc
            df_sorted.loc[timestamp, 'battery_charge'] = charge_power
            df_sorted.loc[timestamp, 'battery_discharge'] = discharge_power
            df_sorted.loc[timestamp, 'grid_purchase'] = grid_purchase
            df_sorted.loc[timestamp, 'grid_sale'] = grid_sale
            df_sorted.loc[timestamp, 'battery_selling_cost'] = hour_cost
        
        # Update the original dataframe
        df.update(df_sorted)
        
        total_cost = df['battery_selling_cost'].sum()
        total_charged = df['battery_charge'].sum()
        total_discharged = df['battery_discharge'].sum()
        total_grid_purchases = df['grid_purchase'].sum()
        total_grid_sales = df['grid_sale'].sum()
        
        print(f"Total annual cost (with battery + selling): {total_cost:.0f} SEK")
        print(f"Total energy purchased from grid: {total_grid_purchases:.0f} kWh")
        print(f"Total energy sold to grid: {total_grid_sales:.0f} kWh")
        print(f"Total energy charged to battery: {total_charged:.0f} kWh")
        print(f"Total energy discharged from battery: {total_discharged:.0f} kWh")
        if total_charged > 0:
            print(f"Battery utilization efficiency: {(total_discharged/total_charged)*100:.1f}%")
        
        return df, total_cost
    
    def battery_scenario_with_selling(self, df):
        """Calculate costs for battery scenario with grid selling capability"""
        print(f"\nCalculating battery scenario with GRID SELLING capability...")
        print(f"Battery: {self.battery_capacity}kWh, {self.max_power}kW, Can sell to grid at energy price")
        
        # Initialize battery state
        df = df.copy()
        df['battery_soc'] = 0.0  # State of charge in kWh
        df['battery_charge'] = 0.0  # Power charged to battery (positive)
        df['battery_discharge'] = 0.0  # Power discharged from battery (positive)
        df['grid_purchase'] = 0.0  # Power bought from grid
        df['grid_sale'] = 0.0  # Power sold to grid
        df['battery_selling_cost'] = 0.0  # Cost for this hour
        
        # Power transfer cost
        power_transfer_cost = 0.685  # SEK/kWh
        
        # Enhanced strategy with selling capability
        df_sorted = df.sort_index()
        current_soc = self.battery_capacity * 0.5  # Start with 50% charge
        
        for i, (timestamp, row) in enumerate(df_sorted.iterrows()):
            net_demand = row['net_consumption']
            price = row['price_sek_kwh']
            
            # Look ahead 24 hours for advanced price analysis
            end_lookahead = min(i + 24, len(df_sorted))
            lookahead_prices = df_sorted.iloc[i:end_lookahead]['price_sek_kwh']
            
            # Advanced price thresholds for aggressive arbitrage
            min_price_24h = lookahead_prices.min()
            max_price_24h = lookahead_prices.max()
            
            # Calculate price ranking (0-1, where 0 is cheapest, 1 is most expensive)
            price_range = max_price_24h - min_price_24h
            if price_range > 0:
                price_rank = (price - min_price_24h) / price_range
            else:
                price_rank = 0.5
            
            charge_power = 0.0
            discharge_power = 0.0
            grid_purchase = 0.0
            grid_sale = 0.0
            
            # SELLING-ENABLED ARBITRAGE STRATEGY
            
            if net_demand > 0:  # House needs power
                
                # VERY CHEAP prices (bottom 10%): Charge aggressively even beyond immediate need
                if price_rank < 0.1 and current_soc < self.battery_capacity * 0.95:
                    max_charge = min(self.max_power, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                # CHEAP prices (bottom 25%): Charge moderately while meeting demand
                elif price_rank < 0.25 and current_soc < self.battery_capacity * 0.85:
                    max_charge = min(self.max_power * 0.8, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                # EXPENSIVE prices (top 25%): Use battery aggressively
                elif price_rank > 0.75 and current_soc > self.battery_capacity * 0.05:
                    discharge_power = min(net_demand, self.max_power, current_soc * self.efficiency)
                    remaining_demand = net_demand - discharge_power
                    grid_purchase = max(0, remaining_demand)
                
                # VERY EXPENSIVE prices (top 10%): Use every bit of battery
                elif price_rank > 0.9 and current_soc > 0.01:
                    discharge_power = min(net_demand, self.max_power, current_soc * self.efficiency)
                    remaining_demand = net_demand - discharge_power
                    grid_purchase = max(0, remaining_demand)
                
                # NEGATIVE prices: Charge to maximum regardless of demand
                elif price < 0 and current_soc < self.battery_capacity:
                    max_charge = min(self.max_power, (self.battery_capacity - current_soc) / self.efficiency)
                    total_purchase = net_demand + max_charge
                    grid_purchase = total_purchase
                    charge_power = max_charge
                
                else:
                    # Normal/medium prices: just meet demand
                    grid_purchase = net_demand
            
            else:  # House has excess solar production
                excess = abs(net_demand)
                
                # VERY EXPENSIVE prices (top 10%): Sell all excess, even discharge battery to sell more
                if price_rank > 0.9 and current_soc > 0.01:
                    # Sell excess production
                    grid_sale = excess
                    # Also discharge battery to sell more at high prices
                    additional_discharge = min(self.max_power, current_soc * self.efficiency)
                    discharge_power = additional_discharge
                    grid_sale += additional_discharge
                
                # EXPENSIVE prices (top 25%): Sell all excess production
                elif price_rank > 0.75:
                    grid_sale = excess
                
                # CHEAP prices or battery not full: Use excess to charge battery first
                elif current_soc < self.battery_capacity:
                    charge_power = min(excess, self.max_power, 
                                     (self.battery_capacity - current_soc) / self.efficiency)
                    remaining_excess = excess - charge_power
                    grid_sale = remaining_excess
                
                # Battery full: Sell all excess
                else:
                    grid_sale = excess
            
            # Update battery state of charge
            energy_in = charge_power * self.efficiency
            energy_out = discharge_power / self.efficiency
            current_soc += energy_in - energy_out
            current_soc = max(0, min(self.battery_capacity, current_soc))
            
            # Calculate cost for this hour (purchases include transfer cost, sales don't)
            purchase_cost = grid_purchase * (price + power_transfer_cost)
            sale_revenue = grid_sale * price  # No transfer cost on sales
            hour_cost = purchase_cost - sale_revenue
            
            # Store results
            df_sorted.loc[timestamp, 'battery_soc'] = current_soc
            df_sorted.loc[timestamp, 'battery_charge'] = charge_power
            df_sorted.loc[timestamp, 'battery_discharge'] = discharge_power
            df_sorted.loc[timestamp, 'grid_purchase'] = grid_purchase
            df_sorted.loc[timestamp, 'grid_sale'] = grid_sale
            df_sorted.loc[timestamp, 'battery_selling_cost'] = hour_cost
        
        # Update the original dataframe
        df.update(df_sorted)
        
        total_cost = df['battery_selling_cost'].sum()
        total_charged = df['battery_charge'].sum()
        total_discharged = df['battery_discharge'].sum()
        total_grid_purchases = df['grid_purchase'].sum()
        total_grid_sales = df['grid_sale'].sum()
        
        print(f"Total annual cost (with battery + selling): {total_cost:.0f} SEK")
        print(f"Total energy purchased from grid: {total_grid_purchases:.0f} kWh")
        print(f"Total energy sold to grid: {total_grid_sales:.0f} kWh")
        print(f"Total energy charged to battery: {total_charged:.0f} kWh")
        print(f"Total energy discharged from battery: {total_discharged:.0f} kWh")
        if total_charged > 0:
            print(f"Battery utilization efficiency: {(total_discharged/total_charged)*100:.1f}%")
        
        return df, total_cost
    
    def analyze_savings(self, baseline_cost, battery_cost):
        """Analyze potential savings and return on investment"""
        annual_savings = baseline_cost - battery_cost
        savings_percentage = (annual_savings / abs(baseline_cost)) * 100 if baseline_cost != 0 else 0
        
        print(f"\n=== ECONOMIC ANALYSIS (10kWh Battery, 50,000 SEK) ===")
        print(f"Annual cost savings with battery: {annual_savings:.0f} SEK ({savings_percentage:.1f}%)")
        print(f"Total system cost: {self.system_cost_sek:,} SEK")
        
        if annual_savings > 0:
            payback_years = self.system_cost_sek / annual_savings
            print(f"Simple payback period: {payback_years:.1f} years")
            
            # NPV calculation with Swedish market parameters
            discount_rate = 0.03  # Lower discount rate for Sweden
            system_life = 15
            degradation_rate = 0.02  # 2% capacity loss per year
            
            total_npv = 0
            for year in range(1, system_life + 1):
                # Account for battery degradation
                effective_savings = annual_savings * (1 - degradation_rate * (year - 1))
                discounted_savings = effective_savings / (1 + discount_rate)**year
                total_npv += discounted_savings
            
            npv = total_npv - self.system_cost_sek
            print(f"Net Present Value (15 years, 3% discount, 2% degradation): {npv:,.0f} SEK")
            
            # ROI calculation
            roi = (annual_savings / self.system_cost_sek) * 100
            print(f"Return on investment: {roi:.1f}% per year")
            
            if npv > 0:
                print("✅ Investment appears economically viable")
            else:
                print("❌ Investment does not appear economically viable")
                
            # Break-even analysis
            if payback_years <= 8:
                print("✅ Good payback period (≤8 years)")
            elif payback_years <= 12:
                print("⚠️  Marginal payback period (8-12 years)")
            else:
                print("❌ Poor payback period (>12 years)")
                
        else:
            print("❌ No savings - battery system increases costs")
        
        return annual_savings, self.system_cost_sek, npv if annual_savings > 0 else -self.system_cost_sek
    
    def create_summary_sek(self, df, baseline_cost, battery_cost):
        """Create a summary in SEK"""
        print("\n=== DETAILED RESULTS SUMMARY (SEK) ===")
        
        # Basic statistics
        annual_savings = baseline_cost - battery_cost
        
        # Monthly analysis
        monthly_baseline = df.groupby(df.index.month)['baseline_cost'].sum()
        monthly_battery = df.groupby(df.index.month)['battery_cost'].sum()
        monthly_savings = monthly_baseline - monthly_battery
        
        print("\nMonthly Savings Breakdown (SEK):")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num, month_name in enumerate(months, 1):
            if month_num in monthly_savings.index:
                savings = monthly_savings[month_num]
                baseline = monthly_baseline[month_num]
                battery = monthly_battery[month_num]
                print(f"{month_name}: Baseline {baseline:.0f} SEK, Battery {battery:.0f} SEK, Savings {savings:.0f} SEK")
        
        # Battery utilization
        total_charged = df['battery_charge'].sum()
        total_discharged = df['battery_discharge'].sum()
        
        print(f"\nBattery Performance:")
        print(f"Total energy charged: {total_charged:.0f} kWh")
        print(f"Total energy discharged: {total_discharged:.0f} kWh")
        print(f"Round-trip efficiency: {(total_discharged/total_charged)*100:.1f}%" if total_charged > 0 else "No battery usage")
        
        # Enhanced price analysis
        negative_price_hours = (df['price_sek_kwh'] < 0).sum()
        very_cheap_hours = (df['price_sek_kwh'] < df['price_sek_kwh'].quantile(0.1)).sum()
        cheap_hours = (df['price_sek_kwh'] < df['price_sek_kwh'].quantile(0.25)).sum()
        expensive_hours = (df['price_sek_kwh'] > df['price_sek_kwh'].quantile(0.75)).sum()
        very_expensive_hours = (df['price_sek_kwh'] > df['price_sek_kwh'].quantile(0.9)).sum()
        
        charging_negative = df[(df['price_sek_kwh'] < 0)]['battery_charge'].sum()
        charging_very_cheap = df[(df['price_sek_kwh'] < df['price_sek_kwh'].quantile(0.1))]['battery_charge'].sum()
        charging_cheap = df[(df['price_sek_kwh'] < df['price_sek_kwh'].quantile(0.25))]['battery_charge'].sum()
        
        discharging_expensive = df[(df['price_sek_kwh'] > df['price_sek_kwh'].quantile(0.75))]['battery_discharge'].sum()
        discharging_very_expensive = df[(df['price_sek_kwh'] > df['price_sek_kwh'].quantile(0.9))]['battery_discharge'].sum()
        
        print(f"\nNo-Sell Strategy Performance:")
        print(f"Hours with negative prices: {negative_price_hours} ({negative_price_hours/len(df)*100:.1f}%)")
        print(f"Charging during negative prices: {charging_negative:.0f} kWh")
        print(f"Charging during cheapest 10% of hours: {charging_very_cheap:.0f} kWh")
        print(f"Charging during cheapest 25% of hours: {charging_cheap:.0f} kWh")
        print(f"Discharging during most expensive 25% of hours: {discharging_expensive:.0f} kWh")
        print(f"Discharging during most expensive 10% of hours: {discharging_very_expensive:.0f} kWh")
        
        if negative_price_hours > 0:
            avg_negative_price = df[df['price_sek_kwh'] < 0]['price_sek_kwh'].mean()
            print(f"Average negative price: {avg_negative_price:.3f} SEK/kWh")
        
        # Calculate charging costs and discharge savings (including transfer costs)
        power_transfer_cost = 0.685
        total_cost_per_kwh = df['price_sek_kwh'] + power_transfer_cost
        
        # For charging: we pay energy price + transfer cost when buying from grid
        # For discharging: we save the total cost per kWh by not buying from grid
        grid_charging_mask = df['grid_purchase'] > df['net_consumption'].clip(lower=0)
        grid_charging_energy = df[grid_charging_mask]['battery_charge'].sum()
        
        if df['battery_charge'].sum() > 0:
            avg_charge_cost = (df['battery_charge'] * total_cost_per_kwh).sum() / df['battery_charge'].sum()
            avg_discharge_save = (df['battery_discharge'] * total_cost_per_kwh).sum() / df['battery_discharge'].sum()
            
            print(f"Average charging cost (energy + transfer): {avg_charge_cost:.3f} SEK/kWh")
            print(f"Average discharge savings (avoided cost): {avg_discharge_save:.3f} SEK/kWh")
            
            if avg_charge_cost > 0:
                net_benefit = avg_discharge_save - avg_charge_cost
                print(f"Net arbitrage benefit: {net_benefit:.3f} SEK/kWh ({net_benefit/avg_charge_cost*100:.1f}%)")
        
        # Solar utilization
        total_solar = df['production_kwh'].sum()
        total_wasted = df['excess_solar_wasted'].sum()
        solar_utilization = (total_solar - total_wasted) / total_solar * 100 if total_solar > 0 else 0
        print(f"Solar energy utilization: {solar_utilization:.1f}% ({total_solar-total_wasted:.0f}/{total_solar:.0f} kWh)")
        
        return monthly_savings
    
    def create_summary_selling(self, df, baseline_cost, battery_cost):
        """Create a summary for the grid selling scenario"""
        print("\n=== DETAILED RESULTS SUMMARY - GRID SELLING MODEL (SEK) ===")
        
        # Basic statistics
        annual_savings = baseline_cost - battery_cost
        
        # Monthly analysis
        monthly_baseline = df.groupby(df.index.month)['baseline_selling_cost'].sum()
        monthly_battery = df.groupby(df.index.month)['battery_selling_cost'].sum()
        monthly_savings = monthly_baseline - monthly_battery
        
        print("\nMonthly Savings Breakdown (SEK):")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num, month_name in enumerate(months, 1):
            if month_num in monthly_savings.index:
                savings = monthly_savings[month_num]
                baseline = monthly_baseline[month_num]
                battery = monthly_battery[month_num]
                print(f"{month_name}: Baseline {baseline:.0f} SEK, Battery {battery:.0f} SEK, Savings {savings:.0f} SEK")
        
        # Battery utilization
        total_charged = df['battery_charge'].sum()
        total_discharged = df['battery_discharge'].sum()
        total_purchased = df['grid_purchase'].sum()
        total_sold = df['grid_sale'].sum()
        
        print(f"\nBattery & Grid Performance:")
        print(f"Total energy charged to battery: {total_charged:.0f} kWh")
        print(f"Total energy discharged from battery: {total_discharged:.0f} kWh")
        print(f"Total energy purchased from grid: {total_purchased:.0f} kWh")
        print(f"Total energy sold to grid: {total_sold:.0f} kWh")
        print(f"Net grid interaction: {total_purchased - total_sold:.0f} kWh (+ = net purchase)")
        print(f"Round-trip efficiency: {(total_discharged/total_charged)*100:.1f}%" if total_charged > 0 else "No battery usage")
        
        # Price analysis
        if total_purchased > 0:
            avg_purchase_price = (df['grid_purchase'] * (df['price_sek_kwh'] + 0.685)).sum() / total_purchased
            print(f"Average purchase price (incl. transfer): {avg_purchase_price:.3f} SEK/kWh")
        
        if total_sold > 0:
            avg_sale_price = (df['grid_sale'] * df['price_sek_kwh']).sum() / total_sold
            print(f"Average sale price (energy only): {avg_sale_price:.3f} SEK/kWh")
        
        return monthly_savings
    
    def create_summary_selling(self, df, baseline_cost, battery_cost):
        """Create a summary for the grid selling scenario"""
        print("\n=== DETAILED RESULTS SUMMARY - GRID SELLING MODEL (SEK) ===")
        
        # Basic statistics
        annual_savings = baseline_cost - battery_cost
        
        # Monthly analysis
        monthly_baseline = df.groupby(df.index.month)['baseline_selling_cost'].sum()
        monthly_battery = df.groupby(df.index.month)['battery_selling_cost'].sum()
        monthly_savings = monthly_baseline - monthly_battery
        
        print("\nMonthly Savings Breakdown (SEK):")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num, month_name in enumerate(months, 1):
            if month_num in monthly_savings.index:
                savings = monthly_savings[month_num]
                baseline = monthly_baseline[month_num]
                battery = monthly_battery[month_num]
                print(f"{month_name}: Baseline {baseline:.0f} SEK, Battery {battery:.0f} SEK, Savings {savings:.0f} SEK")
        
        # Battery utilization
        total_charged = df['battery_charge'].sum()
        total_discharged = df['battery_discharge'].sum()
        total_purchased = df['grid_purchase'].sum()
        total_sold = df['grid_sale'].sum()
        
        print(f"\nBattery & Grid Performance:")
        print(f"Total energy charged to battery: {total_charged:.0f} kWh")
        print(f"Total energy discharged from battery: {total_discharged:.0f} kWh")
        print(f"Total energy purchased from grid: {total_purchased:.0f} kWh")
        print(f"Total energy sold to grid: {total_sold:.0f} kWh")
        print(f"Net grid interaction: {total_purchased - total_sold:.0f} kWh (+ = net purchase)")
        print(f"Round-trip efficiency: {(total_discharged/total_charged)*100:.1f}%" if total_charged > 0 else "No battery usage")
        
        # Price analysis
        if total_purchased > 0:
            avg_purchase_price = (df['grid_purchase'] * (df['price_sek_kwh'] + 0.685)).sum() / total_purchased
            print(f"Average purchase price (incl. transfer): {avg_purchase_price:.3f} SEK/kWh")
        
        if total_sold > 0:
            avg_sale_price = (df['grid_sale'] * df['price_sek_kwh']).sum() / total_sold
            print(f"Average sale price (energy only): {avg_sale_price:.3f} SEK/kWh")
        
        return monthly_savings
    
    def create_visualization_sek(self, df):
        """Create visualization with SEK currency"""
        print("\nCreating SEK-based visualization...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('10kWh Battery Storage Analysis - 50,000 SEK System Cost', 
                     fontsize=16, fontweight='bold')
        
        # 1. Price distribution in SEK
        ax1 = axes[0, 0]
        ax1.hist(df['price_sek_kwh'], bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Electricity Price (SEK/kWh)')
        ax1.set_ylabel('Hours')
        ax1.set_title('Electricity Price Distribution')
        ax1.axvline(df['price_sek_kwh'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["price_sek_kwh"].mean():.3f} SEK/kWh')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Daily energy profile
        ax2 = axes[0, 1]
        daily_consumption = df['consumption_kwh'].groupby(df.index.hour).mean()
        daily_production = df['production_kwh'].groupby(df.index.hour).mean()
        
        hours = range(24)
        ax2.plot(hours, daily_consumption, 'b-', linewidth=2, label='Consumption')
        ax2.plot(hours, daily_production, 'orange', linewidth=2, label='Solar Production')
        ax2.fill_between(hours, daily_consumption, alpha=0.3, color='blue')
        ax2.fill_between(hours, daily_production, alpha=0.3, color='orange')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Energy (kWh)')
        ax2.set_title('Average Daily Energy Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 4))
        
        # 3. Battery operation sample week
        sample_week = df.iloc[24*30:24*37]  # Week in February
        ax3 = axes[0, 2]
        ax3.plot(sample_week.index, sample_week['battery_soc'], 'g-', linewidth=2, label='Battery SOC (kWh)')
        ax3.bar(sample_week.index, sample_week['battery_charge'], alpha=0.6, color='green', 
                width=0.02, label='Charging')
        ax3.bar(sample_week.index, -sample_week['battery_discharge'], alpha=0.6, color='orange', 
                width=0.02, label='Discharging')
        ax3.set_ylabel('Energy (kWh)')
        ax3.set_title('Battery Operation (Sample Week)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly cost comparison
        ax4 = axes[1, 0]
        monthly_baseline = df.groupby(df.index.month)['baseline_cost'].sum()
        monthly_battery = df.groupby(df.index.month)['battery_cost'].sum()
        
        months_available = list(monthly_baseline.index)
        x = np.arange(len(months_available))
        width = 0.35
        
        ax4.bar(x - width/2, monthly_baseline.values, width, label='Baseline', alpha=0.8)
        ax4.bar(x + width/2, monthly_battery.values, width, label='With Battery', alpha=0.8)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Cost (SEK)')
        ax4.set_title('Monthly Electricity Costs')
        ax4.set_xticks(x)
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax4.set_xticklabels([month_labels[m-1] for m in months_available])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Daily savings distribution
        ax5 = axes[1, 1]
        daily_baseline = df.groupby(df.index.date)['baseline_cost'].sum()
        daily_battery = df.groupby(df.index.date)['battery_cost'].sum()
        daily_savings = daily_baseline - daily_battery
        
        ax5.hist(daily_savings, bins=50, alpha=0.7, edgecolor='black')
        ax5.axvline(daily_savings.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {daily_savings.mean():.1f} SEK')
        ax5.set_xlabel('Daily Savings (SEK)')
        ax5.set_ylabel('Number of Days')
        ax5.set_title('Distribution of Daily Savings')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Economic summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate key statistics
        total_savings = (df['baseline_cost'].sum() - df['battery_cost'].sum())
        payback_years = self.system_cost_sek / total_savings if total_savings > 0 else float('inf')
        roi = (total_savings / self.system_cost_sek) * 100 if total_savings > 0 else 0
        
        total_solar = df['production_kwh'].sum()
        total_wasted = df['excess_solar_wasted'].sum()
        solar_utilization = (total_solar - total_wasted) / total_solar * 100 if total_solar > 0 else 0
        
        summary_text = f"""
ECONOMIC SUMMARY (No-Sell Model)

10kWh Battery System:
• System Cost: {self.system_cost_sek:,} SEK
• Annual Savings: {total_savings:.0f} SEK
• Payback Period: {payback_years:.1f} years
• ROI: {roi:.1f}% per year

House Profile:
• Annual Consumption: {df['consumption_kwh'].sum():.0f} kWh
• Annual Solar Production: {total_solar:.0f} kWh
• Solar Utilization: {solar_utilization:.1f}%
• Avg Cost (energy + transfer): {(df['price_sek_kwh'].mean() + 0.685):.3f} SEK/kWh

Conclusion:
{"✅ VIABLE" if payback_years <= 12 else "❌ NOT VIABLE"}

Battery Performance:
• Charged: {df['battery_charge'].sum():.0f} kWh/year
• Discharged: {df['battery_discharge'].sum():.0f} kWh/year
• Efficiency: {(df['battery_discharge'].sum()/df['battery_charge'].sum()*100):.1f}%
• Excess Solar Wasted: {total_wasted:.0f} kWh/year
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('battery_analysis_10kWh_50kSEK.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()
        
        print("Analysis charts saved as 'battery_analysis_10kWh_50kSEK.png'")
    
    def run_analysis(self):
        """Run the complete battery analysis with three scenarios"""
        print("=== 10kWh BATTERY STORAGE ANALYSIS (50,000 SEK) ===")
        print("=== THREE SCENARIOS COMPARISON ===")
        print(f"Exchange rate used: 1 EUR = {self.eur_to_sek} SEK")
        print(f"Power transfer cost: 0.685 SEK/kWh added to all grid purchases\n")
        
        # Load data
        df = self.load_data()
        
        if len(df) < 100:  # Too little data
            print("❌ Insufficient data for reliable analysis")
            return None, 0, 0
        
        # SCENARIO 1: No-sell model (no battery)
        print("\n" + "="*60)
        print("SCENARIO 1: NO-SELL MODEL (excess solar wasted)")
        print("="*60)
        df1, baseline_nosell_cost = self.baseline_scenario(df.copy())
        df1, battery_nosell_cost = self.battery_scenario(df1)
        savings_nosell = baseline_nosell_cost - battery_nosell_cost
        
        # SCENARIO 2: Grid selling model (no battery)
        print("\n" + "="*60)
        print("SCENARIO 2: GRID SELLING MODEL (sell at energy price)")
        print("="*60)
        df2, baseline_selling_cost = self.baseline_scenario_with_selling(df.copy())
        df2, battery_selling_cost = self.battery_scenario_with_selling(df2)
        savings_selling = baseline_selling_cost - battery_selling_cost
        
        # Comparison summary
        print("\n" + "="*60)
        print("THREE SCENARIO COMPARISON SUMMARY")
        print("="*60)
        
        print(f"\nSCENARIO 1 - NO-SELL MODEL:")
        print(f"  Baseline cost: {baseline_nosell_cost:,.0f} SEK")
        print(f"  Battery cost:  {battery_nosell_cost:,.0f} SEK")
        print(f"  Annual savings: {savings_nosell:,.0f} SEK")
        print(f"  Payback: {self.system_cost_sek/savings_nosell:.1f} years" if savings_nosell > 0 else "  No savings")
        
        print(f"\nSCENARIO 2 - GRID SELLING MODEL:")
        print(f"  Baseline cost: {baseline_selling_cost:,.0f} SEK")
        print(f"  Battery cost:  {battery_selling_cost:,.0f} SEK")
        print(f"  Annual savings: {savings_selling:,.0f} SEK")
        print(f"  Payback: {self.system_cost_sek/savings_selling:.1f} years" if savings_selling > 0 else "  No savings")
        
        print(f"\nSCENARIO COMPARISON:")
        if savings_selling > savings_nosell:
            improvement = savings_selling - savings_nosell
            print(f"  Grid selling improves savings by {improvement:,.0f} SEK ({improvement/savings_nosell*100:.1f}%)")
        else:
            decline = savings_nosell - savings_selling
            print(f"  Grid selling reduces savings by {decline:,.0f} SEK ({decline/savings_nosell*100:.1f}%)")
        
        # Use the best scenario for detailed analysis
        if savings_selling > savings_nosell:
            print(f"\n✅ Grid selling model shows better economics - using for detailed analysis")
            best_df, best_savings, best_baseline, best_battery = df2, savings_selling, baseline_selling_cost, battery_selling_cost
            model_name = "Grid Selling"
        else:
            print(f"\n✅ No-sell model shows better economics - using for detailed analysis")
            best_df, best_savings, best_baseline, best_battery = df1, savings_nosell, baseline_nosell_cost, battery_nosell_cost
            model_name = "No-Sell"
        
        # Detailed analysis of best scenario
        print(f"\n" + "="*60)
        print(f"DETAILED ANALYSIS - {model_name.upper()} MODEL")
        print("="*60)
        
        savings, system_cost, npv = self.analyze_savings(best_baseline, best_battery)
        
        # Create summary for best scenario
        if model_name == "Grid Selling":
            monthly_savings = self.create_summary_selling(best_df, best_baseline, best_battery)
        else:
            monthly_savings = self.create_summary_sek(best_df, best_baseline, best_battery)
        
        # Create visualization
        self.create_visualization_sek(best_df)
        
        return best_df, best_savings, system_cost

if __name__ == "__main__":
    # Run analysis with 10kWh battery costing 50,000 SEK
    analyzer = BatteryAnalysisSEK(battery_capacity_kwh=10, max_power_kw=10, 
                                  efficiency=0.95, system_cost_sek=50000)
    result = analyzer.run_analysis()
    
    if result and result[0] is not None:
        df, savings, system_cost = result
        
        print("\n=== FINAL ASSESSMENT ===")
        if savings > 0:
            payback = system_cost / savings
            roi = (savings / system_cost) * 100
            
            print(f"Annual savings: {savings:,.0f} SEK")
            print(f"System cost: {system_cost:,} SEK")
            print(f"Payback period: {payback:.1f} years")
            print(f"Return on investment: {roi:.1f}% per year")
            
            if payback <= 8:
                print("✅ RECOMMENDED: Good investment with attractive payback")
            elif payback <= 12:
                print("⚠️  MARGINAL: Consider future electricity price trends")
            else:
                print("❌ NOT RECOMMENDED: Payback period too long")
        else:
            print("❌ NOT RECOMMENDED: Battery would increase costs")
            
        print(f"\nBased on {len(df)} hours of data from 2024")
        print("Exchange rate: 1 EUR = 11.5 SEK")
        print("\nNote: Analysis assumes perfect price forecasting and optimal operation.")
    
    print("\n=== ANALYSIS COMPLETE ===")