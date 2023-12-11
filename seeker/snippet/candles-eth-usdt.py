#date: 2023-12-11T16:57:13Z
#url: https://api.github.com/gists/4cf53c0815254776bd3fd0e2caee4ce3
#owner: https://api.github.com/users/david-hummingbot

def format_status(self) -> str:  
    if not self.ready_to_trade:
        return "Market connectors are not ready."

    lines = ["\n############################################ Market Data ############################################\n"]
    if self.eth_1h_candles.is_ready:
        candles_df = self.eth_1h_candles.candles_df
        # Format timestamp
        candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"], unit="ms").dt.strftime('%Y-%m-%d %H:%M:%S')
        # Select relevant columns for display
        display_columns = ["timestamp", "open", "high", "low", "close"]
        formatted_df = candles_df[display_columns].tail()  # Display last few records
        # Format the data frame as a string for display
        lines.append("One-hour Candles for ETH-USDT:")
        lines.append(formatted_df.to_string(index=False))
        lines.append("\n-----------------------------------------------------------------------------------------------------------\n")
    else:
        lines.append("  One-hour candle data is not ready.")

    return "\n".join(lines)