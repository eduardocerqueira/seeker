#date: 2022-08-01T17:09:58Z
#url: https://api.github.com/gists/e5f7c3df5e83cb02dfdc91cd63d25aae
#owner: https://api.github.com/users/jcubiro

## Define Global Variables
window=30
credentials_file_location="your-creds.json"
slack_client = WebClient("your-bot-user-oauth")
url="sc-domain:your-url.com"
ga_view_id="view-id"
ga_metrics=['ga:sessions', 'ga:newUsers', 'ga:avgTimeOnPage', 'ga:goal18Completions', 'ga:goal18ConversionRate']
ga_dimensions=['ga:landingPagePath']

 ## Specify KPIs for each source. 
gsc_kpis=["impressions","clicks"]
ga_kpis=['newUsers', 'goal18Completions', 'goal18ConversionRate']

## Assign Start & End Dates based on the specified Window. 
start=datetime.today()-timedelta(days=window+2) 
main_start=start.strftime("%Y-%m-%d") 
end=datetime.today()-timedelta(days=3) 
main_end=end.strftime("%Y-%m-%d") 
previous_start=(start-timedelta(window)).strftime("%Y-%m-%d") 
previous_end=(end-timedelta(window)).strftime("%Y-%m-%d")

## Initialize Reporting Services
ga_service=initialize__delegated_google_reporting('ga_service',credentials_file_location)
gsc_service=initialize__delegated_google_reporting("gsc_service",credentials_file_location)


## Fetch Search Console Reports
gsc_last=get_search_console_report(gsc_service,main_start,main_end,url)
gsc_previous=get_search_console_report(gsc_service,previous_start,previous_end,url)


## Fetch Google Analytics Reports
ga_last=get_analytics_report(ga_service,ga_view_id,main_start,main_end,ga_metrics,ga_dimensions)
ga_previous=get_analytics_report(ga_service,ga_view_id,previous_start,previous_end,ga_metrics,ga_dimensions)

# Create Search Console Comparison Data Frame
gsc_comparison_model=create_comparison(gsc_last,gsc_previous,gsc_kpis,join="page")

# Create Google Analytics Comparison Data Frame
ga_comparison_model=create_comparison(ga_last,ga_previous,ga_kpis,join="landingPagePath")

## Create an Empty DataFrame and Append the Highlights around Clicks
clicks_report=pd.DataFrame()
clicks_report=clicks_report.append(gsc_comparison_model.nsmallest(6,"clicks_abs_diff"))
clicks_report=clicks_report.append(gsc_comparison_model.nlargest(6,"clicks_abs_diff"))

## Create an Empty DataFrame and Append the Highlights around the Sign UPs in this example is "goal18Completions"
goals_report=pd.DataFrame()
goals_report=goals_report.append(ga_comparison_model.nsmallest(6,"goal18Completions_abs_diff"))
goals_report=goals_report.append(ga_comparison_model.nlargest(6,"goal18Completions_abs_diff"))

## Create a written report around GA Data
highlights_goals=[]
for index, row in goals_report.iterrows():
    highlight=f"""   -- {row['landingPagePath']} ({row["goal18Completions_last"]}  vs. {row["goal18Completions_previous"]} // {row["goal18Completions_rel_diff"]} vs. previous {window} days)\n    """
    highlights_goals.append(highlight)

## Create a Written report around GSC Data
highlights_clicks=[]
for index, row in clicks_report.iterrows():
    highlight=f"""   -- {row['page']} ({row['clicks_last']}  vs. {row['clicks_previous']} // {row["clicks_rel_diff"]} vs. previous {window} days)\n    """
    highlights_clicks.append(highlight)


if highlights_clicks:
    highlights_clicks=''.join(highlights_clicks)
if highlights_goals:
    highlights_goals=''.join(highlights_goals)

# SEND MESSAGE
response = slack_client.chat_postMessage(channel="the-channel-id-you-copied", text=f"""
Hi, these are the highlights of the last {window} days ({main_start} to {main_end}) for your client (client name).:chart_with_upwards_trend:\n
*:desktop_computer:Clicks Report*
{highlights_clicks}\n\n
*:face_with_monocle:SUs Report*
{highlights_goals}""")