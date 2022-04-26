#date: 2022-04-26T16:57:38Z
#url: https://api.github.com/gists/99418c817daf9e82f99d1fb00424a44b
#owner: https://api.github.com/users/notgonnadie

import streamlit as st
from datetime import datetime
import random
import pandas as pd
import numpy as np
import base64
from bokeh.models import CategoricalTicker

st.set_page_config(
    page_title = "Wanna eat?",
    page_icon = "random",
    menu_items = {
        'Get Help' : 'https://www.youtube.com/watch?v=cGzujGfXrlI&ab_channel=RoomieOfficial',
        'Report a Bug' : 'https://suicidepreventionlifeline.org',
        'About' : "# A common cause of suicidal thoughts among coders is a bug report. Please be careful, debugging is everyone's worst nightmare. :D"
    }
)

cuisine = st.sidebar.selectbox("What is your choice today?",
                     ["", "Italian", "Asian", "Cuban", "Desserts & Coffee", "Fastfood", "Show me all!"])


restuarant_list = ['Al Forno Neapolitan Wood Fire Pizza (Italian) 10674 SW 24th St, Miami, FL 33165. 📞(305) 456-3699', "Giovanni's (Italian) 8522 SW 8th St, Miami, FL 33144. 📞(305) 262-6966",
                   'La Nonna Italian Restaurant (Italian) 117 SW 107th Ave, Miami, FL 33174. 📞(305) 228-4900',
                          '107 Taste (Asian) 1679 SW 107th Ave, Miami, FL 33165. 📞(786) 717-6378', 'Kentucky Fried Chicken (Fastfood) 949 SW 107th Ave, Miami, FL 33174. 📞(305) 221-9169', 'Burger King (Fastfood) 955 SW 107th Ave, Miami, FL 33174. 📞(305) 553-7244', 'Chick-fil-A (Fastfood) 885 SW 109th Ave PG-5/Retail, Miami, FL 33199. 📞(305) 348-7797'
                          , 'Rio Cristal (Cuban) 10481 SW 40th St, Miami, FL 33165. 📞(305) 225-0200', "Sergio's Restaurant (Cuban) 9330 SW 40th St, Miami, FL 33165.📞(305) 552-9626 ", 'El Rinconcito Latino (Cuban) 9872 SW 40th St, Miami, FL 33165. 📞(305) 223-0200',
                        'Nightowl Cookie (Coffe & Desserts) 10534 SW 8th St, Miami, FL 33174. 📞(786) 360-5011', 'Starbucks Coffee (Coffee & Desserts) 11212 SW 11th St, Miami, FL 33199. www.starbucks.com', 'Good Chef Restaurant (Asian) 113 SW 107th Ave, Miami, FL 33174. 📞(305) 227-3111',
                        'Changs Chinese Restaurant (Asian) 1311 SW 107th Ave, Miami, FL 33174. 📞(305) 221-8104', 'Jasmine Suhsi & Thai Cuisine (Asian) 1429 SW 107th Ave, Miami, FL 33174. 📞(305) 207-9700', 'Long Gong Chinese Restaurant (Asian) 11920 SW 8th St, Miami, FL 33184. 📞(305) 553-4644'
                        , 'Kam Wow (Asian) "12895 SW 42nd St, Miami, FL 33175. 📞(305) 229-2888', 'Atelier Minnier Bird Road, French Bakery & Cafe (Coffee & Desserts) 9825 SW 40th St, Miami, FL 33165. 📞(786) 452-7780']


kek = st.sidebar.button("Pick for me I'm lazy!")
if kek:
        st.write("🥁Dr-dr-dr-dr...🥁")
        st.write("🥁Dr-dr-dr-dr...🥁")
        st.write("🎉Ta-da-a!🎉")
        st.write(random.choice(restuarant_list))
        st.info('Please, call to restaurant to check working hours, we can not guarantee it is open now.')
        st.balloons()





if cuisine == "" and kek == False:
    st.title("Hungry? Don't know good places near MMC FIU?")
    st.write(" ← Choose a cuisine on a side menu, we picked good restaurants for you.")
    st.image("https://images.unsplash.com/photo-1608096275140-f190f1fb41b8?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1169&q=80")
if cuisine == "Italian" or cuisine == "Asian" or cuisine == "Cuban" or cuisine == "Desserts & Coffee" or cuisine == "Fastfood":
    budget = st.sidebar.radio('What is your budget?',
                              ["$-$$", "$$-$$$"])
    date_to_eat = st.sidebar.date_input("When do you want to eat?")
    week_day = date_to_eat.isoweekday()
    time_to_eat = st.sidebar.time_input("At what time do you want to eat?")
    distance = st.sidebar.slider('How far away from MMC are you looking for?', 0, 4)



if cuisine == "Italian":
    if budget == "$-$$":
            it_cheap1, it_cheap2, it_cheap3 = st.columns(3)

            with it_cheap1:
                st.header("Al Forno Neapolitan" + " Wood Fire Pizza")
                st.write("0.5 miles from MMC")
                st.write("www.alfornomiami.com")
                st.write("10674 SW 24th St, Miami, FL 33165")
                st.write("📞(305) 456-3699")
                if (time_to_eat.hour < 10) or (time_to_eat.hour >= 23):
                    st.warning("Sorry, we are closed!")


            if distance > 1:

                with it_cheap2:
                    st.header("La Nonna Italian Restaurant")
                    st.write("1.4 miles from MMC")
                    st.write("117 SW 107th Ave, Miami, FL 33174")
                    st.write("📞(305) 228-4900")
                    if week_day == 1:
                        st.warning("Sorry, we are closed!")
                    if (week_day == 6) or (week_day == 7):
                        if (time_to_eat.hour < 12) or (time_to_eat.hour >= 22):
                            st.warning("Sorry, we are closed!")
                        elif (time_to_eat.hour < 11) or (time_to_eat.hour >= 22):
                            st.warning("Sorry, we are closed!")

            if distance > 3:

                with it_cheap3:
                    st.header("Giovanni's")
                    st.write("3.1 miles from MMC")
                    st.write("www.giovannisitalianristorante.com")
                    st.write("8522 SW 8th St, Miami, FL 33144")
                    st.write("📞 (305) 262-6966")
                    if (week_day == 5) or (week_day == 6):
                        if(time_to_eat.hour < 11) or (time_to_eat.hour >= 23):
                            st.warning("Sorry, we are closed!")
                    else:
                        if(time_to_eat.hour < 11) or (time_to_eat.hour >= 22):
                            st.warning("Sorry, we are closed!")


    else:
        st.error("No match found within 4 miles range from MMC.")
        st.image("https://images.unsplash.com/photo-1433162653888-a571db5ccccf?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80", caption='We   are sorry :( Here is a doggo.')


if cuisine == "Asian":
    if budget == "$-$$":
        as_cheap1, as_cheap2, as_cheap3 = st.columns(3)

        if distance > 1:

            with as_cheap1:
                st.header("Good Chef Restaurant")
                st.write("1.1 miles from MMC")
                st.write("www.goodchefrestaurant.com")
                st.write("113 SW 107th Ave, Miami, FL 33174")
                st.write("📞(305) 227-3111")
                if (((time_to_eat.hour < 11) or (time_to_eat.hour <= 11) and (time_to_eat.minute < 30)) or (time_to_eat.hour >= 22)):
                    st.warning("Sorry, we are closed!")
        else:
            st.error("Sorry, nothing found, try to change distance from MMC!")
        if distance > 1:
            with as_cheap2:
                st.header("107 Taste")
                st.write("1.2 miles from MMC")
                st.write("www.107taste.com")
                st.write("1679 SW 107th Ave, Miami, FL 33165")
                st.write("📞(786) 717-6378")
                if (week_day == 5) or (week_day == 6):
                    if ((time_to_eat.hour < 12) or (time_to_eat.hour > 22) or ((time_to_eat.hour >= 22) and (time_to_eat.minute >= 30))):
                        st.warning("Sorry, we are closed!")
                else:
                    if (week_day == 7):
                        if (time_to_eat.hour < 12) or (time_to_eat.hour >= 22):
                            st.warning("Sorry, we are closed!")
                    else:
                        if ((time_to_eat.hour < 11) or((time_to_eat.hour <= 11) and (time_to_eat.minute < 30)) or (time_to_eat.hour >= 22)):
                            st.warning("Sorry, we are closed!")
        if distance > 1:
            with as_cheap3:
                st.header("Chang's Chinese Restaurant")
                st.write("1.7 miles from MMC")
                st.write("www.changsmiami.com")
                st.write("1311 SW 107th Ave, Miami, FL 33174")
                st.write("📞(305) 221-8104")
                if (week_day == 5) or (week_day == 6):
                    if (time_to_eat.hour < 11) or (time_to_eat.hour >= 23):
                            st.warning("Sorry, we are closed!")
                else:
                    if (week_day == 2):
                        st.warning("Sorry, we are closed!")
                    else:
                        if (time_to_eat.hour < 11) or (time_to_eat.hour >= 22):
                            st.warning("Sorry, we are closed!")


    else:
        as_exp1, as_exp2, as_exp3 = st.columns(3)


        if distance > 1:
            with as_exp1:
                st.header("Jasmine Sushi & Thai Cuisine")
                st.write("1.7 miles from MMC")
                st.write("www.jasminesushi.com")
                st.write("1429 SW 107th Ave, Miami, FL 33174")
                st.write("📞(305) 207-9700")
                if (week_day == 1):
                    st.warning("Sorry, we are closed!")
                else:
                    if (time_to_eat.hour < 12) or (time_to_eat.hour >= 22):
                        st.warning("Sorry, we are closed!")

        else:
            st.error("Sorry, nothing found, try to change distance from MMC!")



        if distance > 1:
            with as_exp2:
                st.header("Long Gong Chinese Restaurant")
                st.write("1.7 miles from MMC")
                st.write("www.longgongchinese.com")
                st.write("11920 SW 8th St, Miami, FL 33184")
                st.write("📞(305) 553-4644")
                if (week_day == 3):
                    st.warning("Sorry, we are closed!")
                else:
                    if (time_to_eat.hour < 12) or (time_to_eat.hour >= 21):
                        st.warning("Sorry, we are closed!")


        if distance > 3:
            with as_exp3:
                st.header("Kam Wow")
                st.write("3.7 miles from MMC")
                st.write("www.kamwahmiami.com")
                st.write("12895 SW 42nd St, Miami, FL 33175")
                st.write("📞(305) 229-2888")
                if (week_day == 1):
                    st.warning("Sorry, we are closed!")
                else:
                    if (time_to_eat.hour < 13) or (time_to_eat.hour >= 21):
                        st.warning("Sorry, we are closed!")



if cuisine == "Cuban":
    if budget == "$-$$":
        cb_cheap1, cb_cheap2, cb_cheap3 = st.columns(3)


        if distance > 3:
            with cb_cheap1:
                st.header("El Rinconcito Latino")
                st.write("3.3 miles from MMC")
                st.write("www.rinconcito.miami")
                st.write("9872 SW 40th St, Miami, FL 33165")
                st.write("📞(305) 223-0200")
                if (week_day == 5) or (week_day == 6):
                    if (time_to_eat.hour <= 7) or (time_to_eat.hour >= 23):
                        st.warning("Sorry, we are closed!")
                else:
                    if (time_to_eat.hour <= 7) or (time_to_eat.hour >= 22):
                        st.warning("Sorry, we are closed!")
        else:
            st.error("Sorry, nothing found, try to change distance from MMC!")


        if distance > 3:
            with cb_cheap2:
                st.header("Rio Cristal")
                st.write("3.6 miles from MMC")
                st.write("www.riocristalmiami.com")
                st.write("10481 SW 40th St, Miami, FL 33165")
                st.write("📞(305) 225-0200")
                if (week_day == 1):
                    st.warning("Sorry, we are closed!")
                else:
                    if (week_day == 5) or (week_day == 6):
                        if (time_to_eat.hour < 12) or (time_to_eat.hour >= 21):
                            st.warning("Sorry, we are closed!")
                    else:
                        if (time_to_eat.hour<12) or (time_to_eat.hour >= 20):
                            st.warning("Sorry, we are closed!")

        if distance == 4:
            with cb_cheap3:
                st.header("Sergio's Restaurant")
                st.write("4.0 miles from MMC")
                st.write("www.sergios.com")
                st.write("9330 SW 40th St, Miami, FL 33165")
                st.write("📞(305) 552-9626")
                if (week_day == 5) or (week_day == 6):
                    if (time_to_eat.hour < 6) or (time_to_eat.hour >= 23):
                        st.warning("Sorry, we are closed!")
                else:
                    if (time_to_eat.hour < 6) or (time_to_eat.hour >= 22):
                        st.warning("Sorry, we are closed!")


    else:
        st.error("No match found within 4 miles range from MMC.")
        st.image("https://images.unsplash.com/photo-1518770352423-dce09a3d3307?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1074&q=80",caption="Hey, don't give up! Try something else!")



if cuisine == "Desserts & Coffee":
    if budget == "$-$$":
        dc_cheap1, dc_cheap2, dc_cheap3 = st.columns(3)


        with dc_cheap1:
            st.header("Starbucks Coffee")
            st.write("0.3 miles from MMC")
            st.write("www.starbucks.com")
            st.write("11212 SW 11th St, Miami, FL 33199")
            if (week_day == 6) or (week_day == 7):
                st.warning("Sorry, we are closed!")
            else:
                if (time_to_eat.hour < 8) or (time_to_eat.hour >= 15):
                    st.warning("Sorry, we are closed!")


        if distance > 1:
            with dc_cheap2:
                st.header("Nightowl Cookie")
                st.write("1.4 miles from MMC")
                st.write("www.nightowlcookieco.com")
                st.write("10534 SW 8th St, Miami, FL 33174")
                st.write("📞(786) 360-5011")
                if (time_to_eat.hour >= 2) and (time_to_eat.hour < 11):
                    st.warning("Sorry, we are closed!")

        if distance > 3:
            with dc_cheap3:
                st.header("Atelier Monnier Bird Road, French Bakery & Cafe")
                st.write("3.5 miles from MMC")
                st.write("www.ateliermonnier.com")
                st.write("9825 SW 40th St, Miami, FL 33165")
                st.write("📞(786) 452-7780")
                if (week_day == 1):
                    st.warning("Sorry, we are closed!")
                else:
                    if (week_day == 3) or (week_day == 4) or (week_day == 5):
                        if ((time_to_eat.hour < 8) or (time_to_eat.hour >= 15) or ((time_to_eat.hour >= 15) and (time_to_eat.minute >= 30))):
                            st.warning("Sorry, we are closed!")
                    else:
                        if (week_day == 6):
                            if ((time_to_eat.hour < 7) or ((time_to_eat.hour < 7) and (time_to_eat.minute < 30)) or (time_to_eat.hour >= 16)):
                                st.warning("Sorry, we are closed!")
                        else:
                            if (week_day == 2):
                                if (time_to_eat.hour < 8) or (time_to_eat.hour >= 12):
                                    st.warning("Sorry, we are closed!")
                            else:
                                if (week_day == 7):
                                    if ((time_to_eat.hour < 7) or (time_to_eat.hour >= 12) or ((time_to_eat.hour < 7) and (time_to_eat.minute < 30))):
                                        st.warning("Sorry, we are closed!")
    else:
        st.error("No match found within 4 miles range from MMC.")
        st.image("https://images.unsplash.com/photo-1525785967371-87ba44b3e6cf?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1173&q=80", caption='Meow?')

if cuisine == "Fastfood":
    if budget == "$-$$":
        ff_cheap1, ff_cheap2, ff_cheap3 = st.columns(3)

        with ff_cheap1:
            st.header("Kentucky Fried Chicken")
            st.write("0.5 miles from MMC")
            st.write("www.kfc.com")
            st.write("949 SW 107th Ave, Miami, FL 33174")
            st.write("📞(305) 221-9169")
            if (time_to_eat.hour < 10) or (time_to_eat.hour >= 24):
                st.warning("Sorry, we are closed!")

        with ff_cheap2:
            st.header("Burger King")
            st.write("0.5 miles from MMC")
            st.write("www.bk.com")
            st.write("955 SW 107th Ave, Miami, FL 33174")
            st.write("📞(305) 553-7244")
            if (week_day == 5) or (week_day == 6):
                if (time_to_eat.hour<6) or (time_to_eat.hour >= 14):
                    st.warning("Sorry, we are closed!")
            else:
                if (time_to_eat.hour < 6) or (time_to_eat.hour >= 24):
                    st.warning("Sorry, we are closed!")

        with ff_cheap3:
            st.header("Chick-fil-A")
            st.write("0.1  miles from MMC")
            st.write("www.chick-fil-a.com")
            st.write("885 SW 109th Ave PG-5/Retail, Miami, FL 33199")
            st.write("📞(305) 348-7797")
            if (week_day == 6) or (week_day == 7):
                st.warning("Sorry, we are closed!")
            else:
                if (week_day == 5):
                    if (time_to_eat.hour < 7) or (time_to_eat.hour >= 17):
                        st.warning("Sorry, we are closed!")
                else:
                    if ((time_to_eat.hour < 7) or (time_to_eat.hour > 20) or ((time_to_eat.hour >= 20) and (time_to_eat.minute >= 30))):
                        st.warning("Sorry, we are closed!")
    else:
        if distance > 3:
            st.header("Flanigan's Seafood Bar and Grill")
            st.write("3.6  miles from MMC")
            st.write("www.flanigans.net")
            st.write("9857 SW 40th St, Miami, FL 33165")
            st.write("📞(305) 207-7427")
            if (week_day == 5) or (week_day == 6):
                if ((1 < time_to_eat.hour < 11) or ((11>=time_to_eat.hour >= 1) and (time_to_eat.minute >= 30))):
                    st.warning("Sorry, we are closed!")
            else:
                if ((0<time_to_eat.hour < 11) or ((11>=time_to_eat.hour >= 0) and (time_to_eat.minute >= 30))):
                    st.warning("Sorry, we are closed!")
        else:
            st.error("Sorry, nothing found, try to change distance from MMC!")
if cuisine == "Show me all!":
    df = pd.read_csv("work.csv")
    st.dataframe(df, width=900, height=600)
    Score = pd.DataFrame(
        {"List of Restaurants": ["Kentucky Fried Chicken", "Chick-fil-A",
                                 "Burger King", "Starbucks Coffee", "Sergio's Restaurant",
                                  "Rio Cristal", "La Nonna Italian Restaurant",
                                  "Nightowl Cookie", "Chang's Chinese Restaurant", "Flanigan's Seafood Bar and Grill", "El Rinconcito Latino",
                                 "Long Gong Chinese Restaurant", "Kam Wow", "Jasmine Sushi & Thai Cuisine", "107 Taste",
                                 "Аtelier Monnier Bird Road French Bakery & Cafe", "Giovanni's", "Al Forno Neapolitan Wood Fire Pizza",
                                 "Good Chef Restaurant"],
         "rating_Score": [3.6, 3.6, 3.7, 3.8, 4.0, 4.3, 4.3, 4.4, 4.4, 4.4, 4.4, 4.4, 4.5, 4.6, 4.6, 4.8, 4.8, 4.8, 4.9]})

    from bokeh.plotting import figure

    x = Score['List of Restaurants']
    y = Score['rating_Score']
    p = figure(title='Restaurant Ratings', y_range=x.values, x_axis_label='Rating', y_axis_label='Restaurants')

    p.yaxis.ticker = CategoricalTicker()
    p.hbar(right=y, y=x, height=0.25)
    st.bokeh_chart(p)

    st.title("This chart shows how busy restaurants on days of the week.")
    st.write("Scale from 0-5. 5 is the highest definition of busyness.")
    st.write("0 - Most of the restaurants are closed.")
    st.write("1 - Most of the restaurants are not busy.")
    st.write("5 - Most of the restaurants are extremely busy.")
    df = pd.DataFrame(pd.read_csv('okay.csv'), columns=["Al Forno Neapolitan Wood Fire Pizza", "La Nonna Italian Restaurant",
                                 "Giovanni's", "Good Chef Restaurant", " 107 Taste",
                                  "Chang's Chinese Restaurant", "Jasmine Sushi & Thai Cuisine",
                                  "Long Gong Chinese Restaurant", "Kam Wow", " El Rinconcito Latino", "Rio Cristal",
                                 "Sergio's Restaurant", "Starbucks Coffee", "Nightowl Cookie", "Atelier Monnier Bird Road, French Bakery & Cafe",
                                 " Kentucky Fried Chicken", "Burger King"])
    st.line_chart(df)
    st.write("X-axis is day of the week. 0 is Monday, 6 is Sunday.")

if cuisine != "":
    restaurants_map = st.checkbox("See all the FIU MMC restaurants on the map.")
    if restaurants_map:
        map_data = pd.DataFrame(
            np.array([
                [25.74693, -80.36714],
                [25.76758, -80.36769],
                [25.76110, -80.33434],
                [25.76784, -80.36803],
                [25.75702, -80.36801],
                [25.75684, -80.36749],
                [25.75587, -80.36657],
                [25.76020, -80.38916],
                [25.73047, -80.40331],
                [25.73451, -80.40369],
                [25.73248, -80.35509],
                [25.73562, -80.34661],
                [25.76004, -80.37785],
                [25.76085, -80.36672],
                [25.73400, -80.35492],
                [25.76151, -80.36904],
                [25.76243, -80.36835],
                [25.76164, -80.37187],
                [25.73584, -80.35083]]),
            columns=['lat', 'lon'])
        st.map(map_data)