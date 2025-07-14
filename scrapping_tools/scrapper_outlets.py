from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from datetime import datetime
import requests
import sqlite3

conn = sqlite3.connect("zus_outlets.db")  
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS outlets (
        id_outlet INTEGER PRIMARY KEY AUTOINCREMENT,
        store_name TEXT,
        address TEXT,
        direction_url TEXT
    )
""")
conn.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS operation (
        id_operation INTEGER PRIMARY KEY AUTOINCREMENT,
        store_name TEXT,
        day TEXT,
        opening_time TEXT,
        closing_time TEXT
    )
""")
conn.commit()


def google_scrape(URL):
    import time

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(URL)

    time.sleep(8)  

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    hours = {}
    for row in soup.find_all("tr", class_="y0skZc"):
        day_tag = row.find("td", class_="ylH6lf")
        time_tag = row.find("td", class_="mxowUb")
        if day_tag and time_tag:
            day = day_tag.get_text(strip=True)
            time_range = time_tag.get_text(strip=True)
            hours[day] = time_range

    driver.quit()

    return hours



headers = {
    "User-Agent": "Mozilla/5.0"
}



URL_main = "https://zuscoffee.com/category/store/kuala-lumpur-selangor/"


outlets = []
operations = []
def scrape_zus(URL_main):
    page = 1
    while page <= 2:
        if page == 1:
            url = URL_main
        else:
            url = f"{URL_main}/page/{page}/"
            
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print("No more pages or request failed.")
            break

        
        soup = BeautifulSoup(response.content, "html.parser")
        sections = soup.select("div.elementor > section.elementor-section")

        for i in range(0, len(sections), 3):
            try:
                name_section = sections[i]
                address_section = sections[i + 1]
                map_section = sections[i + 2]

                title_tag = name_section.select_one("p.elementor-heading-title")
                store_name = title_tag.get_text(strip=True) if title_tag else "N/A"

                if not store_name.startswith("ZUS Coffee"):
                    continue

                address_tag = address_section.select_one("p")
                address = address_tag.get_text(strip=True) if address_tag else "N/A"

                link_tag = map_section.select_one("a.premium-button[href]")
                direction_url = link_tag["href"] if link_tag and "maps.app.goo.gl" in link_tag["href"] else "N/A"
                
                operation = google_scrape(direction_url)
                print(operation)
                outlets.append({
                    "store_name": store_name,
                    "address": address,
                    "direction_url": direction_url,
                })
                for day, time_range in operation.items():
                    time_range = time_range.replace('\u202f', '').replace(' ', '')  

                    try:
                        opening_str, closing_str = time_range.split("–")
                        opening_24 = datetime.strptime(opening_str.strip(), "%I%p").strftime("%H:%M")
                        closing_24 = datetime.strptime(closing_str.strip(), "%I:%M%p").strftime("%H:%M")

                        operations.append({
                        "store_name": store_name,
                        "opening_time": opening_24,
                        "closing_time": closing_24,
                        "days": day
                    })
                    except ValueError:
                        print(f"{day} | Invalid time format: {time_range}")
                        operations.append({
                            "store_name": store_name,
                            "opening_time": "closed",
                            "closing_time": "closed",
                            "days": day
                        })
                
                

            except IndexError:
                break
        
        page += 1

        # Print results
        for outlet in outlets:
            cursor.execute("""
                INSERT INTO outlets (store_name, address, direction_url)
                VALUES (?, ?, ?)
            """, (
                outlet["store_name"],
                outlet["address"],
                outlet["direction_url"],
            ))
        for operation in operations:
            cursor.execute("""
                INSERT INTO operation (store_name, day, opening_time, closing_time)
                VALUES (?, ?, ?, ?)
            """, (
                operation["store_name"],
                operation["days"],
                operation["opening_time"],
                operation["closing_time"]
            ))

        conn.commit()
    conn.close()


scrape_zus(URL_main)


"""

so i need to loop
get name of outlet
get address of outlet
get url of the google maps 
go into the url and scrape the opening hours
close the driver, repeat for all cards in the page



"""