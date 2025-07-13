from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
from datetime import datetime
import requests
import sqlite3

conn = sqlite3.connect("zus_outlets.db")  # This creates a file-based DB
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

    # Setup browser
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Comment this if you want to see the browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Go to the URL

    driver.get(URL)

    # Wait for the hours table to load
    time.sleep(8)  # Adjust based on internet speed or use WebDriverWait

    # Get the rendered HTML
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Parse the hours
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
        # Get all <section> blocks
        sections = soup.select("div.elementor > section.elementor-section")

        # Process 3 sections at a time
        for i in range(0, len(sections), 3):
            try:
                name_section = sections[i]
                address_section = sections[i + 1]
                map_section = sections[i + 2]

                # Extract store name
                title_tag = name_section.select_one("p.elementor-heading-title")
                store_name = title_tag.get_text(strip=True) if title_tag else "N/A"

                # Skip invalid entries
                if not store_name.startswith("ZUS Coffee"):
                    continue

                # Extract address
                address_tag = address_section.select_one("p")
                address = address_tag.get_text(strip=True) if address_tag else "N/A"

                # Extract Google Maps link
                link_tag = map_section.select_one("a.premium-button[href]")
                direction_url = link_tag["href"] if link_tag and "maps.app.goo.gl" in link_tag["href"] else "N/A"
                
                operation = google_scrape(direction_url)
                print(operation)
                outlets.append({
                    "store_name": store_name,
                    "address": address,
                    "direction_url": direction_url,
                })
                    # Convert to 24-hour format and print
                for day, time_range in operation.items():
                    # Fix non-breaking spaces or unicode nbsps
                    time_range = time_range.replace('\u202f', '').replace(' ', '')  

                    try:
                        opening_str, closing_str = time_range.split("–")
                        # Handle AM/PM time formats
                        opening_24 = datetime.strptime(opening_str.strip(), "%I%p").strftime("%H:%M")
                        closing_24 = datetime.strptime(closing_str.strip(), "%I:%M%p").strftime("%H:%M")
                        # print(f"{day} | {opening_24} | {closing_24}")

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
                # Not enough sections left for a full outlet block
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

        # for outlet in outlets:
        #     print(outlet)

scrape_zus(URL_main)


"""

so i need to loop
get name of outlet
get address of outlet
get url of the google maps 
go into the url and scrape the opening hours
close the driver, repeat for all cards in the page



"""