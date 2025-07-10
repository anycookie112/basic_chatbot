from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

# Setup browser
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Comment this if you want to see the browser
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Go to the URL
URL = "https://www.google.com/maps/place/ZUS+Coffee+-+Temu+Business+Centre+City+Of+Elmina/@3.1858012,101.5265532,1035m/data=!3m2!1e3!4b1!4m6!3m5!1s0x31cc4fef33ad5851:0x4e713b1b2136bea3!8m2!3d3.1858012!4d101.5265532!16s%2Fg%2F11kbljysxb?entry=ttu"
driver.get(URL)

# Wait for the hours table to load
time.sleep(5)  # Adjust based on internet speed or use WebDriverWait

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

# Print results
for day, time in hours.items():
    print(f"{day}: {time}")

driver.quit()

"""

so i need to loop
get name of outlet
get address of outlet
get url of the google maps 
go into the url and scrape the opening hours
close the driver, repeat for all cards in the page



"""