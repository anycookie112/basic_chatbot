from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time


"""

sequential conversation would need memory up to 3 
so if not in database, then search the web

supervisor + agents

supervisor prompt
you are a supervisor for a team of 2 agents.
a customer service agent that answers general questions about the store and a math agent solves arithmetic problems.
answer based on the data give to you only, if you dont have the answer to prices say i dont know
if incomplete information, ask the customer to clarify.

so math agent + cs agent
example prompt would be 
what are the latest products you have in your store? are there any discounts? if so what are the prices? and how much money am i saving.



"""
# Setup
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # Disable for debugging
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
url = "https://shop.zuscoffee.com/collections/drinkware"
driver.get(url)
wait = WebDriverWait(driver, 10)

# Wait for products
wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".product-card__info")))
cards = driver.find_elements(By.CSS_SELECTOR, ".product-card__info")

base_url = "https://shop.zuscoffee.com"


for card in cards:
    try:
        # Product name and URL
        title_tag = card.find_element(By.CSS_SELECTOR, ".product-card__title a")
        name = title_tag.text.strip()
        href = title_tag.get_attribute("href")
        full_url = href if href.startswith("http") else base_url + href
        color_data = []
        swatches = card.find_elements(By.CSS_SELECTOR, "label.thumbnail-swatch")
        labels = card.find_elements(By.CSS_SELECTOR, "label.thumbnail-swatch")



        # Sale price
        try:
            sale_raw = card.find_element(By.CSS_SELECTOR, "sale-price").get_attribute("innerText")
            sale_price = next((line.strip() for line in sale_raw.splitlines() if line.strip().startswith("RM")), "N/A")
        except:
            sale_price = "N/A"

        # Regular price (compare-at)
        try:
            reg_raw = card.find_element(By.CSS_SELECTOR, "compare-at-price").get_attribute("innerText")
            regular_price = next((line.strip() for line in reg_raw.splitlines() if line.strip().startswith("RM")), "N/A")
        except:
            regular_price = "N/A"

        # labels = driver.find_elements(By.CSS_SELECTOR, "label.thumbnail-swatch")

        for label in labels:
            try:
                # Get input ID from label's "for" attribute
                input_id = label.get_attribute("for")

                # Find the corresponding input by ID
                input_elem = card.find_element(By.ID, input_id)


                # Extract color name
                color = input_elem.get_attribute("value").strip()

                # Extract image src from <img> inside label
                img = label.find_element(By.TAG_NAME, "img")
                image_url = img.get_attribute("src")

                color_data.append({"color": color, "image": image_url})
            except Exception as e:
                print(f"⚠️ Error extracting color swatch: {e}")

        # Image URLs
        image_tags = card.find_elements(By.CSS_SELECTOR, ".product-card__variant-list img")
        image_urls = [img.get_attribute("src") for img in image_tags]

        # Print results
        print(f"🧾 {name}")
        print(f"🔗 URL: {full_url}")
        print(f"💵 Sale Price: {sale_price}")
        print(f"💲 Regular Price: {regular_price}")
        for item in color_data:
            print(f"🎨 {item['color']}: {item['image']}")
        print("-" * 50)

    except Exception as e:
        print("⚠️ Failed to extract product:", e)

driver.quit()
