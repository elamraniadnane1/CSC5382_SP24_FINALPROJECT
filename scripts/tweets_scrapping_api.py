from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
# Import Chrome options
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService

def twitter_login(driver, username, password):
    login_url = 'https://twitter.com/i/flow/login'
    driver.get(login_url)

    try:
        # Wait for the username field
        WebDriverWait(driver, 80).until(
            EC.presence_of_element_located((By.NAME, 'session[username_or_email]'))
        )
        username_field = driver.find_element(By.NAME, 'session[username_or_email]')
        username_field.send_keys(username)

        password_field = driver.find_element(By.NAME, 'session[password]')
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)

        print("Login successful")
    except TimeoutException:
        print("Timeout waiting for login page elements or login did not succeed")
    except NoSuchElementException:
        print("Login page elements not found")
    except Exception as e:
        print(f"An error occurred: {e}")



def scrape_twitter(hashtags, start_date, num_tweets, username, password):
    # Initialize the WebDriver for Chrome
    service = ChromeService(executable_path='/usr/bin/chromedriver')  # Update the path to chromedriver
    options = ChromeOptions()
    # For headless browsing (optional)

    driver = webdriver.Chrome(service=service, options=options)

    # Login to Twitter
    twitter_login(driver, username, password)

    data = []

    for hashtag in hashtags:
        # Navigate to Twitter
        url = f"https://twitter.com/search?q={hashtag}%20since%3A{start_date}&src=typed_query"
        driver.get(url)

        while len(data) < num_tweets:
            # Scroll down
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(1)

            # Find tweets
            tweets = driver.find_elements(By.XPATH, '//div[@data-testid="tweet"]')
            for tweet in tweets:
                if len(data) >= num_tweets:
                    break
                try:
                    tweet_data = {}
                    tweet_data['tweet_id'] = tweet.get_attribute('data-item-id')
                    tweet_data['text'] = tweet.find_element(By.XPATH, './/div[2]/div[2]/div[1]').text
                    tweet_data['label'] = 'NONE'
                    data.append(tweet_data)
                except Exception as e:
                    print(f"Error: {e}")
                    pass

    driver.close()

    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Save to CSV
    df.to_csv('tweets.csv', index=False)

# Credentials should be stored securely
username = os.environ.get('TWITTER_USERNAME')
password = os.environ.get('TWITTER_PASSWORD')

# Example usage
scrape_twitter(['#Biden'], '2024-03-01', 100, username, password)
