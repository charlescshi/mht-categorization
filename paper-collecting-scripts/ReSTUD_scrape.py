import time
import os
import shutil
import pandas as pd
import regex as re

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup

# CHECK IF IT'S RUNNING AT ALL.

print("hello")

# INSTANTIATE THE SELENIUM DRIVER.

options = webdriver.ChromeOptions()

# ----------------------------------------------
# CHANGE DOWNLOAD FOLDER TO MATCH WHERE YOU WANT THE FILES TO DOWNLOAD TO
# ----------------------------------------------

download_folder = "C:\\Users\\BFI User\\Downloads"

profile = {"plugins.plugins_list": [{"enabled": False,
                                     "name": "Chrome PDF Viewer"}],
           "download.default_directory": download_folder,
           "plugins.always_open_pdf_externally": True,
           "download.extensions_to_open": "applications/pdf"}

options.add_experimental_option("prefs", profile)

# ----------------------------------------------
# DOWNLOAD CHROME DRIVER AND PUT IT IN A FOLDER, THEN CHANGE
# TO WHERE YOU PUT THE CHROMEDRIVER IN.
# LINK: https://chromedriver.chromium.org/downloads
# ----------------------------------------------

driver = webdriver.Chrome("C:\\chromedriver\\chromedriver.exe", options=options)

# READ THE EXCEL FOR THE JOURNAL TO RETRIEVE THE LINKS

# ----------------------------------------------
# CHANGE DIRECTORY TO MATCH LOCATION OF EXCEL FILE.
# ----------------------------------------------

journal_data = pd.read_excel("C:\\Users\\BFI User\\Desktop\\ReSTUD_2000-2022.xlsx")

journal_data.reset_index()

# -------------------
# MAIN LOOP
# -------------------

for index, row in journal_data.iterrows():
    # OPEN THE CORRESPONDING LINK.

    driver.get(row['url'])

    # -------------------
    # QJE-SPECIFIC TASKS
    # -------------------

    # ENSURE WE SEE "QUARTERLY" IN THE TITLE.

    # assert "Quarterly" in driver.title

    # GET THE RAW HTML CODE

    html = driver.page_source
    time.sleep(5)

    # FIND THE SEMI-LINK TO THE PDF ARTICLE AND CONSTRUCT A LINK THAT WORKS.

    soup = BeautifulSoup(html)

    m = soup.find_all("a", {"class": "al-link pdf article-pdfLink"})
    tail = m[0].attrs['href']
    x = 'https://academic.oup.com' + tail

    driver.get(x)

    # -------------------
    # END OF QJE SPECIFIC TASK
    # -------------------

    time.sleep(4)

    # MAKE SURE THAT ARTICLE TITLE CAN BE SAVED (REMOVE WEIRD CHARACTERS AND SUBSTITUTE WITH UNDERSCORE)

    article_title = re.sub('[^A-Za-z0-9]+', '_', row['title'])

    time.sleep(25)

    filename = max([download_folder + "\\" + f for f in os.listdir(download_folder)], key=os.path.getctime)

    # CHANGE "QJE_" TO THE CORRESPONDING JOURNAL

    shutil.move(filename, os.path.join(download_folder, "QJE_" + str(row['id']) + "_" + article_title + ".pdf"))

    time.sleep(10)

driver.close()
