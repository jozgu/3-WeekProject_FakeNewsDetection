#WebScraper
#Credit Emil Gungaard
#Inspiration from Python&stuff https://www.youtube.com/watch?v=RvCBzhhydNk&t=660s

from bs4 import BeautifulSoup
import requests
from csv import writer

#The website you want to scrape
url= "https://www.washingtonpost.com/local/obituaries/sidney-poitier-dies/2022/01/07/01ba4ea0-c189-11ea-9fdd-b7ac6b051dc8_story.html"
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')
# lists = soup.find_all('section', class_="listing-search-item")

#HTML class you want the scraper to search in
lists = soup.find_all('article')

#Name the CSV file you want to be created
with open('my_new_csv_file.csv' , 'w', encoding='utf8', newline='') as f:
    thewriter = writer(f)

    #Give column titles
    header = ['title', 'text', 'label', 'source']
    thewriter.writerow(header)

    for list in lists:

        #Depending on website, use list.find or list.find_all
        title = 'Could not be scraped'
        # title = list.find('span', dataqa_="headline-text").text.replace('\n', '')
        
        #Depending on website, use list.find or list.find_all
        text = list.find('div', class_="article-body",).text.replace('\n', '')
        # text = list.find_all('p').text.replace('\n', '')

        #Label: 0 = Non fake news, 1 = fake news
        label = '0'

        #Provide the source
        source = 'WashingtonPost'
        
        #Convert to list for thewriter
        info = [title, text, label, source]
        thewriter.writerow(info)