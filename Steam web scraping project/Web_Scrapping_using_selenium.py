from selenium import webdriver
import time
import csv


class SteamScraper():
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.get(
            'https://store.steampowered.com/games/#p=0&tab=TopSellers')
        # e1.get_attribute('data-ds-appid')


scraper = SteamScraper()
games_ids = []
games_names = []
counter = 0
time.sleep(10)
with open('steam1.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile, dialect='excel')
    wr.writerow(['Game name', 'Game_id'])
    while True:
        games = scraper.driver.find_element_by_css_selector('#TopSellersRows')
        number_of_games_per_page = len(
            games.find_elements_by_css_selector('a'))
        for i in range(number_of_games_per_page):
            game_id = scraper.driver.find_element_by_css_selector(
                '#TopSellersRows > a:nth-child({index})'.format(index=i + 1)).get_attribute('data-ds-appid')
            game_name = scraper.driver.find_element_by_css_selector(
                '#TopSellersRows > a:nth-child({index}) > div.tab_item_content > div.tab_item_name'.format(index=i + 1)).text
            game_id = str(game_id).split(',')[0]
            games_ids.append(str(game_id))
            games_names.append(game_name)
            wr.writerow([game_id, game_name.encode("utf-8")])
        button = scraper.driver.find_element_by_css_selector(
            '#TopSellers_btn_next')
        button.click()
        # print(len(set(games_ids)))
        counter = counter + 1
        # Here we used a counter because we coouldn't find a way to stop at the last page without using one (maybe in the future)
        # But we noticed weird behavior  after 100 pages, for example, the button to move to the next page somehow becomes unclickable with
        # selenium but works perfectly fine with a human click
        if counter == 100:
            break
        time.sleep(10)

# with open('steam1.csv', 'w', newline='') as csvfile:
#     wr = csv.writer(csvfile, dialect='excel')
#     wr.writerow(['Game name', 'Game_id','Num of reviews'])
#     for app_id,app_name in list(zip(games_ids,games_names)):
#         wr.writerow([app_name,app_id])

# TopSellersRows > a:nth-child(5)
