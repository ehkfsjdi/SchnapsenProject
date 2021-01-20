import sys
sys.path.append(r"C:\Users\X\PycharmProjects\pythonProject\venv\Lib\site-packages")
from scipy import stats
TOTAL_GAMES = 1000

file = open("results").readlines()
new_file = open('binomial_test_results.txt', 'w')

for line in file:
    bots = line.split(",")

    for bot in bots:
        results = bot.split()
        won_games = results[-1]
        bot_name = results[::-1]

        binomial_test = stats.binom_test(won_games, n=TOTAL_GAMES, p=0.5, alternative='greater')
        print(binomial_test)
        new_file.write(binomial_test)

new_file.close()
