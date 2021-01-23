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
        spaces = " "
        bot_name = spaces.join(results[:-1:])

        binomial_test = stats.binom_test(won_games, n=TOTAL_GAMES, p=0.5, alternative='greater')
        print("The binomial test result of bot %s is " % bot_name)
        print(binomial_test)
        new_file.write("The binomial test result of bot %s is " % bot_name)
        new_file.write(str(binomial_test))
        new_file.write("\n")

new_file.close()
