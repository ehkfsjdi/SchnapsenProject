# logistic regression for feature importance
import sys
sys.path.append(r"C:\Users\X\PycharmProjects\pythonProject\venv\Lib\site-packages")
import sklearn
from sklearn.datasets import make_classification
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
from matplotlib import pyplot

# define dataset
with open('dataset.pkl', 'rb') as output:
    data, target = pickle.load(output)

X, y = data, target

# define the model
model = LogisticRegression()
# model = MLPClassifier(hidden_layer_sizes=(64, 32), learning_rate_init=0.0001, alpha=0.0001, verbose=False, early_stopping=True, n_iter_no_change=6)

# fit the model
model.fit(X, y)

# get importance
importance = model.coef_[0]

feature_names = ['possible_clubs_marriage', 'possible_diamonds_marriage', 'possible_hearts_marriage',
                 'possible_spades_marriage', 'no_possible_marriages', 'clubs_trump_exchange',
                 'diamonds_trump_exchange', 'hearts_trump_exchange', 'spades_trump_exchange',
                 'no_trump_exchange', 'no_previous_trick', 'yes_ previous_trick',
                 'perspective_clubs_a_unknown', 'perspective_clubs_a_s', 'perspective_clubs_a_p1h',
                 'perspective_clubs_a_p2h', 'perspective_clubs_a_won1', 'perspective_clubs_a_won2',
                 'perspective_clubs_10_unknown', 'perspective_clubs_10_s', 'perspective_clubs_10_p1h',
                 'perspective_clubs_10_p2h', 'perspective_clubs_10_won1', 'perspective_clubs_10_won2',
                 'perspective_clubs_k_unknown', 'perspective_clubs_k_s', 'perspective_clubs_k_p1h',
                 'perspective_clubs_k_p2h', 'perspective_clubs_k_won1', 'perspective_clubs_k_won2',
                 'perspective_clubs_q_unknown', 'perspective_clubs_q_s', 'perspective_clubs_q_p1h',
                 'perspective_clubs_q_p2h', 'perspective_clubs_q_won1', 'perspective_clubs_q_wom2',
                 'perspective_clubs_j_unknown', 'perspective_clubs_j_s', 'perspective_clubs_j_p1h',
                 'perspective_clubs_j_p2h', 'perspective_clubs_j_won1', 'perspective_clubs_j_won2',
                 'perspective_diamonds_a_unknown', 'perspective_diamonds_a_s', 'perspective_diamonds_a_p1h',
                 'perspective_diamonds_a_p2h', 'perspective_diamonds_a_won1', 'perspective_diamonds_a_won2',
                 'perspective_diamonds_10_unknown', 'perspective_diamonds_10_s', 'perspective_diamonds_10_p1h',
                 'perspective_diamonds_10_p2h', 'perspective_diamonds_10_won1', 'perspective_diamonds_10_won2',
                 'perspective_diamonds_k_unknown', 'perspective_diamonds_k_s', 'perspective_diamonds_k_p1h',
                 'perspective_diamonds_k_p2h', 'perspective_diamonds_k_won1', 'perspective_diamonds_k_won2',
                 'perspective_diamonds_q_unknown', 'perspective_diamonds_q_s', 'perspective_diamonds_q_p1h',
                 'perspective_diamonds_q_p2h', 'perspective_diamonds_q_won1', 'perspective_diamonds_q_wom2',
                 'perspective_diamonds_j_unknown', 'perspective_diamonds_j_s', 'perspective_diamonds_j_p1h',
                 'perspective_diamonds_j_p2h', 'perspective_diamonds_j_won1', 'perspective_diamonds_j_won2',
                 'perspective_hearts_a_unknown', 'perspective_hearts_a_s', 'perspective_hearts_a_p1h',
                 'perspective_hearts_a_p2h', 'perspective_hearts_a_won1', 'perspective_hearts_a_won2',
                 'perspective_hearts_10_unknown', 'perspective_hearts_10_s', 'perspective_hearts_10_p1h',
                 'perspective_hearts_10_p2h', 'perspective_hearts_10_won1', 'perspective_hearts_10_won2',
                 'perspective_hearts_k_unknown', 'perspective_hearts_k_s', 'perspective_hearts_k_p1h',
                 'perspective_hearts_k_p2h', 'perspective_hearts_k_won1', 'perspective_hearts_k_won2',
                 'perspective_hearts_q_unknown', 'perspective_hearts_q_s', 'perspective_hearts_q_p1h',
                 'perspective_hearts_q_p2h', 'perspective_hearts_q_won1', 'perspective_hearts_q_wom2',
                 'perspective_hearts_j_unknown', 'perspective_hearts_j_s', 'perspective_hearts_j_p1h',
                 'perspective_hearts_j_p2h', 'perspective_hearts_j_won1', 'perspective_hearts_j_won2',
                 'perspective_spades_a_unknown', 'perspective_spades_a_s', 'perspective_spades_a_p1h',
                 'perspective_spades_a_p2h', 'perspective_spades_a_won1', 'perspective_spades_a_won2',
                 'perspective_spades_10_unknown', 'perspective_spades_10_s', 'perspective_spades_10_p1h',
                 'perspective_spades_10_p2h', 'perspective_spades_10_won1', 'perspective_spades_10_won2',
                 'perspective_spades_k_unknown', 'perspective_spades_k_s', 'perspective_spades_k_p1h',
                 'perspective_spades_k_p2h', 'perspective_spades_k_won1', 'perspective_spades_k_won2',
                 'perspective_spades_q_unknown', 'perspective_spades_q_s', 'perspective_spades_q_p1h',
                 'perspective_spades_q_p2h', 'perspective_spades_q_won1', 'perspective_spades_q_wom2',
                 'perspective_spades_j_unknown', 'perspective_spades_j_s', 'perspective_spades_j_p1h',
                 'perspective_spades_j_p2h', 'perspective_spades_j_won1', 'perspective_spades_j_won2',
                 'p1_points', 'p2_points', 'p1_pending_points', 'p2_pending_points', 'trump_suit_clubs',
                 'trump_suit_diamonds', 'trump_suit_hearts', 'trump_suit_spades', 'phase_1', 'phase_2',
                 'stock_size', 'leader_1', 'leader_2', 'whose_turn_1', 'whose_turn_2',
                 'opponent_played_card_clubs_a', 'opponent_played_card_clubs_10', 'opponent_played_card_clubs_k',
                 'opponent_played_card_clubs_q', 'opponent_played_card_clubs_j', 'opponent_played_card_diamonds_a',
                 'opponent_played_card_diamonds_10', 'opponent_played_card_diamonds_k',
                 'opponent_played_card_diamonds_q', 'opponent_played_card_diamonds_j', 'opponent_played_card_hearts_a',
                 'opponent_played_card_hearts_10', 'opponent_played_card_hearts_k', 'opponent_played_card_hearts_q',
                 'opponent_played_card_hearts_j', 'opponent_played_card_spades_a', 'opponent_played_card_spades_10',
                 'opponent_played_card_spades_k', 'opponent_played_card_spades_q', 'opponent_played_card_spades_j',
                 'opponent_played_card_none', 'design_matrix_p1_points', 'design_matrix_p2_points',
                 'design_matrix_pe1p_points', 'design_matrix_pe2p_points', 'design_matrix_trump_suit',
                 'design_matrix_phase', 'design_matrix_stock_size', 'design_matrix_leader',
                 'design_matrix_whose_turn', 'design_matrix_opponent_card', 'design_matrix_perspective']

print(len(feature_names))
print(len(importance))
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (feature_names[i],v))

# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

""" import eli5
from eli5.sklearn import PermutationImportance
from eli5 import formatters
from IPython.display import display, HTML

with open('dataset.pkl', 'rb') as output:
    data, target = pickle.load(output)



model = joblib.load('./bots/ml/model.pkl')

permutation = PermutationImportance(model).fit(data, target)
eli5.show_weights(permutation, feature_names=feature_names)

html = eli5.format_as_html(eli5.explain_weights(permutation))
with open('eli5-test.html', 'w') as f:
    f.write(html)"""
