from numpy import random

class Deck:
	"""
	Represents the deck at any given turn.
	"""

	__RANKS = ["A", "10", "K", "Q", "J"]
	__SUITS = ["C", "D", "H", "S"]

	# A list of length 20 representing all cards and their states
	__card_state = None # type: list[str]

	#We use the following index representations for cards:

	# Suit order: CLUBS, DIAMONDS, HEARTS, SPADES

	# 0, 5, 10, 15 - Aces
	# 1, 6, 11, 16 - 10s
	# 2, 7, 12, 17 - Kings
	# 3, 8, 13, 18 - Queens
	# 4, 9, 14, 19 - Jacks

	# List that holds cards which are played at any one time.
	# Can contain two Nones, one None and an int, or two ints.
	# The ints represent the index of the played cards according to the scheme above.
	__trick = [None, None] # type: list[int], list[None]

	# A variable length list of card indexes representing the
	# cards currently in stock, and more importantly, their order.
	# First index in this list is always the trump card, last index
	# is where the cards are taken from the stock.
	__stock = None # type: list[int]

	# The suit of the trump card for this given deck instance.
	__trump = None # type: String

	def __init__(self,
				card_state,	# type: list[str]
				trick, 		# type: list[int]
				stock,		# type: list[int]
				trump 		# type: str
				):
		"""
		:param card_state: list of current card states
		:param stock: list of indexes of cards in stock
		:param trump: {C,D,H,S}
		"""

		self.__card_state = card_state
		self.__trick	 = trick
		self.__stock	 = stock
		self.__trump	 = trump


	# Computes the rank of a given card index, following the ordering given above.
	@staticmethod
	def get_rank(index):
		return Deck.__RANKS[index % 5]
		

	# Computes the suit of a given card index, following the ordering given above.
	@staticmethod
	def get_suit(index):
		return Deck.__SUITS[int(index/5)]

	def get_card_states(self):
		return list(self.__card_state)

	def get_card_state(self, index):
		return self.__card_state[index]

	def get_stock(self):
		return self.__stock

	def set_card(self, index, state):
		self.__card_state[index] = state

	def get_trick(self):
		return list(self.__trick)

	def set_trick(self, player, card):
		self.__trick[player] = card

	def clear_trick(self):
		self.__trick = [None, None]

	def get_player_hand(self, player_id):
		search_term = "P1H" if player_id == 1 else "P2H"
		return [i for i, x in enumerate(self.__card_state) if x == search_term]



	@staticmethod
	def generate():

		shuffled_cards = random.permutation(range(20))

		card_state = [0]*20
		stock = [] # Can be thought of as a stack data structure.

		# Three separate for loops assign a state to the cards in the
		# shuffled deck depending on their position. The indices of the
		# stock cards are pushed onto the stock stack to save their order.
		for i in range(10):
			card_state[shuffled_cards[i]] = "S"
			stock.append(shuffled_cards[i])

		for i in range(10, 15):
			card_state[shuffled_cards[i]] = "P1H"

		for i in range(15, 20):
			card_state[shuffled_cards[i]] = "P2H"

		trump = Deck.get_suit(shuffled_cards[0])

		return Deck(card_state, stock, trump)


	def clone(self):
		return Deck(list(self.__card_state), list(self.__trick), list(self.__stock), self.__trump)
