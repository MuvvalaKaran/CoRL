import numpy as np
import matplotlib.pyplot as plt
from .custom_players import *
from .custom_games import *


class Tester:

    def __init__(self, game, playerA=None, playerB=None):
        self.game = game
        self.playerA = playerA
        self.playerB = playerB

    def resultToReward(self, result, actionA=None, actionB=None):
        if result >= 0:
            reward = (result*(-2) + 1)
        else:
            reward = 0
        return reward

    def restrictActions(self):
        return [None, None]

    def plotPolicy(self, player):
        for state in range(player.numStates):
            print("\n=================")
            self.game.draw(*self.stateToBoard(state))
            # print("State value: %s" % player.V[state])
            print(player.Q[state])
            player.policyForState(state)

    def plotResult(self, wins):
        lenWins = len(wins)
        sumWins = (wins == [[0], [1], [-2]]).sum(1)
        print("Wins A : %d (%0.1f%%)" % (sumWins[0], (100. * sumWins[0] / lenWins)))
        print("Wins B : %d (%0.1f%%)" % (sumWins[1], (100. * sumWins[1] / lenWins)))
        print("Draws  : %d (%0.1f%%)" % (sumWins[2], (100. * sumWins[2] / lenWins)))

        plt.plot((wins == 0).cumsum())
        plt.plot((wins == 1).cumsum())
        plt.legend(('WinsA', 'WinsB'), loc=(0.6, 0.8))
        plt.show()

class SoccerTester(Tester):

    def __init__(self, game):
        Tester.__init__(self, game)

    def boardToState(self):
        game = self.game
        xA, yA = game.positions[0]
        xB, yB = game.positions[1]
        sA = yA * game.w + xA
        sB = yB * game.w + xB
        sB -= 1 if sB > sA else 0
        state = (sA * (game.w * game.h - 1) + sB) + (game.w * game.h) * (game.w * game.h - 1) * game.ballOwner
        return state

    def stateToBoard(self, state):
        game = self.game
        ballOwner = state / ((game.w * game.h) * (game.w * game.h - 1))
        state = state % ((game.w * game.h) * (game.w * game.h - 1))

        sA = state / (game.w * game.h - 1)
        sB = state % (game.w * game.h - 1)
        sB += 1 if sB >= sA else 0

        xA = sA % game.w
        yA = sA / game.w
        xB = sB % game.w
        yB = sB / game.w

        return [[[xA, yA], [xB, yB]], ballOwner]

    def resultToReward(self, result, actionA=None, actionB=None):
        factor = 1
        return Tester.resultToReward(self, result) * factor

def testGame(playerA, playerB, gameTester, iterations):
    wins = np.zeros(iterations)

    for i in np.arange(iterations):
        if (i % (iterations / 10) == 0):
            print("%d%%" % (i * 100 / iterations))
        gameTester.game.restart()
        result = -1
        while result == -1:
            state = gameTester.boardToState()
            restrictA, restrictB = gameTester.restrictActions()
            actionA = playerA.chooseAction(state, restrictA)
            actionB = playerB.chooseAction(state, restrictB)
            result = gameTester.game.play(actionA, actionB)
            reward = gameTester.resultToReward(result, actionA, actionB)
            newState = gameTester.boardToState()

            playerA.getReward(state, newState, [actionA, actionB], reward, [restrictA, restrictB])
            playerB.getReward(state, newState, [actionB, actionA], -reward, [restrictB, restrictA])

        wins[i] = result
    return wins

def testSoccer(iterations):
    boardH = 4
    boardW = 5
    numStates = (boardW * boardH) * (boardW * boardH - 1) * 2
    numActions = 5
    drawProbability = 0.01
    decay = 10**(-2. / iterations * 0.05)

    ### CHOOSE PLAYER_A TYPE
    # playerA = RandomPlayer(numActions)
    playerA = MinimaxQPlayer(numStates, numActions, numActions, decay=decay, expl=0.3, gamma=1-drawProbability)
    # playerA = QPlayer(numStates, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    # playerA = np.load('SavedPlayersminimax/Q_SoccerA_100000.npy').item()

    ### CHOOSE PLAYER_B TYPE
    playerB = RandomPlayer(numActions)
    # playerB = MinimaxQPlayer(numStates, numActions, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    # playerB = QPlayer(numStates, numActions, decay=decay, expl=0.2, gamma=1-drawProbability)
    # playerB = np.load('SavedPlayers/Q_SoccerB_100000.npy').item()

    ### INSTANTIATE GAME AND TESTER
    game = Soccer(boardH, boardW, drawProbability=drawProbability)
    tester = SoccerTester(game)

    ### RUN TEST
    wins = testGame(playerA, playerB, tester, iterations)

    ### DISPLAY RESULTS
    tester.plotPolicy(playerA)
    # tester.plotPolicy(playerB)
    tester.plotResult(wins)

    # np.save("SoccerA_10000", playerA)
    # np.save("SoccerB_10000", playerB)

def testSoccerPerformance():
    boardH, boardW = 4, 5
    numStates = (boardW * boardH) * (boardW * boardH - 1) * 2
    numActions = 5
    drawProbability = 0.01

    ### INSTANTIATE GAME
    game = Soccer(boardH, boardW, drawProbability=drawProbability)

    print("AIM : EVALUATE OUR MINIMAX Q 'PLAYER A' TRAINED OVER 100.000 ITERATIONS")
    print("METHOD : MAKE IT FIGHT AGAINST A DETERMINISTIC PLAYER\n \
        AGAINST THERE EXIST A DETERMINISTIC POLICY THAT ALWAYS WINS")

    print("\n=======================================================")
    print("STEP 1: CREATE A DETERMINISTIC 'PLAYER B' TO FIGHT WITH")

    ### CHOOSE PLAYER_B AS Q LEARNER
    playerB = QPlayer(numStates, numActions, decay=1-1e-4, expl=0.3, gamma=1-drawProbability)

    ### TRAIN IT AGAINST ANOTHER Q LEARNER
    print("\n1.1 - TRAIN OUR 'PLAYER B' (Q LEARNER) AGAINST ANOTHER Q LEARNER - 5000 games")
    playerA1 = QPlayer(numStates, numActions, decay=1-1e-4, expl=0.5, gamma=1-drawProbability)
    tester = SoccerTester(game)
    wins = testGame(playerA1, playerB, tester, 5000)

    ### TRAIN A Q LEARNER TO BEAT IT
    print("\n1.2 - TRAIN ANOTHER Q LEARNER TO BEAT 'PLAYER B' - 10000 games")
    print("('PLAYER B' is not learning anymore)")
    playerB.learning = False
    playerA2 = QPlayer(numStates, numActions, decay=1-1e-4, expl=0.3, gamma=1-drawProbability)
    wins = testGame(playerA2, playerB, tester, 10000)
    tester.plotResult(wins)

    ### CHECK THIS Q LEARNER
    print("\n1.3 - CHECK THIS Q LEARNER ALWAYS BEAT 'PLAYER B' - 1000 games")
    print("(This step is facultative -- 'PLAYER B' should be always losing)")
    print("(Note: If it is not the case, relaunch program)")
    playerA2.learning = False
    wins = testGame(playerA2, playerB, tester, 1000)
    tester.plotResult(wins)

    ### MAKE FIGHT ! PLAYER A vs PLAYER B
    print("\n\n======================================================")
    print("STEP 2: MAKE PLAYER 'A' FIGHT 'PLAYER B' - 10000 games")
    playerA3 = np.load('SavedPlayers/minimaxQ_SoccerA_100000.npy').item()
    playerA3.learning = False
    wins = testGame(playerA3, playerB, tester, 10000)
    tester.plotResult(wins)

    v = playerA2.pi == 1
    prod = sum(playerA2.pi[v] * playerA3.pi[v])
    print('\nApproximate percentage of correct actions : %0.1f%%' % (100 * prod / np.sum(v)))
