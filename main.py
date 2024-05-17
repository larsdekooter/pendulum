from game import Game
import pygame
from network import Network

network = Network()
game = Game(network)


def getMove():
    keys = pygame.key.get_pressed()
    move = [0, 0, 0]
    if keys[pygame.K_LEFT]:
        move[0] = 1
    if keys[pygame.K_RIGHT]:
        move[1] = 1
    if keys[pygame.K_r]:
        move[2] = 1
    return move


while True:
    state = network.getState(game)
    move = network.getMove(state)
    # move = getMove()
    reward, ended = game.step(move)
    nextState = network.getState(game)
    network.trainShort(state, move, reward, nextState)
    if getMove()[2] == 1:
        network.model.save()
    if ended:
        network.trainLong()
    # if move[2] == 1:
    #     pygame.quit()
    #     exit()
