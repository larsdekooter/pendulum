import pygame
import math
from time import sleep
from math import pi
import numpy as np


class Game:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.centerPos = (1280 / 2, 720 / 3)
        self.stickLength = 200
        self.resetBall()
        self.font = pygame.font.Font("roboto1.ttf", 32)
        pass

    def resetBall(self):
        self.startAngle = 1.5 * pi
        self.angle = self.startAngle
        self.ballPos = (
            math.cos(self.angle) * self.stickLength + 1280 / 2,
            -math.sin(self.angle) * self.stickLength + 720 / 3,
        )  # ???
        self.circleCenters = []

    def step(self, move):
        self.events()
        self.screen.fill("white")
        self.calculateBallPosition(move)
        self.draw()
        self.afterLoad()
        return self.getReward()

    def getReward(self):
        ideal_angle = np.pi / 2  # The desired angle is pi/2
        # Calculate the distance to the ideal angle considering periodicity
        distance = np.abs(np.mod(self.angle - ideal_angle, 2 * np.pi))
        # Ensure reward is higher closer to pi/2 (considering periodicity)
        reward = np.exp(-np.power(distance, 2))
        return reward

    def calculateBallPosition(self, move):
        angler = 0.005
        if move[0] == 1:  # Left
            self.angle -= angler
            if round(self.angle / pi, 2) == 0:
                self.angle = 2 * pi

        elif move[1] == 1:  # Right
            self.angle += angler
            if round(self.angle / pi, 2) == 2:
                self.angle = 0

        else:
            if self.angle < self.startAngle:
                self.angle += 0.99 * angler
            elif self.angle > self.startAngle:
                self.angle -= 0.99 * angler
            else:
                pass

        self.ballPos = (
            math.cos(self.angle) * self.stickLength + 1280 / 2,
            -math.sin(self.angle) * self.stickLength + 720 / 3,
        )

    def draw(self):
        pygame.draw.circle(self.screen, "black", (1280 / 2, 720 / 3), 10)
        pygame.draw.line(
            self.screen,
            "black",
            start_pos=self.centerPos,
            end_pos=self.ballPos,
            width=3,
        )
        pygame.draw.circle(self.screen, "black", self.ballPos, 10)
        text = self.font.render(
            f"{round(self.angle/pi, 2)}pi >>> {int((self.angle)/(pi/180))}°",
            True,
            "black",
            "white",
        )
        self.screen.blit(text, (80, 80))

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def afterLoad(self):

        pygame.display.flip()
