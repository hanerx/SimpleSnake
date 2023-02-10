import math
import random

import numpy as np
import pygame


class RenderTarget:
    def __init__(self, screen):
        self.screen = screen

    def draw(self):
        pass


class GameState(RenderTarget):
    def __init__(self, screen):
        super().__init__(screen)
        self.score = 0
        self.step = 0
        self.best = 0
        self.gen = 0
        self.is_dead = False

    def reset(self):
        self.score = 0
        self.step = 0
        self.is_dead = False

    def game_over(self):
        self.best = max(self.best, self.score)
        self.gen += 1
        self.is_dead = True

    def draw(self):
        game_state_font = pygame.font.Font(None, 25)
        if self.is_dead:
            game_over_text = game_state_font.render("Game Over!", True, pygame.Color(255, 0, 0))
            self.screen.blit(game_over_text, [320, 240])
        score_text = game_state_font.render("Score: " + str(self.score), True, pygame.Color(255, 255, 255))
        best_text = game_state_font.render('Best: {} Gen: {}'.format(self.best, self.gen), True,
                                           pygame.Color(0, 0, 255))
        self.screen.blit(score_text, [650, 10])
        self.screen.blit(best_text, [10, 10])


class Map(RenderTarget):
    def __init__(self, map_width, map_height, screen):
        super().__init__(screen)
        self.map = []
        self.map_width = map_width
        self.map_height = map_height
        self.game_state = GameState(screen)

    def add_item(self, entity):
        self.map.append(entity)
        entity.map = self

    def tick(self):
        if self.game_state.is_dead:
            self.reset()
            pygame.time.wait(300)
        else:
            for i in self.map:
                i.tick()
            self.draw()

    def reset(self):
        self.game_state.reset()
        for i in self.map:
            i.reset()

    def can_use(self, block):
        if block[0] < 0 or block[0] >= self.map_width or block[1] < 0 or block[1] >= self.map_height:
            return False
        for i in self.map:
            if block in i.blocks:
                return False
        return True

    def get_current_block_entity(self, block):
        for i in self.map:
            if block in i.blocks:
                return i
        return None

    def game_over(self):
        self.game_state.game_over()
        for i in self.map:
            i.game_over()

    def draw(self):
        screen.fill(pygame.Color(0, 0, 0))
        for i in self.map:
            i.draw()
        self.game_state.draw()
        pygame.display.update()

    def find_entity(self, target_class):
        for i in self.map:
            if isinstance(i, target_class):
                return i
        return None


class MapEntity(RenderTarget):
    def __init__(self, screen, color):
        super().__init__(screen)
        self.blocks = []
        self.map = None
        self.block_size = 20
        self.color = color

    def tick(self):
        pass

    def reset(self):
        pass

    def game_over(self):
        pass

    def draw(self):
        for i in self.blocks:
            pygame.draw.rect(self.screen, self.color,
                             [i[0] * self.block_size, i[1] * self.block_size, self.block_size, self.block_size])


class Food(MapEntity):
    def __init__(self, screen):
        super().__init__(screen, pygame.Color(255, 0, 0))

    def reset(self):
        self.regenFood()

    def regenFood(self):
        if self.map is not None:
            while True:
                new_pos = [random.randint(0, self.map.map_width - 1), random.randint(0, self.map.map_height - 1)]
                if self.map.can_use(new_pos):
                    break
            self.blocks = [new_pos]


class Pilot:
    def get_action(self, target) -> int:
        pass


class Snake(MapEntity):
    def __init__(self, screen, pilot):
        super().__init__(screen, pygame.Color(0, 255, 0))
        self.blocks = [[5, 5]]
        self.direction = 0
        self.pilot = pilot

    def reset(self):
        self.blocks = [[5, 5]]
        self.direction = 0

    def get_head(self):
        return self.blocks[-1]

    def get_next(self):
        if self.direction == 0:
            return [self.get_head()[0], self.get_head()[1] - 1]
        if self.direction == 1:
            return [self.get_head()[0], self.get_head()[1] + 1]
        if self.direction == 2:
            return [self.get_head()[0] - 1, self.get_head()[1]]
        if self.direction == 3:
            return [self.get_head()[0] + 1, self.get_head()[1]]

    def tick(self):
        self.direction = self.pilot.get_action(self)
        next_head = self.get_next()
        if self.map.can_use(next_head):
            self.blocks.append(next_head)
            del self.blocks[0]
            self.map.game_state.step += 1
            return
        target_food = self.map.get_current_block_entity(next_head)
        if isinstance(target_food, Food):
            self.blocks.append(next_head)
            target_food.reset()
            self.map.game_state.score += 1
        else:
            self.map.game_over()


class RandomPilot(Pilot):
    def get_action(self, target) -> int:
        return random.randint(0, 3)


class ManualPilot(Pilot):
    def get_action(self, target) -> int:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and target.direction != 1:
                    return 0
                if event.key == pygame.K_DOWN and target.direction != 0:
                    return 1
                if event.key == pygame.K_LEFT and target.direction != 3:
                    return 2
                if event.key == pygame.K_RIGHT and target.direction != 2:
                    return 3
        return target.direction


class CrossSelectPilot(Pilot):

    def __init__(self, game_map):
        self.snake = None
        self.map = game_map
        self.max_distance = math.sqrt(game_map.map_width ** 2 + game_map.map_height ** 2)

    def get_action(self, target) -> int:
        target_food = self.map.find_entity(Food)
        self.snake = target
        if target_food is not None:
            state = self.generate_state(target, target_food)
            return int(np.argmax(state))
        return 0

    def generate_state(self, target_snake, target_food):
        snake_head = target_snake.get_head()
        return np.array([self.get_distance([snake_head[0], snake_head[1] - 1], target_food.blocks[0]),
                         self.get_distance([snake_head[0], snake_head[1] + 1], target_food.blocks[0]),
                         self.get_distance([snake_head[0] - 1, snake_head[1]], target_food.blocks[0]),
                         self.get_distance([snake_head[0] + 1, snake_head[1]], target_food.blocks[0])])

    def get_distance(self, p1, p2):
        if p1 in self.snake.blocks or p1[0] < 0 or p1[0] >= self.map.map_width or p1[1] < 0 or p1[1] >= self.map.map_height:
            return -1
        return 1 - math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) / self.max_distance


if __name__ == '__main__':
    # Initialize the game
    pygame.init()
    clock = pygame.time.Clock()

    # Set the screen size
    screen_width = 720
    screen_height = 480
    screen = pygame.display.set_mode((screen_width, screen_height))

    game_map = Map(screen_width // 20, screen_height // 20, screen)
    pilot = CrossSelectPilot(game_map)
    snake = Snake(screen, pilot)
    food = Food(screen)
    game_map.add_item(snake)
    game_map.add_item(food)
    food.reset()
    while True:
        game_map.tick()
        clock.tick(60)
