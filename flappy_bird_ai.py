import pygame
import neat
import os
import random

# 游戏参数
WIN_WIDTH = 500
WIN_HEIGHT = 800
GEN = 0

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 1.5 * self.tick_count**2

        if d >= 16:
            d = 16
        if d < 0:
            d -= 2

        self.y = self.y + d

    def draw(self, win):
        # 绘制鸟的身体 - 黄色圆形
        body_radius = 20
        pygame.draw.circle(win, (255, 200, 0), (int(self.x), int(self.y)), body_radius)
        
        # 鸟的肚子 - 浅色
        pygame.draw.circle(win, (255, 255, 200), (int(self.x + 5), int(self.y + 5)), 12)
        
        # 鸟的眼睛 - 白色
        pygame.draw.circle(win, (255, 255, 255), (int(self.x + 8), int(self.y - 5)), 6)
        # 鸟的眼珠 - 黑色
        pygame.draw.circle(win, (0, 0, 0), (int(self.x + 10), int(self.y - 5)), 3)
        
        # 鸟的嘴巴 - 橙色三角形
        beak_points = [
            (self.x + 25, self.y - 2),
            (self.x + 32, self.y),
            (self.x + 25, self.y + 2)
        ]
        pygame.draw.polygon(win, (255, 140, 0), beak_points)
        
        # 鸟的翅膀 - 黄色小翅膀
        wing_points = [
            (self.x - 5, self.y),
            (self.x - 15, self.y - 5),
            (self.x - 15, self.y + 5)
        ]
        pygame.draw.polygon(win, (255, 220, 0), wing_points)

    def get_mask(self):
        # 简化版碰撞检测
        return pygame.Rect(self.x - 20, self.y - 20, 40, 40)

class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - 600 # 假设管道长度为600
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        # 绘制上管道
        pygame.draw.rect(win, (0, 255, 0), (self.x, self.height - 500, 50, 500))
        # 绘制下管道
        pygame.draw.rect(win, (0, 255, 0), (self.x, self.bottom, 50, 500))

    def collide(self, bird):
        bird_rect = bird.get_mask()
        top_rect = pygame.Rect(self.x, self.height - 500, 50, 500)
        bottom_rect = pygame.Rect(self.x, self.bottom, 50, 500)
        
        if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
            return True
        if bird.y + 20 >= 730 or bird.y - 20 <= 0:
            return True
        return False

class Base:
    VEL = 5
    WIDTH = WIN_WIDTH
    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        pygame.draw.rect(win, (150, 75, 0), (self.x1, self.y, self.WIDTH, 70))
        pygame.draw.rect(win, (150, 75, 0), (self.x2, self.y, self.WIDTH, 70))

def draw_window(win, birds, pipes, base, score, gen):
    win.fill((135, 206, 235)) # 天蓝色背景

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    # 分数显示
    font = pygame.font.SysFont("comicsans", 50)
    score_label = font.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    # 代数显示
    gen_label = font.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(gen_label, (10, 10))

    pygame.display.update()

def eval_genomes(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run and len(birds) > 0:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + 50:
                pipe_ind = 1

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1 # 存活奖励

            # 输入数据给神经网络
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        rem = []
        add_pipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1 # 碰撞惩罚
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            pipe.move()
            if pipe.x + 50 < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5 # 通过管道奖励
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + 20 >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(win, birds, pipes, base, score, GEN)

        if score > 50: # 达到一定分数停止训练
            break

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    pygame.init()
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
