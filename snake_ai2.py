import pygame
import numpy as np
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# ================= Snake Environment =================
class SnakeEnv:
    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [[5, 5], [5, 6], [5, 7]]
        self.direction = "up"
        self.apple = [randint(0, self.grid_size - 1), randint(0, self.grid_size - 1)]
        self.done = False
        self.score = 0
        return self.get_state()

    def get_state(self):
        head = self.snake[-1]
        apple = self.apple
        return np.array([
            head[0], head[1],
            apple[0], apple[1],
            self.direction == "up",
            self.direction == "down",
            self.direction == "left",
            self.direction == "right"
        ], dtype=np.float32)

    def step(self, action):
        if action == 0 and self.direction != "down":
            self.direction = "up"
        elif action == 1 and self.direction != "up":
            self.direction = "down"
        elif action == 2 and self.direction != "right":
            self.direction = "left"
        elif action == 3 and self.direction != "left":
            self.direction = "right"

        head_x, head_y = self.snake[-1]
        if self.direction == "up": head_y -= 1
        if self.direction == "down": head_y += 1
        if self.direction == "left": head_x -= 1
        if self.direction == "right": head_x += 1

        new_head = [head_x, head_y]

        if (head_x < 0 or head_x >= self.grid_size or
                head_y < 0 or head_y >= self.grid_size or
                new_head in self.snake):
            self.done = True
            reward = -10
        else:
            self.snake.append(new_head)
            if new_head == self.apple:
                reward = 10
                self.score += 1
                print(f"Score: {self.score}")
                self.apple = [randint(0, self.grid_size - 1), randint(0, self.grid_size - 1)]
            else:
                reward = -0.1
                self.snake.pop(0)
        return self.get_state(), reward, self.done

    # def step(self, action):
    #     if action == 0 and self.direction != "down": self.direction = "up"
    #     elif action == 1 and self.direction != "up": self.direction = "down"
    #     elif action == 2 and self.direction != "right": self.direction = "left"
    #     elif action == 3 and self.direction != "left": self.direction = "right"

    #     head_x, head_y = self.snake[-1]
    #     old_dist = abs(head_x - self.apple[0]) + abs(head_y - self.apple[1])

    #     if self.direction == "up": head_y -= 1
    #     if self.direction == "down": head_y += 1
    #     if self.direction == "left": head_x -= 1
    #     if self.direction == "right": head_x += 1

    #     # Wrap-around
    #     head_x %= self.grid_size
    #     head_y %= self.grid_size

    #     new_head = [head_x, head_y]
    #     new_dist = abs(head_x - self.apple[0]) + abs(head_y - self.apple[1])

    #     self.done = False
    #     self.snake.append(new_head)

    #     if new_head == self.apple:
    #         reward = 10
    #         self.score += 1
    #         print(f"Score: {self.score}")
    #         self.apple = [randint(0, self.grid_size-1), randint(0, self.grid_size-1)]
    #     else:
    #         reward = -0.1
    #         self.snake.pop(0)

    #     # 👇 thêm phần thưởng định hướng
    #     if new_dist < old_dist:
    #         reward += 10
    #     else:
    #         reward -= 10

    #     return self.get_state(), reward, self.done

    def control(self, key):
        if key == pygame.K_UP: self.direction = "up"
        if key == pygame.K_DOWN: self.direction = "down"
        if key == pygame.K_LEFT: self.direction = "left"
        if key == pygame.K_RIGHT: self.direction = "left"

# ================= DQN Model =================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# ================= Training Function =================
def train_snake(episodes=500):
    env = SnakeEnv()
    input_dim = len(env.get_state())
    output_dim = 4
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=5000)

    gamma = 0.9
    batch_size = 64
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state))
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode}: score={env.score}, reward={total_reward:.2f}")
                break

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states)
                actions = torch.tensor(actions).unsqueeze(1)
                rewards = torch.tensor(rewards)
                next_states = torch.tensor(next_states)
                dones = torch.tensor(dones)

                q_values = policy_net(states).gather(1, actions).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                target = rewards + gamma * next_q_values * (1 - dones.float())

                loss = loss_fn(q_values, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), "snake_dqn.pth")
    print("Training finished, model saved as snake_dqn.pth")

# ================= AI Play with Render =================
def play_ai_render():
    env = SnakeEnv()
    model = DQN(len(env.get_state()), 4)
    model.load_state_dict(torch.load("snake_dqn30k.pth"))
    model.eval()

    pygame.init()
    cell_size = 30
    x, y = 0, 0
    screen_size = env.grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    clock = pygame.time.Clock()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    def font(size):
        return pygame.font.Font("Fonts/Pixel Game.otf", size)

    score_font = font(80)
    text_font = font(60)

    drawing = False
    state = "menu"
    running = True
    while running and not env.done:
        if state == "menu":
            pos_x, pos_y = pygame.mouse.get_pos()
            screen.fill(BLACK)

            # for i in range(screen_size):
            #     pygame.draw.line(screen, WHITE, (0, cell_size*i), (screen_size, cell_size*i))
            #     pygame.draw.line(screen, WHITE, (cell_size*i, 0), (cell_size*i, screen_size))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.done = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pos_x, pos_y
                    # print(pos_x, pos_y)
                    drawing = True

                    if pos_x >= 240 and pos_x <= 420 and pos_y >= 420 and pos_y <= 520:
                        state = "ai"
                    if pos_x >= 450 and pos_x <= 520 and pos_y >= 420 and pos_y <= 520:
                        state = "player"

            pygame.draw.rect(screen, WHITE, (240, 420, 180, 100))
            pygame.draw.rect(screen, WHITE, (450, 420, 180, 100))

            text = text_font.render("Who will play first?", True, WHITE)
            coordinates = text_font.render(f"{pos_x} {pos_y}", True, WHITE)
            ai = text_font.render(f"Not me!", True, BLACK)
            player = text_font.render(f"Me!", True, BLACK)

            blit_sequence = [
                (ai, (250, 440)),
                (player, (510, 440)),
                (text, (200, 300)),
                (coordinates, (50, 50)),
            ]
            screen.blits(blit_sequence)

            pygame.display.flip()

        elif state == "ai":
            score = score_font.render(str(env.score), True, WHITE)
            screen.fill(BLACK)

            # Draw apple
            pygame.draw.rect(screen, RED,
                             (env.apple[0] * cell_size, env.apple[1] * cell_size, cell_size, cell_size))
            # Draw snake
            for seg in env.snake:
                pygame.draw.rect(screen, GREEN,
                                 (seg[0] * cell_size, seg[1] * cell_size, cell_size, cell_size))

            # AI move
            action = torch.argmax(model(torch.tensor(env.get_state()))).item()
            env.step(action)

            # Handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.done = True

            screen.blit(score, (screen_size/2 - 10, 100))
            pygame.display.flip()
            clock.tick(10)

        elif state == "player":
            score = score_font.render(str(env.score), True, WHITE)
            screen.fill(BLACK)

            # Draw apple
            pygame.draw.rect(screen, RED,
                             (env.apple[0] * cell_size, env.apple[1] * cell_size, cell_size, cell_size))
            # Draw snake
            for seg in env.snake:
                pygame.draw.rect(screen, GREEN,
                                 (seg[0] * cell_size, seg[1] * cell_size, cell_size, cell_size))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.done = True
                if event.type == pygame.KEYDOWN:
                    env.control(event.key)

            screen.blit(score, (screen_size/2 - 10, 100))
            pygame.display.flip()
            clock.tick(10)

    pygame.quit()
    print("Game Over! Score:", env.score)


# ================= Main =================
if __name__ == "__main__":
    choice = input("Train AI (t) or Play AI (p)? ")
    if choice.lower() == "t":
        train_snake(episodes=2000)  # bạn chỉnh số episodes nếu muốn
    elif choice.lower() == "p":
        play_ai_render()
    else:
        print("Invalid choice")
    # while True:
    #     play_ai_render()
