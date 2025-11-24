import pygame
import numpy as np
from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
import random
from random import choice
from collections import deque
import textwrap


# ================= Snake Environment =================
class SnakeEnv:
    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [[5, 5], [5, 6], [5, 7]]
        self.direction = "right"
        self.apple = [randint(0, self.grid_size - 1), randint(0, self.grid_size - 1)]
        self.done = False
        self.score = 0
        self.speed = 10
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
                self.speed += 0.3
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
        if (key == pygame.K_UP or key == pygame.K_w) and self.direction != "down": 
            self.direction = "up"
        elif (key == pygame.K_DOWN or key == pygame.K_s) and self.direction != "up": 
            self.direction = "down"
        elif (key == pygame.K_LEFT or key == pygame.K_a) and self.direction != "right": 
            self.direction = "left"
        elif (key == pygame.K_RIGHT or key == pygame.K_d) and self.direction != "left": 
            self.direction = "right"

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
    model_trained = "snake_dqn30k.pth"
    model.load_state_dict(torch.load(model_trained))
    model.eval()

    pygame.init()
    pygame.display.set_caption("Snake Game But You're Not Controller")

    cell_size = 30
    x, y = 0, 0
    screen_size = env.grid_size * cell_size
    screen = pygame.display.set_mode((screen_size, screen_size))
    clock = pygame.time.Clock()

    player_score = 0
    snake_score = 0
    playing_count = 1

    WHITE = (255, 255, 255)
    GREY = (189, 189, 189)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    def font(size):
        return pygame.font.Font("Pixel Game.otf", size)

    score_font = font(80)
    text_font = font(60)
    bye_font = font(90)
    winner_font = font(110)
    # --- DEFAULT COLORS ---
    ai_color = WHITE
    player_color = WHITE
    bye_color = WHITE

    on_score = False
    drawing = False
    state = "menu_scene"

    running = True
    while running:
        screen.fill(BLACK)
        score = score_font.render(str(env.score), True, WHITE)

        # Draw apple
        pygame.draw.rect(screen, RED,
                         (env.apple[0] * cell_size, env.apple[1] * cell_size, cell_size, cell_size))
        # Draw snake
        for seg in env.snake:
            pygame.draw.rect(screen, GREEN,
                             (seg[0] * cell_size, seg[1] * cell_size, cell_size, cell_size))

        if state == "menu_scene":
            mouse_pos = pygame.mouse.get_pos()
            pos_x, pos_y = mouse_pos
            screen.fill(BLACK)

            # indent = 0
            # text1 = "Even quitting can be a power move."
            # wrapped = textwrap.wrap(text1, width=20)
            # y = 50
            # for line in wrapped:
            #     img = bye_font.render(line, True, (255,255,255))
            #     screen.blit(img, (80 + indent, y))
            #     y += bye_font.get_height()
            #     indent += 120

            # for i in range(screen_size):
            #     pygame.draw.line(screen, WHITE, (0, cell_size*i), (screen_size, cell_size*i))
            #     pygame.draw.line(screen, WHITE, (cell_size*i, 0), (cell_size*i, screen_size))

            # --- BUTTON AREAS ---
            ai_btn     = pygame.Rect(240, 420, 180, 100)
            player_btn = pygame.Rect(450, 420, 180, 100)
            bye_btn    = pygame.Rect(290, 550, 300, 100)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.done = True

                if ai_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                    ai_color = GREY
                elif ai_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONUP:
                    state = "ai_scene"
                    ai_color = WHITE
                elif not ai_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONUP:
                    ai_color = WHITE
                
                if playing_count > 1 and player_score > snake_score:
                    if player_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                        player_color = GREY
                    elif player_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONUP:
                        player_color = WHITE
                    elif not player_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONUP:
                        player_color = WHITE
                else:
                    if player_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                        player_color = GREY
                    elif player_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONUP:
                        state = "player_scene"
                        player_color = WHITE
                    elif not player_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONUP:
                        player_color = WHITE

                if bye_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONDOWN:
                    bye_color = GREY
                elif bye_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONUP:
                    state = "bye_scene"
                elif not bye_btn.collidepoint(mouse_pos) and event.type == pygame.MOUSEBUTTONUP:
                    bye_color = WHITE

            pygame.draw.rect(screen, ai_color, ai_btn)
            pygame.draw.rect(screen, player_color, player_btn)
            pygame.draw.rect(screen, bye_color, bye_btn)

            coordinates = text_font.render(f"{pos_x} {pos_y}", True, WHITE)

            if on_score:
                player_score_text = text_font.render(f"Player's score: {str(player_score)}", True, WHITE)
                snake_score_text = text_font.render(f"Snake's score: {str(snake_score)}", True, WHITE)
            else:
                player_score_text = text_font.render("", True, WHITE)
                snake_score_text = text_font.render("", True, WHITE)

            play_first = text_font.render("Who will play first?", True, WHITE)
            play_next = text_font.render("Who will play next?", True, WHITE)
            ai = text_font.render(f"Not me!", True, BLACK)
            player = text_font.render(f"Me!", True, BLACK)
            another_choice = text_font.render(f"Not today!", True, BLACK)

            blit_sequence = [
                # (coordinates, (50, 50)),
                (player_score_text, (50, 50)),
                (snake_score_text, (50, 100)),
                (ai, (250, 440)),
                (player, (510, 440)),
                (another_choice, (320, 570)),
            ]
            screen.blits(blit_sequence)

            if playing_count > 1:
                screen.blit(play_next, (200, 300))
            else:
                screen.blit(play_first, (200, 300))

            pygame.display.flip()

        elif state == "ai_scene":
            if env.score > snake_score: snake_score = env.score

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

            clock.tick(env.speed)

        elif state == "player_scene":
            if env.score > player_score: player_score = env.score

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.done = True
                if event.type == pygame.KEYDOWN:
                    env.control(event.key)

            env.step(-1)

            screen.blit(score, (screen_size/2 - 10, 100))
            pygame.display.flip()
            
            clock.tick(env.speed)
        
        elif state == "bye_scene":
            screen.fill(BLACK)

            all_phrases = [
                # Bỏ cuộc đôi khi cũng là một lựa chọn tốt!
                "Sometimes walking away is the smartest move.",
                "Even quitting can be a power move.",

                # Làm tốt lắm!
                "Nice work.",
                "Well done.",
                "Great job.",

                # Lựa chọn tốt / hay đấy!
                "Good choice.",
                "Smart move.",
                "Not bad at all.",
                "I see you. Nice choice."
            ]
            
            choosed = choice(all_phrases)
            heheh = bye_font.render(f"{choosed}", True, WHITE)
            if choosed == all_phrases[0] or choosed == all_phrases[1]:
                new_line = True
                x = 30
            elif choosed == all_phrases[-1]:
                new_line = True
                x = 220
            else:
                new_line = False
                x = 220
            
            indent = 0
            if new_line:
                wrapped = textwrap.wrap(choosed, width=22)
                y = 360
                for line in wrapped:
                    img = bye_font.render(line, True, (255,255,255))
                    screen.blit(img, (x + indent, y))
                    y += bye_font.get_height()
                    indent += 100
            else:
                screen.blit(heheh, (300, 360))

            pygame.display.flip()
            pygame.event.pump()
            pygame.time.wait(2000)

            running = False
            env.done = True

        elif state == "score_scene":
            print(f"Player's score: {player_score}")
            print(f"Snake's score: {snake_score}")
            screen.fill(BLACK)
            game_over = bye_font.render("Game Over!", True, WHITE)

            if playing_count > 2:
                if player_score > snake_score:
                    if player_score >= 8:
                        model_trained = "snake_dqn50k.pth"
                        model.load_state_dict(torch.load(model_trained))

                    player_wins = [
                        "Great job… finally!",
                        "Well done, about time.",
                        "Victory at last!",
                        "You actually did it!",
                        "Impressive… for once."
                    ]

                    choosed = choice(player_wins)
                    score_text = text_font.render(
                        f"Your score ({player_score}) > Snake'score ({snake_score})", True, WHITE
                    )
                    text = winner_font.render(f"{player_wins}", True, WHITE)
                    indent = 80
                    x = 80
                    new_line = True

                elif player_score < snake_score:
                    bot_taunts = [
                        "For someone this good, losing to a bot is wild.",
                        "You lost to a bot. Crazy.",
                        "You weren’t stupid… the bot was just smarter today.",
                        "Not calling you an idiot, but… that loss didn’t help.",
                        "You really got bot-diffed.",
                    ]

                    choosed = choice(bot_taunts)
                    score_text = text_font.render(
                        f"Your score ({player_score}) < Snake'score ({snake_score})", True, WHITE
                    )
                    text = winner_font.render(f"{choosed}", True, WHITE)
                    indent = 0
                    x = 80
                    new_line = True

                else:
                    bot_ties = [
                        "Can't even beat a bot?",
                        "A tie with a bot… impressive?",
                        "Wow, just a tie with the bot.",
                        "Bot didn’t even need to win, huh?",
                        "Holding the bot to a tie? Bold move."
                    ]
                        
                    choosed = choice(bot_ties)
                    score_text = text_font.render(
                        f"Your score ({player_score}) = Snake'score ({snake_score})", True, WHITE
                    )
                    text = winner_font.render(f"{bot_ties}", True, WHITE)
                    indent = 40
                    x = 80
                    new_line = True
                
                if new_line:
                    wrapped = textwrap.wrap(choosed, width=22)
                    indent = 0
                    y = 360
                    for line in wrapped:
                        img = bye_font.render(line, True, (255,255,255))
                        screen.blit(img, (x + indent, y))
                        y += bye_font.get_height()
                        indent += 60
                
                    screen.blit(score_text, (80, 200))
                
            else:
                screen.blit(game_over, (300, 400))

            pygame.display.flip()
            pygame.event.pump()
            pygame.time.wait(3000)

            state = "menu_scene"
            
        if env.done == True:
            playing_count += 1
            on_score = True
            state = "score_scene"
            env.done = False
            env.reset()
        
    pygame.quit()
    print("Game Over! Score:", env.score)


# ================= Main =================
if __name__ == "__main__":
    # choice = input("Train AI (t) or Play AI (p)? ")
    # if choice.lower() == "t":
    #     train_snake(episodes=2000)  # bạn chỉnh số episodes nếu muốn
    # elif choice.lower() == "p":
    #     play_ai_render()
    # else:
    #     print("Invalid choice")
    
    play_ai_render()
