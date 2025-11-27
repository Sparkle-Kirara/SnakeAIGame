import pygame
from sympy.physics.units import speed

pygame.init()

WIDTH, HEIGHT = 800, 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))

player = [WIDTH/2, HEIGHT/2]
player_width = 50
player_height = 50
speed = 5

BLACK = (0, 0, 0)
CYAN = (0, 255, 255)


running = True
while running:
    screen.fill(BLACK)

    player[0] = max(0, min(player[0], WIDTH - player_width))
    player[1] = max(0, min(player[1], HEIGHT - player_height))

    pygame.draw.rect(screen, CYAN, (player[0], player[1], player_width, player_height))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        player[0] -= speed
    if keys[pygame.K_RIGHT]:
        player[0] += speed
    if keys[pygame.K_UP]:
        player[1] -= speed
    if keys[pygame.K_DOWN]:
        player[1] += speed

    pygame.display.flip()

pygame.quit()
