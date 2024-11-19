import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))  # One large window
pygame.display.set_caption("Simulated Two Windows")

# Define areas for the two "windows"
window1 = screen.subsurface((0, 0, 400, 600))  # Left half
window2 = screen.subsurface((400, 0, 400, 600))  # Right half

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Draw on the first "window"
    window1.fill((255, 0, 0))
    pygame.draw.circle(window1, (0, 255, 0), (200, 300), 50)

    # Draw on the second
    # Draw on the second "window"
    window2.fill((0, 0, 255))
    pygame.draw.rect(window2, (255, 255, 0), (100, 200, 200, 100))

    # Update the entire screen
    pygame.display.flip()

pygame.quit()
