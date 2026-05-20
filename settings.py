# -*- coding: utf-8 -*-
from pygame.font import SysFont
from pygame import init
init()
# ==================================

#Window Settings
XWIN, YWIN = 600,600 #                Resolution
HALF_XWIN,HALF_YWIN = XWIN/2,YWIN/2 # Center
DISPLAY = (XWIN,YWIN)
FLAGS = 0 #                           Fullscreen, resizeable... 
FPS = 60 #                            Render frame rate

# Colors
BLACK = (0,0,0)
WHITE = (255,255,255)
GRAY = (100,100,100)
LIGHT_GREEN = (131,252,107)
ANDROID_GREEN = (164,198,57)
FOREST_GREEN = (87,189,68)

# Player
PLAYER_SIZE = (25,35)
PLAYER_COLOR = ANDROID_GREEN
PLAYER_MAX_SPEED = 20
PLAYER_JUMPFORCE = 20
PLAYER_BONUS_JUMPFORCE = 70
GRAVITY = .98

# Platforms
PLATFORM_COLOR = FOREST_GREEN
PLATFORM_COLOR_LIGHT = LIGHT_GREEN
PLATFORM_SIZE = (100,10)
PLATFORM_DISTANCE_GAP = (20,100)
MAX_PLATFORM_NUMBER = 50
BONUS_SPAWN_CHANCE = 0 #10
BREAKABLE_PLATFORM_CHANCE = 0 #12

# Fonts
LARGE_FONT = SysFont("",128)
SMALL_FONT = SysFont("arial",24)