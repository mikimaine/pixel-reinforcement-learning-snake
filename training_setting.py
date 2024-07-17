import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QFormLayout, QPushButton, )

from agent import train


class GameSettings:
    def __init__(self, ):
        self.MAX_MEMORY = 100
        self.SPEED = 20

class SettingScreen(QWidget):
    def __init__(self, game_settings, start_game_callback):
        super().__init__()
        self.game_settings = game_settings
        self.start_game_callback = start_game_callback
        self.ui()
    def ui(self):
        layout = QFormLayout()
        self.max_memory_input = QLineEdit(str(self.game_settings.MAX_MEMORY))
        self.speed_input = QLineEdit(str(self.game_settings.SPEED))

        layout.addRow('Max Memory Size', self.max_memory_input)
        layout.addRow('Speed', self.speed_input)

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_game)
        layout.addWidget(self.start_button)

        self.setLayout(layout)
        self.setWindowTitle('Game Settings')
        self.show()

    def start_game(self):
        self.game_settings.MAX_MEMORY = int(self.max_memory_input.text())
        self.game_settings.SPEED = int(self.speed_input.text())

        self.start_game_callback(self.game_settings)


def start_game(game_settings):
    print(f'Game settings: {game_settings}')
    train()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    game_settings = GameSettings()
    ap = SettingScreen(game_settings, start_game)
    sys.exit(app.exec_())
