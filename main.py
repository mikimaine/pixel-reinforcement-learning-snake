import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QFormLayout, QPushButton

from trainingagent import train

class GameSettings:
    def __init__(self):
        self.settings = {
            'speed': 20,
            'state_size': 11,
            'hidden_layer_size': 256,
            'action_size': 3,
            'max_memory': 100,
            'epsilon_decay': 80,
            'learning_rate': 0.001,
            'batch_size': 1000,
            'gamma': 0.9
        }

class SettingScreen(QWidget):
    def __init__(self, settings, start_game_callback):
        super().__init__()
        self.game_settings = settings
        self.start_game_callback = start_game_callback
        self.input_fields = {}
        self.ui()

    def ui(self):
        layout = QFormLayout()

        for setting, value in self.game_settings.settings.items():
            input_field = QLineEdit(str(value))
            self.input_fields[setting] = input_field
            layout.addRow(f'{setting.replace("_", " ").title()}', input_field)

        self.train_agent_button = QPushButton('Train Agent', self)
        self.train_agent_button.clicked.connect(self.train_agent)
        layout.addWidget(self.train_agent_button)

        self.setLayout(layout)
        self.setWindowTitle('Game Settings')
        self.show()

    def train_agent(self):
        for setting, input_field in self.input_fields.items():
            try:
                self.game_settings.settings[setting] = int(input_field.text())
            except ValueError:
                input_field.setText(str(self.game_settings.settings[setting]))  # Reset to the last valid value
                continue

        self.start_game_callback(self.game_settings)

def train_agent(game_settings):
    print(f'Game settings: {game_settings.settings}')
    train()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game_settings = GameSettings()
    ap = SettingScreen(game_settings, train_agent)
    sys.exit(app.exec_())
